using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>SeedVR2 causal video VAE encoder: <c>conv_in → 4 down stages (2 resnets + downsampler on 0–2)
/// → mid → per-frame GN+SiLU → conv_out</c> emitting <c>[B,32,T',H/8,W/8]</c> = mean‖logvar (no quant conv;
/// <c>use_quant_conv: False</c>). The reference SAMPLES the posterior at inference (<c>use_sample: True</c>);
/// <see cref="Encode"/> returns mean and logvar so the pipeline injects seeded noise itself. Latent scaling
/// (×0.9152, channels-last rearrange) is the pipeline's job, not the VAE's.</summary>
public sealed class SeedVr2VaeEncoder
{
    private readonly SeedVr2VaeConfig _config;
    private readonly SeedVr2VaeResnetBlock3d[][] _resnets;
    private readonly SeedVr2VaeDownsample3d?[] _downsamplers;
    private readonly SeedVr2VaeMidBlock3d _mid;
    private CausalConv3d _convIn = null!, _convOut = null!;
    private Tensor _normOutW = null!, _normOutB = null!;

    /// <summary>Creates the encoder graph for <paramref name="config"/>; weights via <see cref="LoadWeights"/>.</summary>
    public SeedVr2VaeEncoder(SeedVr2VaeConfig config)
    {
        _config = config;
        int stages = config.BlockOutChannels.Length;
        _resnets = new SeedVr2VaeResnetBlock3d[stages][];
        _downsamplers = new SeedVr2VaeDownsample3d?[stages];
        for (int s = 0; s < stages; s++)
        {
            _resnets[s] = new SeedVr2VaeResnetBlock3d[config.LayersPerBlock];
            for (int r = 0; r < config.LayersPerBlock; r++)
                _resnets[s][r] = new SeedVr2VaeResnetBlock3d(config.NormGroups, config.NormEps, config.ActivationDType);
            if (config.StageHasDownsample(s))
                _downsamplers[s] = new SeedVr2VaeDownsample3d(config.StageDownsamplesTime(s), config.ActivationDType);
        }
        _mid = new SeedVr2VaeMidBlock3d(config.BlockOutChannels[^1], config.NormGroups, config.NormEps, config.ActivationDType);
    }

    /// <summary>Latent channels (16).</summary>
    public int LatentChannels => _config.LatentChannels;

    /// <summary>Config this encoder was built for.</summary>
    public SeedVr2VaeConfig Config => _config;

    /// <summary>Binds diffusers-style <c>encoder.*</c> keys (verbatim checkpoint names).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _convIn = SeedVr2VaeOps.Conv(weights, "encoder.conv_in", computeDtype: _config.ActivationDType);
        for (int s = 0; s < _resnets.Length; s++)
        {
            for (int r = 0; r < _resnets[s].Length; r++)
                _resnets[s][r].LoadWeights(weights, $"encoder.down_blocks.{s}.resnets.{r}");
            _downsamplers[s]?.LoadWeights(weights, $"encoder.down_blocks.{s}.downsamplers.0");
        }
        _mid.LoadWeights(weights, "encoder.mid_block");
        _normOutW = weights["encoder.conv_norm_out.weight"];
        _normOutB = weights["encoder.conv_norm_out.bias"];
        _convOut = SeedVr2VaeOps.Conv(weights, "encoder.conv_out", computeDtype: _config.ActivationDType);
    }

    /// <summary>Encodes <c>[B,3,F,H,W]</c> F32 in [-1,1] to (mean, logvar), each <c>[B,16,T',H/8,W/8]</c>.
    /// Requires <c>(F−1) % 4 == 0</c> and H,W divisible by 8.</summary>
    public (Tensor Mean, Tensor LogVar) Encode(IBackend backend, Tensor pixels)
    {
        if (pixels.Shape.Rank != 5 || pixels.Shape[1] != 3)
            throw new HartsyInferenceException($"Encoder expects [B,3,F,H,W], got {pixels.Shape}.");
        int f = (int)pixels.Shape[2];
        if (f != 1 && (f - 1) % _config.TemporalCompression != 0)
            throw new HartsyInferenceException($"Frame count {f} violates (F-1) % {_config.TemporalCompression} == 0.");

        // BF16 mode: cast at the pixel boundary only — the caller keeps its F32 tensor either way.
        Tensor entry = DtypeCastHelper.EnsureDtype(backend, pixels, _config.ActivationDType, disposeSourceOnCast: false);
        Tensor h = _convIn.Forward(backend, entry);
        if (!ReferenceEquals(entry, pixels))
            entry.Dispose();
        for (int s = 0; s < _resnets.Length; s++)
        {
            foreach (SeedVr2VaeResnetBlock3d resnet in _resnets[s])
                h = Advance(backend, h, resnet.Forward);
            if (_downsamplers[s] is not null)
                h = Advance(backend, h, _downsamplers[s]!.Forward);
        }
        h = Advance(backend, h, _mid.Forward);
        h = Advance(backend, h, (b, x) => SeedVr2VaeOps.NormSiluPerFrame(b, x, _normOutW, _normOutB, _config.NormGroups, _config.NormEps));
        Tensor moments = _convOut.Forward(backend, h);
        h.Dispose();
        moments = DtypeCastHelper.EnsureF32(backend, moments);

        int c = _config.LatentChannels;
        (Tensor mean, Tensor logvar) = (SliceChannels(backend, moments, 0, c), SliceChannels(backend, moments, c, c));
        moments.Dispose();
        return (mean, logvar);
    }

    /// <summary>Runs one stage and disposes the previous activation — the whole-clip fp32 stages are ~1 GB
    /// each at 720p-area; retaining them serially OOMs the activation cache (caught by the first CUDA run).</summary>
    private static Tensor Advance(IBackend backend, Tensor previous, Func<IBackend, Tensor, Tensor> stage)
    {
        Tensor next = stage(backend, previous);
        previous.Dispose();
        return next;
    }

    private static unsafe Tensor SliceChannels(IBackend backend, Tensor x, int start, int count)
    {
        backend.Sync();
        int b = (int)x.Shape[0], cIn = (int)x.Shape[1], t = (int)x.Shape[2], hh = (int)x.Shape[3], ww = (int)x.Shape[4];
        Tensor output = new Tensor(new TensorShape([(long)b, count, t, hh, ww]), DType.F32);
        long plane = (long)t * hh * ww;
        float* src = (float*)x.DataPointer;
        float* dst = (float*)output.DataPointer;
        for (int bi = 0; bi < b; bi++)
            Buffer.MemoryCopy(
                src + ((long)bi * cIn + start) * plane, dst + (long)bi * count * plane,
                count * plane * sizeof(float), count * plane * sizeof(float));
        return output;
    }

    /// <summary>Weights for backend preload/free.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        for (int s = 0; s < _resnets.Length; s++)
        {
            foreach (SeedVr2VaeResnetBlock3d resnet in _resnets[s])
                foreach (Tensor t in resnet.EnumerateWeights()) yield return t;
            if (_downsamplers[s] is not null)
                foreach (Tensor t in _downsamplers[s]!.EnumerateWeights()) yield return t;
        }
        foreach (Tensor t in _mid.EnumerateWeights()) yield return t;
        yield return _normOutW;
        yield return _normOutB;
        foreach (Tensor t in _convOut.EnumerateWeights()) yield return t;
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>SeedVR2 causal video VAE decoder: <c>conv_in (16→512) → mid → 4 up stages (3 resnets each,
/// pixel-shuffle upsampler on 0–2) → per-frame GN+SiLU → conv_out (128→3)</c>. No post-quant conv
/// (<c>use_post_quant_conv: False</c>). Channel widths run reversed ([512,512,256,128]); upsamplers 0–1
/// double time (T→2T−1 each), giving F = 4·(T_lat−1)+1. Latent un-scaling (÷0.9152) is the pipeline's job.</summary>
public sealed class SeedVr2VaeDecoder
{
    private readonly SeedVr2VaeConfig _config;
    private readonly SeedVr2VaeResnetBlock3d[][] _resnets;
    private readonly SeedVr2VaeUpsampler3d?[] _upsamplers;
    private readonly SeedVr2VaeMidBlock3d _mid;
    private CausalConv3d _convIn = null!, _convOut = null!;
    private Tensor _normOutW = null!, _normOutB = null!;

    /// <summary>Creates the decoder graph for <paramref name="config"/>; weights via <see cref="LoadWeights"/>.</summary>
    public SeedVr2VaeDecoder(SeedVr2VaeConfig config)
    {
        _config = config;
        int stages = config.BlockOutChannels.Length;
        _resnets = new SeedVr2VaeResnetBlock3d[stages][];
        _upsamplers = new SeedVr2VaeUpsampler3d?[stages];
        for (int s = 0; s < stages; s++)
        {
            _resnets[s] = new SeedVr2VaeResnetBlock3d[config.LayersPerBlock + 1];
            for (int r = 0; r < _resnets[s].Length; r++)
                _resnets[s][r] = new SeedVr2VaeResnetBlock3d(config.NormGroups, config.NormEps);
            if (config.StageHasUpsample(s))
                _upsamplers[s] = new SeedVr2VaeUpsampler3d(config.StageUpsamplesTime(s));
        }
        _mid = new SeedVr2VaeMidBlock3d(config.BlockOutChannels[^1], config.NormGroups, config.NormEps);
    }

    /// <summary>Config this decoder was built for.</summary>
    public SeedVr2VaeConfig Config => _config;

    /// <summary>Binds diffusers-style <c>decoder.*</c> keys (verbatim checkpoint names).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _convIn = SeedVr2VaeOps.Conv(weights, "decoder.conv_in");
        _mid.LoadWeights(weights, "decoder.mid_block");
        for (int s = 0; s < _resnets.Length; s++)
        {
            for (int r = 0; r < _resnets[s].Length; r++)
                _resnets[s][r].LoadWeights(weights, $"decoder.up_blocks.{s}.resnets.{r}");
            _upsamplers[s]?.LoadWeights(weights, $"decoder.up_blocks.{s}.upsamplers.0");
        }
        _normOutW = weights["decoder.conv_norm_out.weight"];
        _normOutB = weights["decoder.conv_norm_out.bias"];
        _convOut = SeedVr2VaeOps.Conv(weights, "decoder.conv_out");
    }

    /// <summary>Decodes <c>[B,16,T',h,w]</c> F32 (already un-scaled) to <c>[B,3,F,h·8,w·8]</c> in [-1,1] range
    /// (unclamped — the pipeline clamps at the pixel boundary).</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        if (latent.Shape.Rank != 5 || latent.Shape[1] != _config.LatentChannels)
            throw new HartsyInferenceException($"Decoder expects [B,{_config.LatentChannels},T,h,w], got {latent.Shape}.");

        Tensor h = _convIn.Forward(backend, latent);
        h = Advance(backend, h, _mid.Forward);
        for (int s = 0; s < _resnets.Length; s++)
        {
            foreach (SeedVr2VaeResnetBlock3d resnet in _resnets[s])
                h = Advance(backend, h, resnet.Forward);
            if (_upsamplers[s] is not null)
                h = Advance(backend, h, _upsamplers[s]!.Forward);
        }
        h = Advance(backend, h, (b, x) => SeedVr2VaeOps.NormSiluPerFrame(b, x, _normOutW, _normOutB, _config.NormGroups, _config.NormEps));
        Tensor output = _convOut.Forward(backend, h);
        h.Dispose();
        return output;
    }

    /// <summary>Runs one stage and disposes the previous activation (VRAM discipline; see encoder note).</summary>
    private static Tensor Advance(IBackend backend, Tensor previous, Func<IBackend, Tensor, Tensor> stage)
    {
        Tensor next = stage(backend, previous);
        previous.Dispose();
        return next;
    }

    /// <summary>Weights for backend preload/free.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _mid.EnumerateWeights()) yield return t;
        for (int s = 0; s < _resnets.Length; s++)
        {
            foreach (SeedVr2VaeResnetBlock3d resnet in _resnets[s])
                foreach (Tensor t in resnet.EnumerateWeights()) yield return t;
            if (_upsamplers[s] is not null)
                foreach (Tensor t in _upsamplers[s]!.EnumerateWeights()) yield return t;
        }
        yield return _normOutW;
        yield return _normOutB;
        foreach (Tensor t in _convOut.EnumerateWeights()) yield return t;
    }
}

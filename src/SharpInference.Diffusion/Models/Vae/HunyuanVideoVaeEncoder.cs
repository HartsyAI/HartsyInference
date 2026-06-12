using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>HunyuanVideo 3D causal VAE encoder (<c>AutoencoderKLHunyuanVideo</c> encode path), needed for
/// Kandinsky 5.0 I2V first-frame conditioning. Mirrors <see cref="HunyuanVideoVaeDecoder"/>:
/// <c>conv_in(3→128) → 4 down stages (2 resnets each; downsamplers: stage 0 spatial-only,
/// stages 1–2 spatial+temporal, stage 3 none) → mid → GroupNorm+SiLU → conv_out(512→32, mean+logvar) →
/// quant_conv(32→32)</c>. <see cref="Encode"/> returns the posterior <b>mean</b> (deterministic mode) —
/// the raw VAE-space latent; the caller applies the scaling factor. Numerics validation-pending.</summary>
public sealed unsafe class HunyuanVideoVaeEncoder
{
    private readonly HunyuanVideoVaeConfig _config;

    private CausalConv3d? _convIn;
    private DownStage[] _stages = [];
    private HunyuanVideoMidBlock3d? _mid;
    private Tensor? _normOutWeight, _normOutBias;
    private CausalConv3d? _convOut;
    private CausalConv3d? _quantConv;

    private sealed class DownStage
    {
        public required HunyuanVideoResnetBlock3d[] Resnets;
        public CausalConv3d? DownsampleConv;
    }

    public HunyuanVideoVaeEncoder(HunyuanVideoVaeConfig? config = null)
    {
        _config = config ?? HunyuanVideoVaeConfig.Default;
    }

    /// <summary>The configuration this encoder was built with.</summary>
    public HunyuanVideoVaeConfig Config => _config;

    /// <summary>Loads weights keyed <c>encoder.*</c> + <c>quant_conv.*</c> (diffusers state-dict names).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int[] channels = _config.BlockOutChannels;

        _convIn = HunyuanVideoVaeKeys.Conv(w, "encoder.conv_in", padT: 1, padH: 1, padW: 1);

        int numStages = channels.Length;
        _stages = new DownStage[numStages];
        int prevOut = channels[0];
        for (int i = 0; i < numStages; i++)
        {
            int outCh = channels[i];
            HunyuanVideoResnetBlock3d[] resnets = new HunyuanVideoResnetBlock3d[_config.LayersPerBlock];
            int cur = prevOut;
            for (int j = 0; j < _config.LayersPerBlock; j++)
            {
                resnets[j] = new HunyuanVideoResnetBlock3d(cur, outCh, _config.NormGroups, _config.NormEps);
                resnets[j].LoadWeights(w, $"encoder.down_blocks.{i}.resnets.{j}");
                cur = outCh;
            }

            bool spatial = _config.StageHasSpatialResample(i);
            bool temporal = _config.StageHasTemporalResample(i);
            DownStage stage = new DownStage { Resnets = resnets };
            if (spatial || temporal)
                stage.DownsampleConv = HunyuanVideoVaeKeys.Conv(w, $"encoder.down_blocks.{i}.downsamplers.0.conv",
                    padT: 1, padH: 1, padW: 1,
                    strideT: temporal ? 2 : 1, strideH: spatial ? 2 : 1, strideW: spatial ? 2 : 1);
            _stages[i] = stage;
            prevOut = outCh;
        }

        _mid = new HunyuanVideoMidBlock3d(channels[^1], _config.MidBlockAttention, _config.NormGroups, _config.NormEps);
        _mid.LoadWeights(w, "encoder.mid_block");

        _normOutWeight = w["encoder.conv_norm_out.weight"];
        _normOutBias   = w["encoder.conv_norm_out.bias"];
        _convOut = HunyuanVideoVaeKeys.Conv(w, "encoder.conv_out", padT: 1, padH: 1, padW: 1);
        _quantConv = new CausalConv3d(w["quant_conv.weight"], Bias(w, "quant_conv.bias"));
    }

    /// <summary>Enumerates all weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convIn is not null) foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        foreach (DownStage s in _stages)
        {
            foreach (HunyuanVideoResnetBlock3d r in s.Resnets)
                foreach (Tensor t in r.EnumerateWeights()) yield return t;
            if (s.DownsampleConv is not null) foreach (Tensor t in s.DownsampleConv.EnumerateWeights()) yield return t;
        }
        if (_mid is not null) foreach (Tensor t in _mid.EnumerateWeights()) yield return t;
        if (_normOutWeight is not null) yield return _normOutWeight;
        if (_normOutBias is not null) yield return _normOutBias;
        if (_convOut is not null) foreach (Tensor t in _convOut.EnumerateWeights()) yield return t;
        if (_quantConv is not null) foreach (Tensor t in _quantConv.EnumerateWeights()) yield return t;
    }

    /// <summary>Encodes RGB <c>[B, 3, F, H, W]</c> in [−1, 1] (with <c>(F−1) % 4 == 0</c>) to the posterior mean <c>[B, 16, (F−1)/4+1, H/8, W/8]</c> (raw VAE space — multiply by the scaling factor for model space).</summary>
    public Tensor Encode(IBackend backend, Tensor rgb)
    {
        if (rgb.Shape.Rank != 5 || (int)rgb.Shape[1] != _config.InChannels)
            throw new ArgumentException($"Expected RGB [B, {_config.InChannels}, F, H, W]; got {rgb.Shape}.", nameof(rgb));
        int frames = (int)rgb.Shape[2];
        if ((frames - 1) % _config.TemporalCompression != 0)
            throw new ArgumentException(
                $"(frames−1) must be divisible by {_config.TemporalCompression}; got {frames}.", nameof(rgb));

        Tensor cur = _convIn!.Forward(backend, rgb);
        foreach (DownStage stage in _stages)
        {
            foreach (HunyuanVideoResnetBlock3d resnet in stage.Resnets)
            {
                Tensor next = resnet.Forward(backend, cur);
                cur.Dispose();
                cur = next;
            }
            if (stage.DownsampleConv is not null)
            {
                Tensor down = stage.DownsampleConv.Forward(backend, cur);
                cur.Dispose();
                cur = down;
            }
        }

        Tensor mid = _mid!.Forward(backend, cur);
        cur.Dispose();

        Tensor normed = HunyuanVideoVaeKeys.GroupNormSilu3d(backend, mid, _normOutWeight!, _normOutBias!,
            _config.NormGroups, _config.NormEps);
        mid.Dispose();

        Tensor moments = _convOut!.Forward(backend, normed);   // [B, 32, T', H', W'] = mean + logvar
        normed.Dispose();
        Tensor quant = _quantConv!.Forward(backend, moments);
        moments.Dispose();

        Tensor mean = SliceChannels(quant, 0, _config.LatentChannels);
        quant.Dispose();
        return mean;
    }

    /// <summary>Encodes a single interleaved-RGB24 frame to the raw-VAE-space mean latent <c>[1, 16, 1, H/8, W/8]</c> — the Kandinsky 5 I2V conditioning entry. The caller owns the returned tensor.</summary>
    public Tensor EncodeRgbFrame(IBackend backend, ReadOnlySpan<byte> rgb24, int width, int height)
    {
        if (rgb24.Length < width * height * 3)
            throw new ArgumentException($"expected {width * height * 3} bytes; got {rgb24.Length}.", nameof(rgb24));
        Tensor rgb = new Tensor(new TensorShape([1L, 3, 1, height, width]), DType.F32);
        float* p = (float*)rgb.DataPointer;
        long frame = (long)height * width;
        for (long pix = 0; pix < frame; pix++)
            for (int c = 0; c < 3; c++)
                p[c * frame + pix] = rgb24[(int)(pix * 3 + c)] / 127.5f - 1f;
        try
        {
            return Encode(backend, rgb);
        }
        finally
        {
            rgb.Dispose();
        }
    }

    private static Tensor SliceChannels(Tensor x, int start, int count)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor o = new Tensor(new TensorShape([(long)b, count, t, h, w]), DType.F32);
        long per = (long)t * h * w;
        float* sp = (float*)x.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int bi = 0; bi < b; bi++)
            Buffer.MemoryCopy(
                sp + ((long)bi * c + start) * per,
                op + (long)bi * count * per,
                count * per * 4, count * per * 4);
        return o;
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? b) ? b : null;
}

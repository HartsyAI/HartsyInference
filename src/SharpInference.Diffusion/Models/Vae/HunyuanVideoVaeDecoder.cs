using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>HunyuanVideo 3D causal VAE decoder (<c>AutoencoderKLHunyuanVideo</c> decode path), used by
/// Kandinsky 5.0 T2V/I2V. Assembled from the shared primitives: <see cref="CausalConv3d"/> (replicate
/// padding mode), <see cref="HunyuanVideoResnetBlock3d"/>, <see cref="HunyuanVideoMidBlock3d"/>
/// (frame-causal attention via <see cref="VaeAttention.Forward3D"/>), and <see cref="Vae3dLayout"/>.
///
/// <para>Decode: <c>post_quant_conv(16→16) → conv_in(16→512) → mid → 4 up stages (3 resnets each;
/// upsamplers: stage 0 spatial-only, stages 1–2 spatial+temporal, stage 3 none) → GroupNorm+SiLU →
/// conv_out(128→3)</c>. The upsampler treats frame 0 specially (spatial-only nearest), so
/// <c>T → 1 + 2·(T−1)</c> per temporal stage and <c>F = (T_lat−1)·4 + 1</c> overall. Latent scaling
/// (÷0.476986) is the caller's responsibility. <b>Untiled</b> — large clips need the diffusers-style
/// tiling follow-up before GPU-scale decodes; numerics are validation-pending vs the real checkpoint.</para></summary>
public sealed class HunyuanVideoVaeDecoder
{
    private readonly HunyuanVideoVaeConfig _config;

    private CausalConv3d? _postQuantConv;
    private CausalConv3d? _convIn;
    private HunyuanVideoMidBlock3d? _mid;
    private UpStage[] _stages = [];
    private Tensor? _normOutWeight, _normOutBias;
    private CausalConv3d? _convOut;

    private sealed class UpStage
    {
        public required HunyuanVideoResnetBlock3d[] Resnets;
        public required bool Spatial;
        public required bool Temporal;
        public CausalConv3d? UpsampleConv;
    }

    public HunyuanVideoVaeDecoder(HunyuanVideoVaeConfig? config = null)
    {
        _config = config ?? HunyuanVideoVaeConfig.Default;
    }

    /// <summary>The configuration this decoder was built with.</summary>
    public HunyuanVideoVaeConfig Config => _config;

    /// <summary>Loads weights keyed <c>post_quant_conv.*</c> + <c>decoder.*</c> (diffusers state-dict names).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int[] channels = _config.BlockOutChannels;
        int last = channels[^1];

        _postQuantConv = new CausalConv3d(w["post_quant_conv.weight"], Bias(w, "post_quant_conv.bias"));
        _convIn = HunyuanVideoVaeKeys.Conv(w, "decoder.conv_in", padT: 1, padH: 1, padW: 1);

        _mid = new HunyuanVideoMidBlock3d(last, _config.MidBlockAttention, _config.NormGroups, _config.NormEps);
        _mid.LoadWeights(w, "decoder.mid_block");

        int numStages = channels.Length;
        int resnetsPerStage = _config.LayersPerBlock + 1;
        _stages = new UpStage[numStages];
        int prevOut = last;
        for (int i = 0; i < numStages; i++)
        {
            int outCh = channels[numStages - 1 - i];
            HunyuanVideoResnetBlock3d[] resnets = new HunyuanVideoResnetBlock3d[resnetsPerStage];
            int cur = prevOut;
            for (int j = 0; j < resnetsPerStage; j++)
            {
                resnets[j] = new HunyuanVideoResnetBlock3d(cur, outCh, _config.NormGroups, _config.NormEps);
                resnets[j].LoadWeights(w, $"decoder.up_blocks.{i}.resnets.{j}");
                cur = outCh;
            }

            bool spatial = _config.StageHasSpatialResample(i);
            bool temporal = _config.StageHasTemporalResample(i);
            UpStage stage = new UpStage { Resnets = resnets, Spatial = spatial, Temporal = temporal };
            if (spatial || temporal)
                stage.UpsampleConv = HunyuanVideoVaeKeys.Conv(w, $"decoder.up_blocks.{i}.upsamplers.0.conv",
                    padT: 1, padH: 1, padW: 1);
            _stages[i] = stage;
            prevOut = outCh;
        }

        _normOutWeight = w["decoder.conv_norm_out.weight"];
        _normOutBias   = w["decoder.conv_norm_out.bias"];
        _convOut = HunyuanVideoVaeKeys.Conv(w, "decoder.conv_out", padT: 1, padH: 1, padW: 1);
    }

    /// <summary>Enumerates all weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_postQuantConv is not null) foreach (Tensor t in _postQuantConv.EnumerateWeights()) yield return t;
        if (_convIn is not null) foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        if (_mid is not null) foreach (Tensor t in _mid.EnumerateWeights()) yield return t;
        foreach (UpStage s in _stages)
        {
            foreach (HunyuanVideoResnetBlock3d r in s.Resnets)
                foreach (Tensor t in r.EnumerateWeights()) yield return t;
            if (s.UpsampleConv is not null) foreach (Tensor t in s.UpsampleConv.EnumerateWeights()) yield return t;
        }
        if (_normOutWeight is not null) yield return _normOutWeight;
        if (_normOutBias is not null) yield return _normOutBias;
        if (_convOut is not null) foreach (Tensor t in _convOut.EnumerateWeights()) yield return t;
    }

    /// <summary>Decodes a raw VAE-space latent <c>[B, 16, T_lat, H_lat, W_lat]</c> (already divided by the scaling factor) → RGB <c>[B, 3, (T_lat−1)·4+1, H_lat·8, W_lat·8]</c> in [−1, 1].</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        if (latent.Shape.Rank != 5 || (int)latent.Shape[1] != _config.LatentChannels)
            throw new ArgumentException(
                $"Expected latent [B, {_config.LatentChannels}, T, H, W]; got {latent.Shape}.", nameof(latent));

        Tensor x = _postQuantConv!.Forward(backend, latent);
        Tensor h = _convIn!.Forward(backend, x);
        x.Dispose();

        Tensor cur = _mid!.Forward(backend, h);
        h.Dispose();

        foreach (UpStage stage in _stages)
        {
            foreach (HunyuanVideoResnetBlock3d resnet in stage.Resnets)
            {
                Tensor next = resnet.Forward(backend, cur);
                cur.Dispose();
                cur = next;
            }
            if (stage.UpsampleConv is not null)
            {
                Tensor up = Upsample(backend, cur, stage.Temporal, stage.Spatial, stage.UpsampleConv);
                cur.Dispose();
                cur = up;
            }
        }

        Tensor normed = HunyuanVideoVaeKeys.GroupNormSilu3d(backend, cur, _normOutWeight!, _normOutBias!,
            _config.NormGroups, _config.NormEps);
        cur.Dispose();
        Tensor rgb = _convOut!.Forward(backend, normed);
        normed.Dispose();
        return rgb;
    }

    /// <summary>HunyuanVideo causal upsampler: frame 0 is spatially nearest-×2 only; frames 1..T−1 get the full nearest interpolation (incl. temporal ×2 when <paramref name="temporal"/>), then a k3 causal conv.</summary>
    private static Tensor Upsample(IBackend backend, Tensor x, bool temporal, bool spatial, CausalConv3d conv)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];

        Tensor spatialUp;
        if (spatial)
        {
            Tensor frames = Vae3dLayout.ToFrames(x, b, c, t, h, w);
            Tensor upFrames = new Tensor(new TensorShape(b * t, c, h * 2, w * 2), DType.F32);
            backend.UpsampleNearest2D(upFrames, frames, 2, 2);
            frames.Dispose();
            spatialUp = Vae3dLayout.FromFrames(upFrames, b, c, t, h * 2, w * 2);
            upFrames.Dispose();
        }
        else
        {
            spatialUp = Vae3dLayout.SliceFrames(x, 0, t);   // contiguous copy so we can dispose uniformly
        }

        Tensor interp = spatialUp;
        if (temporal && t > 1)
        {
            List<Tensor> parts = new(1 + 2 * (t - 1));
            Tensor first = Vae3dLayout.SliceFrames(spatialUp, 0, 1);
            parts.Add(first);
            for (int ti = 1; ti < t; ti++)
            {
                Tensor frame = Vae3dLayout.SliceFrames(spatialUp, ti, 1);
                parts.Add(frame);
                parts.Add(frame);   // nearest temporal ×2 = each later frame repeated twice
            }
            interp = Vae3dLayout.ConcatFrames(parts);
            // Repeated frames appear twice in the list; dispose each distinct tensor once.
            first.Dispose();
            for (int pi = 1; pi < parts.Count; pi += 2) parts[pi].Dispose();
            spatialUp.Dispose();
        }

        Tensor output = conv.Forward(backend, interp);
        interp.Dispose();
        return output;
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? b) ? b : null;
}

using System.Linq;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>LTX-2 video VAE decoder (<c>LTX2VideoDecoder3d</c> in <c>AutoencoderKLLTX2Video</c>), ported from
/// diffusers for the LTX-2 config: latent <c>[1, 128, T, H, W]</c> → per-channel latent un-normalization →
/// <c>conv_in</c> → mid resnets → 3 up-blocks → parameter-free channel-RMS <c>norm_out</c> → SiLU →
/// <c>conv_out</c> → spatial pixel-unshuffle (patch 4) → RGB <c>[1, 3, (T−1)·8+1, H·32, W·32]</c>.
///
/// <para>Distinct from LTX v1 (<see cref="LtxVideoVaeDecoder"/>): (1) the redesigned 128-ch latent ships real
/// per-channel <c>latents_mean</c>/<c>latents_std</c> (must un-normalize before decode); (2) up-blocks have
/// NO <c>conv_in</c> — the channel reduction happens inside the upsampler, which takes <c>out·upscale</c>
/// input (verified: per-block <c>in==out</c>, <c>output_channel = block_out[i] / upscale</c>); (3)
/// <c>upscale_factor = 2</c>, <c>upsample_residual = true</c>. Reuses
/// <see cref="LtxVaeResnetBlock3d"/>/<see cref="LtxVaeUpsampler3d"/>/<see cref="CausalConv3d"/>.</para>
///
/// <para>Keys (verified from <c>ltx-2.3-22b-dev.safetensors</c> header, under the <c>vae.</c> prefix the
/// loader strips): <c>decoder.conv_in.conv.*</c>, <c>decoder.mid_block.resnets.{j}.*</c>,
/// <c>decoder.up_blocks.{i}.upsamplers.0.conv.conv.*</c>, <c>decoder.up_blocks.{i}.resnets.{j}.*</c>,
/// <c>decoder.conv_out.conv.*</c>. All resnet/out RMS norms are parameter-free (no affine keys), eps 1e-8.
/// <c>decoder_causal=false</c>. Numerics vs the real checkpoint are validation-pending.</para></summary>
public sealed unsafe class LtxVideo2VaeDecoder
{
    private sealed class UpStage
    {
        public LtxVaeResnetBlock3d? ConvIn;      // present only if a block's nominal in != out (not for LTX-2)
        public LtxVaeUpsampler3d? Upsampler;     // spatio-temporal ×2, channel reduce by upscale
        public LtxVaeResnetBlock3d[] Resnets = [];
    }

    private readonly int _latentChannels;
    private readonly int _outChannels;
    private readonly int _patch;
    private readonly bool _isCausal;
    private readonly int[] _blockOutRev;         // reversed block_out_channels
    private readonly bool[] _scalingRev;         // reversed spatio_temporal_scaling
    private readonly int[] _layersRev;           // reversed layers_per_block (mid + up-blocks)
    private readonly int[] _upscaleRev;          // reversed upsample_factor
    private readonly bool[] _residualRev;        // reversed upsample_residual
    private readonly float[]? _latentsMean;      // [latentChannels], pre-cast to F32
    private readonly float[]? _latentsStd;       // [latentChannels], pre-cast to F32
    // Activation width for the whole decode. BF16 halves the peak transient set (a handful of full-output-grid
    // tensors), which is what decides whether the DiT's resident prefix has to be evicted to make room. Never
    // F16 — a VAE's resnet activations exceed F16's 65504 (the SDXL rule in VaePrecisionHelper).
    private readonly DType _computeDtype;

    private CausalConv3d? _convIn, _convOut;
    private LtxVaeResnetBlock3d[] _midResnets = [];
    private UpStage[] _upStages = [];
    private bool[] _temporalScaleInferred = [];  // per up-stage: does it upscale the temporal axis (st0 == 2)?

    public LtxVideo2VaeDecoder(
        int latentChannels = 128, int outChannels = 3,
        int[]? blockOutChannels = null, bool[]? spatioTemporalScaling = null, int[]? layersPerBlock = null,
        int[]? upsampleFactor = null, bool[]? upsampleResidual = null,
        int patchSize = 4, bool isCausal = false,
        float[]? latentsMean = null, float[]? latentsStd = null, DType? computeDtype = null)
    {
        _computeDtype = computeDtype ?? DType.F32;
        _latentChannels = latentChannels;
        _outChannels = outChannels;
        _patch = patchSize;
        _isCausal = isCausal;
        blockOutChannels ??= [256, 512, 1024];
        spatioTemporalScaling ??= [true, true, true];
        layersPerBlock ??= [5, 5, 5, 5];
        upsampleFactor ??= [2, 2, 2];
        upsampleResidual ??= [true, true, true];
        _blockOutRev = Reverse(blockOutChannels);
        _scalingRev = Reverse(spatioTemporalScaling);
        _layersRev = Reverse(layersPerBlock);
        _upscaleRev = Reverse(upsampleFactor);
        _residualRev = Reverse(upsampleResidual);
        _latentsMean = latentsMean;
        _latentsStd = latentsStd;
    }

    /// <summary>Output frames for a given latent frame count: each temporally-scaling up-stage maps T → T·2−1.</summary>
    public int OutputFrames(int latentFrames)
    {
        int t = latentFrames;
        foreach (bool s in _temporalScaleInferred) if (s) t = t * 2 - 1;
        return t;
    }

    /// <summary>Loads the decoder, INFERRING its full structure (number of mid/up resnets, per-stage channels, and
    /// each upsampler's stride + channel-upscale) from the weight shapes. This makes one decoder cover every LTX-2 VAE
    /// variant — the 19B (3 spatio-temporal up-stages) and the 22B (4 up-stages with mixed spatial/temporal/
    /// spatio-temporal upsamplers, irregular channels) — without the caller knowing the config. The constructor's
    /// block_out_channels/layers_per_block/etc. are ignored here in favour of the checkpoint's own shapes.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _convIn = new CausalConv3d(w["decoder.conv_in.conv.weight"], Bias(w, "decoder.conv_in.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal, spatialReflectPad: true, computeDtype: _computeDtype);
        int output = (int)w["decoder.conv_in.conv.weight"].Shape[0];   // mid / conv_in output channels

        int nMid = CountResnets(w, "decoder.mid_block");
        _midResnets = new LtxVaeResnetBlock3d[nMid];
        for (int j = 0; j < nMid; j++)
        {
            _midResnets[j] = new LtxVaeResnetBlock3d(output, output, timestepCond: false, isCausal: _isCausal, spatialReflectPad: true, computeDtype: _computeDtype);
            _midResnets[j].LoadWeights(w, $"decoder.mid_block.resnets.{j}");
        }

        List<UpStage> stages = [];
        List<bool> temporal = [];
        for (int i = 0; ; i++)
        {
            string p = $"decoder.up_blocks.{i}";
            bool hasUp = w.ContainsKey($"{p}.upsamplers.0.conv.conv.weight");
            int nRes = CountResnets(w, p);
            if (!hasUp && nRes == 0) break;                    // no more up-stages

            UpStage stage = new();
            int outNom;
            if (hasUp)
            {
                // Upsampler conv weight is [outNom·strideProd, output(=incoming)]; invert to recover the geometry.
                Tensor uw = w[$"{p}.upsamplers.0.conv.conv.weight"];
                int convOut = (int)uw.Shape[0], convIn = (int)uw.Shape[1];
                // Resnet channel (in==out) is the stage's nominal out; falls back to convIn/strideProd if no resnets.
                outNom = nRes > 0 ? (int)w[$"{p}.resnets.0.conv1.conv.weight"].Shape[0] : convIn;
                int strideProd = convOut / outNom;             // diffusers: convOut = outNom · ∏stride
                int upscale = convIn / outNom;                 // diffusers: convIn  = outNom · upscale
                (int T, int H, int W) stride = StrideFromProduct(strideProd);
                stage.Upsampler = new LtxVaeUpsampler3d(convIn, stride, upscaleFactor: upscale,
                    residual: true, isCausal: _isCausal, spatialReflectPad: true, computeDtype: _computeDtype);
                stage.Upsampler.LoadWeights(w, $"{p}.upsamplers.0");
                temporal.Add(stride.T == 2);
            }
            else
            {
                outNom = (int)w[$"{p}.resnets.0.conv1.conv.weight"].Shape[0];
                if (output != outNom)
                {
                    stage.ConvIn = new LtxVaeResnetBlock3d(output, outNom, timestepCond: false, isCausal: _isCausal, spatialReflectPad: true, computeDtype: _computeDtype);
                    stage.ConvIn.LoadWeights(w, $"{p}.conv_in");
                }
                temporal.Add(false);
            }

            stage.Resnets = new LtxVaeResnetBlock3d[nRes];
            for (int j = 0; j < nRes; j++)
            {
                stage.Resnets[j] = new LtxVaeResnetBlock3d(outNom, outNom, timestepCond: false, isCausal: _isCausal, spatialReflectPad: true, computeDtype: _computeDtype);
                stage.Resnets[j].LoadWeights(w, $"{p}.resnets.{j}");
            }
            stages.Add(stage);
            output = outNom;
        }
        _upStages = [.. stages];
        _temporalScaleInferred = [.. temporal];

        _convOut = new CausalConv3d(w["decoder.conv_out.conv.weight"], Bias(w, "decoder.conv_out.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal, spatialReflectPad: true, computeDtype: _computeDtype);
    }

    /// <summary>Counts the contiguous <c>{prefix}.resnets.{j}.conv1.conv.weight</c> entries present in the dict.</summary>
    private static int CountResnets(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        int n = 0;
        while (w.ContainsKey($"{prefix}.resnets.{n}.conv1.conv.weight")) n++;
        return n;
    }

    /// <summary>Maps an upsampler's stride product to its (T,H,W) stride, per diffusers <c>upsample_type</c>:
    /// 8 = spatio-temporal (2,2,2); 4 = spatial (1,2,2); 2 = temporal (2,1,1); 1 = identity.</summary>
    private static (int T, int H, int W) StrideFromProduct(int strideProd) => strideProd switch
    {
        8 => (2, 2, 2),
        4 => (1, 2, 2),
        2 => (2, 1, 1),
        1 => (1, 1, 1),
        _ => throw new NotSupportedException($"Unexpected LTX-2 VAE upsampler stride product {strideProd}."),
    };

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convIn is not null) foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        foreach (LtxVaeResnetBlock3d r in _midResnets) foreach (Tensor t in r.EnumerateWeights()) yield return t;
        foreach (UpStage s in _upStages)
        {
            if (s.ConvIn is not null) foreach (Tensor t in s.ConvIn.EnumerateWeights()) yield return t;
            if (s.Upsampler is not null) foreach (Tensor t in s.Upsampler.EnumerateWeights()) yield return t;
            foreach (LtxVaeResnetBlock3d r in s.Resnets) foreach (Tensor t in r.EnumerateWeights()) yield return t;
        }
        if (_convOut is not null) foreach (Tensor t in _convOut.EnumerateWeights()) yield return t;
    }

    /// <summary>Decodes a latent <c>[1, latentChannels, T_lat, H, W]</c> to RGB <c>[1, 3, (T_lat−1)·8+1,
    /// H·32, W·32]</c> in [-1,1]. Applies per-channel latent un-normalization (<c>z = latent·std + mean</c>)
    /// first when stats are present (diffusers <c>_denormalize_latents</c>, scaling_factor 1.0).</summary>
    /// <summary>Activation width the decode runs at. The caller sizes its VRAM reserve off this: the peak is a
    /// fixed number of full-output-grid tensors, so it scales linearly with the element width.</summary>
    public DType ComputeDtype => _computeDtype;

    public Tensor Decode(IBackend backend, Tensor latent)
    {
        if ((int)latent.Shape[1] != _latentChannels)
            throw new ArgumentException($"latent channels {latent.Shape[1]} != {_latentChannels}.", nameof(latent));

        bool probe = Environment.GetEnvironmentVariable("HARTSY_LTX2_PROBE") == "1";
        Tensor denorm = Denormalize(latent);
        if (probe) ProbeStats("denorm", denorm);
        // The latent is tiny (one stage-0 grid) so un-normalizing on the host stays cheap; widen to the decode's
        // activation dtype here so every stage below runs in it.
        if (_computeDtype != DType.F32)
        {
            Tensor cast = new Tensor(denorm.Shape, _computeDtype);
            backend.CastToBf16(cast, denorm);
            denorm.Dispose();
            denorm = cast;
        }
        Tensor h = _convIn!.Forward(backend, denorm);
        denorm.Dispose();
        if (probe) ProbeStats("conv_in", h);
        foreach (LtxVaeResnetBlock3d r in _midResnets) { Tensor n = r.Forward(backend, h, null); h.Dispose(); h = n; }
        if (probe) ProbeStats("mid_resnets", h);

        int stageIdx = 0;
        foreach (UpStage s in _upStages)
        {
            if (s.ConvIn is not null) { Tensor n = s.ConvIn.Forward(backend, h, null); h.Dispose(); h = n; }
            if (s.Upsampler is not null) { Tensor n = s.Upsampler.Forward(backend, h); h.Dispose(); h = n; }
            foreach (LtxVaeResnetBlock3d r in s.Resnets) { Tensor n = r.Forward(backend, h, null); h.Dispose(); h = n; }
            if (probe) ProbeStats($"up_stage_{stageIdx}", h);
            stageIdx++;
        }

        // norm_out + final pixel-unshuffle run on-device: the host loops here D2H-drained the full-res tensor
        // twice (the literal Krea2-VAE pattern). WanRmsNormChannel is the same channel-RMS math; UnpatchifyVae's
        // channel unpack (oc = c·p² + r·p + q, q→H, r→W) is exactly this decoder's (c·p + pa)·p + pb layout.
        Tensor normed = new Tensor(h.Shape, h.DType);
        backend.WanRmsNormChannel(normed, h, null, 1e-8f);
        h.Dispose();
        if (probe) ProbeStats("norm_out", normed);
        backend.Silu(normed, normed);
        Tensor patched = _convOut!.Forward(backend, normed);   // [1, outChannels·p², F, 8H, 8W]
        normed.Dispose();
        if (probe) ProbeStats("conv_out", patched);
        int pb = (int)patched.Shape[0], pc = (int)patched.Shape[1], pf = (int)patched.Shape[2];
        int ph = (int)patched.Shape[3], pw = (int)patched.Shape[4];
        int oc = pc / (_patch * _patch);
        Tensor rgb = new Tensor(new TensorShape([(long)pb, oc, pf, (long)ph * _patch, (long)pw * _patch]), patched.DType);
        backend.UnpatchifyVae(rgb, patched, _patch);
        patched.Dispose();
        // The decode's INTERNAL width is _computeDtype, but the returned RGB is always F32: every consumer
        // (VideoRgbFrames.ExtractFrame, ProbeStats, the restore chain) reads it as raw floats. Widening here
        // rather than earlier keeps the BF16 saving across the stages, where the big tensors are.
        if (rgb.DType != DType.F32)
        {
            Tensor rgbF32 = new Tensor(rgb.Shape, DType.F32);
            backend.CastToF32(rgbF32, rgb);
            rgb.Dispose();
            rgb = rgbF32;
        }
        if (probe) ProbeStats("unpatchify", rgb);
        return rgb;
    }

    private static unsafe void ProbeStats(string label, Tensor t)
    {
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        float* p = (float*)f32.DataPointer;
        long n = f32.ElementCount;
        float mn = float.MaxValue, mx = float.MinValue; double sum = 0; long nanCount = 0, infCount = 0;
        for (long e = 0; e < n; e++)
        {
            float v = p[e];
            if (float.IsNaN(v)) { nanCount++; continue; }
            if (float.IsInfinity(v)) { infCount++; continue; }
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        HartsyInference.Core.Logging.Logs.Warning(
            $"[ltx2-vae-probe] {label}: shape=[{string.Join(",", Enumerable.Range(0, t.Shape.Rank).Select(i => t.Shape[i]))}] min={mn:F4} max={mx:F4} mean={sum / n:F4} nan={nanCount} inf={infCount}");
        if (!ReferenceEquals(f32, t)) f32.Dispose();
    }

    /// <summary>Per-channel latent un-normalization <c>z[b,c,t,h,w] = latent·std[c] + mean[c]</c> over the
    /// 5-D latent. Returns a clone when no stats are configured.</summary>
    private Tensor Denormalize(Tensor latent)
    {
        Tensor latF32 = latent.DType == DType.F32 ? latent : latent.CastTo(DType.F32);
        if (_latentsMean is null || _latentsStd is null)
        {
            Tensor clone = new Tensor(latF32.Shape, DType.F32);
            Buffer.MemoryCopy((void*)latF32.DataPointer, (void*)clone.DataPointer,
                latF32.Shape.ElementCount * sizeof(float), latF32.Shape.ElementCount * sizeof(float));
            if (!ReferenceEquals(latF32, latent)) latF32.Dispose();
            return clone;
        }
        int b = (int)latF32.Shape[0], c = (int)latF32.Shape[1];
        long spatial = latF32.Shape.ElementCount / ((long)b * c);
        Tensor outT = new Tensor(latF32.Shape, DType.F32);
        float* xp = (float*)latF32.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                float std = _latentsStd[ci], mean = _latentsMean[ci];
                long basePos = ((long)bi * c + ci) * spatial;
                for (long s = 0; s < spatial; s++) op[basePos + s] = xp[basePos + s] * std + mean;
            }
        if (!ReferenceEquals(latF32, latent)) latF32.Dispose();
        return outT;
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? t : null;

    private static int[] Reverse(int[] a) { int[] r = (int[])a.Clone(); Array.Reverse(r); return r; }
    private static bool[] Reverse(bool[] a) { bool[] r = (bool[])a.Clone(); Array.Reverse(r); return r; }
}

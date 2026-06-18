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

    private CausalConv3d? _convIn, _convOut;
    private LtxVaeResnetBlock3d[] _midResnets = [];
    private UpStage[] _upStages = [];
    private int _lastChannel;                    // channels into conv_out

    public LtxVideo2VaeDecoder(
        int latentChannels = 128, int outChannels = 3,
        int[]? blockOutChannels = null, bool[]? spatioTemporalScaling = null, int[]? layersPerBlock = null,
        int[]? upsampleFactor = null, bool[]? upsampleResidual = null,
        int patchSize = 4, bool isCausal = false,
        float[]? latentsMean = null, float[]? latentsStd = null)
    {
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

    /// <summary>Output frames for a given latent frame count: <c>(T_lat − 1)·8 + 1</c> (3 temporal upsamplers ×2).</summary>
    public int OutputFrames(int latentFrames)
    {
        int t = latentFrames;
        foreach (bool s in _scalingRev) if (s) t = t * 2 - 1;
        return t;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int output = _blockOutRev[0];
        _convIn = new CausalConv3d(w["decoder.conv_in.conv.weight"], Bias(w, "decoder.conv_in.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal);

        _midResnets = new LtxVaeResnetBlock3d[_layersRev[0]];
        for (int j = 0; j < _midResnets.Length; j++)
        {
            _midResnets[j] = new LtxVaeResnetBlock3d(output, output, timestepCond: false, isCausal: _isCausal);
            _midResnets[j].LoadWeights(w, $"decoder.mid_block.resnets.{j}");
        }

        _upStages = new UpStage[_blockOutRev.Length];
        for (int i = 0; i < _blockOutRev.Length; i++)
        {
            int upscale = _upscaleRev[i];
            int inNom = output / upscale;             // nominal block in  (diffusers: output_channel // upscale)
            int outNom = _blockOutRev[i] / upscale;   // nominal block out (diffusers: block_out[i] // upscale)
            UpStage stage = new();
            string p = $"decoder.up_blocks.{i}";
            if (inNom != outNom)
            {
                // Not exercised by the LTX-2 checkpoint (in==out), but kept faithful to LTX2VideoUpBlock3d.
                stage.ConvIn = new LtxVaeResnetBlock3d(inNom, outNom, timestepCond: false, isCausal: _isCausal);
                stage.ConvIn.LoadWeights(w, $"{p}.conv_in");
            }
            if (_scalingRev[i])
            {
                // Upsampler in_channels = out·upscale (= the real incoming channel count); it reduces to outNom.
                stage.Upsampler = new LtxVaeUpsampler3d(outNom * upscale, (2, 2, 2),
                    upscaleFactor: upscale, residual: _residualRev[i], isCausal: _isCausal);
                stage.Upsampler.LoadWeights(w, $"{p}.upsamplers.0");
            }
            stage.Resnets = new LtxVaeResnetBlock3d[_layersRev[i + 1]];
            for (int j = 0; j < stage.Resnets.Length; j++)
            {
                stage.Resnets[j] = new LtxVaeResnetBlock3d(outNom, outNom, timestepCond: false, isCausal: _isCausal);
                stage.Resnets[j].LoadWeights(w, $"{p}.resnets.{j}");
            }
            _upStages[i] = stage;
            output = outNom;
        }

        _lastChannel = output;
        _convOut = new CausalConv3d(w["decoder.conv_out.conv.weight"], Bias(w, "decoder.conv_out.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal);
    }

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
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        if ((int)latent.Shape[1] != _latentChannels)
            throw new ArgumentException($"latent channels {latent.Shape[1]} != {_latentChannels}.", nameof(latent));

        Tensor denorm = Denormalize(latent);
        Tensor h = _convIn!.Forward(backend, denorm);
        denorm.Dispose();
        foreach (LtxVaeResnetBlock3d r in _midResnets) { Tensor n = r.Forward(backend, h, null); h.Dispose(); h = n; }

        foreach (UpStage s in _upStages)
        {
            if (s.ConvIn is not null) { Tensor n = s.ConvIn.Forward(backend, h, null); h.Dispose(); h = n; }
            if (s.Upsampler is not null) { Tensor n = s.Upsampler.Forward(backend, h); h.Dispose(); h = n; }
            foreach (LtxVaeResnetBlock3d r in s.Resnets) { Tensor n = r.Forward(backend, h, null); h.Dispose(); h = n; }
        }

        Tensor normed = ChannelRms(h, _lastChannel);
        h.Dispose();
        backend.Silu(normed, normed);
        Tensor patched = _convOut!.Forward(backend, normed);   // [1, outChannels·p², F, 8H, 8W]
        normed.Dispose();
        Tensor rgb = PixelUnshuffle(patched);
        patched.Dispose();
        return rgb;
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

    /// <summary>Spatial pixel-unshuffle (patch p): <c>[1, oc·p², F, H, W] → [1, oc, F, H·p, W·p]</c>
    /// (channel = <c>(c·p + p_a)·p + p_b</c>, p_b→H, p_a→W). Matches the upstream reshape/permute.</summary>
    private Tensor PixelUnshuffle(Tensor x)
    {
        int b = (int)x.Shape[0], srcC = (int)x.Shape[1], f = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int p = _patch;
        int oc = srcC / (p * p);
        int outH = h * p, outW = w * p;
        Tensor outT = new Tensor(new TensorShape([(long)b, oc, f, outH, outW]), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        long srcFrame = (long)h * w, dstFrame = (long)outH * outW;
        for (int bi = 0; bi < b; bi++)
            for (int c = 0; c < oc; c++)
                for (int fi = 0; fi < f; fi++)
                    for (int ho = 0; ho < outH; ho++)
                    {
                        int hi = ho / p, pb = ho % p;
                        for (int wo = 0; wo < outW; wo++)
                        {
                            int wi = wo / p, pa = wo % p;
                            int ch = (c * p + pa) * p + pb;
                            long srcOff = (((long)bi * srcC + ch) * f + fi) * srcFrame + (long)hi * w + wi;
                            long dstOff = (((long)bi * oc + c) * f + fi) * dstFrame + (long)ho * outW + wo;
                            op[dstOff] = sp[srcOff];
                        }
                    }
        return outT;
    }

    /// <summary>Parameter-free per-location channel RMS norm (eps 1e-8, matching LTX-2 <c>PerChannelRMSNorm</c>).</summary>
    private static Tensor ChannelRms(Tensor x, int c)
    {
        int b = (int)x.Shape[0];
        long spatial = x.Shape.ElementCount / ((long)b * c);
        Tensor outT = new Tensor(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (long s = 0; s < spatial; s++)
            {
                long basePos = (long)bi * c * spatial + s;
                double sum = 0;
                for (int ci = 0; ci < c; ci++) { float v = xp[basePos + (long)ci * spatial]; sum += (double)v * v; }
                float inv = 1f / MathF.Sqrt((float)(sum / c) + 1e-8f);
                for (int ci = 0; ci < c; ci++) { long off = basePos + (long)ci * spatial; op[off] = xp[off] * inv; }
            }
        return outT;
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? t : null;

    private static int[] Reverse(int[] a) { int[] r = (int[])a.Clone(); Array.Reverse(r); return r; }
    private static bool[] Reverse(bool[] a) { bool[] r = (bool[])a.Clone(); Array.Reverse(r); return r; }
}

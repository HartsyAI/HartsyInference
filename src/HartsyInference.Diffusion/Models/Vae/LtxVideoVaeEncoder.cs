using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>LTX-Video VAE encoder (<c>LTXVideoEncoder3d</c> in <c>AutoencoderKLLTXVideo</c>), ported from diffusers
/// for the base LTX-Video 0.9 config — the mirror image of <see cref="LtxVideoVaeDecoder"/> (Tier 3.4): spatial
/// pixel-shuffle-down (patch 4, <c>3→48</c> channels) → <c>conv_in</c> → down blocks (resnets at unchanged width →
/// optional strided downsample conv (spatio-temporal ×2, base-0.9 keeps channels unchanged) → optional
/// channel-change resnet) → mid resnets → channel-RMS <c>norm_out</c> → SiLU → <c>conv_out</c> (128+1 channels).
///
/// <para><b>Read from the actual diffusers source before writing this</b> (not assumed from decoder symmetry —
/// two things would have been wrong by analogy): (1) <c>AutoencoderKLLTXVideo</c> defaults
/// <c>encoder_causal=True</c>, <c>decoder_causal=False</c> — the encoder and decoder do NOT share the same
/// causal setting despite being the same checkpoint's two halves, confirmed against
/// <c>diffusers/models/autoencoders/autoencoder_kl_ltx.py</c>. (2) The base 0.9 downsampler
/// (<c>LTXVideoDownBlock3D</c>, not the 0.9.5+ reshape-based <c>LTXVideoDownsampler3d</c>) is a plain strided
/// <c>CausalConv3d</c> that keeps channels unchanged (<c>in_channels==out_channels</c>) — the channel INCREASE
/// per stage happens via a separate optional channel-change resnet (<c>conv_out</c> in the down-block, distinct
/// from the encoder's own top-level <c>conv_out</c>) that runs AFTER the downsampler, only when the stage's
/// target width differs from its input width. This is genuinely the mirror of the decoder's per-stage
/// <c>ConvIn</c> (channel-change resnet BEFORE the upsampler) — same idea, opposite order, because down-scaling
/// happens before the channel grows and up-scaling happens after it shrinks.</para>
///
/// <para>The 129th <c>conv_out</c> channel is diffusers' "uniform log-var" convention (a single shared
/// log-variance scalar, expanded back to a full 128-channel logvar for <c>DiagonalGaussianDistribution</c>
/// compatibility in the reference pipeline). This encoder is for deterministic image-conditioning latents
/// (mirroring every other <c>*VaeEncoder.EncodeRgbFrame</c>-shaped site in this codebase, e.g.
/// <c>Wan22VaeEncoder</c>), so <see cref="Encode"/> takes the mean (the first <see cref="_latentChannels"/>
/// channels) and drops the logvar tail — no posterior sampling.</para></summary>
public sealed unsafe class LtxVideoVaeEncoder
{
    private sealed class DownStage
    {
        public LtxVaeResnetBlock3d[] Resnets = [];
        public CausalConv3d? Downsampler;       // spatio-temporal ×2, channels unchanged (base 0.9)
        public LtxVaeResnetBlock3d? ConvOut;    // channel-change resnet, only when this stage's width changes
    }

    private readonly int _latentChannels;
    private readonly int _inChannels;
    private readonly int _patch;
    private readonly bool _isCausal;
    private readonly int[] _blockOut;           // natural order (NOT reversed — encoder narrows spatially, widens channel-wise, in forward order)
    private readonly bool[] _scaling;
    private readonly int[] _layers;

    private CausalConv3d? _convIn, _convOut;
    private DownStage[] _downStages = [];
    private LtxVaeResnetBlock3d[] _midResnets = [];
    private int _lastChannel;                   // channels into norm_out/conv_out (mid_block's width)

    private Tensor? _latentsMean, _latentsStd;

    public LtxVideoVaeEncoder(
        int latentChannels = 128, int inChannels = 3,
        int[]? blockOutChannels = null, bool[]? spatioTemporalScaling = null, int[]? layersPerBlock = null,
        int patchSize = 4, bool isCausal = true)
    {
        _latentChannels = latentChannels;
        _inChannels = inChannels;
        _patch = patchSize;
        _isCausal = isCausal;
        _blockOut = blockOutChannels ?? [128, 256, 512, 512];
        _scaling = spatioTemporalScaling ?? [true, true, true, false];
        _layers = layersPerBlock ?? [4, 3, 3, 3, 4];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int output = _blockOut[0];
        _convIn = new CausalConv3d(w["encoder.conv_in.conv.weight"], Bias(w, "encoder.conv_in.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal);

        int numStages = _blockOut.Length;
        _downStages = new DownStage[numStages];
        for (int i = 0; i < numStages; i++)
        {
            int input = output;
            int target = i + 1 < numStages ? _blockOut[i + 1] : _blockOut[i];
            DownStage stage = new();
            string p = $"encoder.down_blocks.{i}";
            stage.Resnets = new LtxVaeResnetBlock3d[_layers[i]];
            for (int j = 0; j < stage.Resnets.Length; j++)
            {
                stage.Resnets[j] = new LtxVaeResnetBlock3d(input, input, timestepCond: false, isCausal: _isCausal);
                stage.Resnets[j].LoadWeights(w, $"{p}.resnets.{j}");
            }
            if (_scaling[i])
            {
                stage.Downsampler = new CausalConv3d(
                    w[$"{p}.downsamplers.0.conv.weight"], Bias(w, $"{p}.downsamplers.0.conv.bias"),
                    strideT: 2, strideH: 2, strideW: 2,
                    padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal);
            }
            if (input != target)
            {
                stage.ConvOut = new LtxVaeResnetBlock3d(input, target, timestepCond: false, isCausal: _isCausal);
                stage.ConvOut.LoadWeights(w, $"{p}.conv_out");
            }
            _downStages[i] = stage;
            output = target;
        }

        _lastChannel = output;
        _midResnets = new LtxVaeResnetBlock3d[_layers[^1]];
        for (int j = 0; j < _midResnets.Length; j++)
        {
            _midResnets[j] = new LtxVaeResnetBlock3d(output, output, timestepCond: false, isCausal: _isCausal);
            _midResnets[j].LoadWeights(w, $"encoder.mid_block.resnets.{j}");
        }

        _convOut = new CausalConv3d(w["encoder.conv_out.conv.weight"], Bias(w, "encoder.conv_out.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal);

        if (w.TryGetValue("latents_mean", out Tensor? lm)) _latentsMean = lm.DType == DType.F32 ? lm : lm.CastTo(DType.F32);
        if (w.TryGetValue("latents_std", out Tensor? ls)) _latentsStd = ls.DType == DType.F32 ? ls : ls.CastTo(DType.F32);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convIn is not null) foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        foreach (DownStage s in _downStages)
        {
            foreach (LtxVaeResnetBlock3d r in s.Resnets) foreach (Tensor t in r.EnumerateWeights()) yield return t;
            if (s.Downsampler is not null) foreach (Tensor t in s.Downsampler.EnumerateWeights()) yield return t;
            if (s.ConvOut is not null) foreach (Tensor t in s.ConvOut.EnumerateWeights()) yield return t;
        }
        foreach (LtxVaeResnetBlock3d r in _midResnets) foreach (Tensor t in r.EnumerateWeights()) yield return t;
        if (_convOut is not null) foreach (Tensor t in _convOut.EnumerateWeights()) yield return t;
    }

    /// <summary>Encodes RGB <c>[1, 3, F, H, W]</c> in [-1,1] (H,W divisible by <c>patch·8</c> — 32 for the default
    /// patch=4/three ×2 downsamples) to a deterministic latent <c>[1, latentChannels, F', H/32, W/32]</c> (mean
    /// only — the logvar tail is dropped, see class doc). Normalizes per-channel when the checkpoint carries
    /// <c>latents_mean</c>/<c>latents_std</c> (identity otherwise, matching <see cref="LtxVideoVaeDecoder"/>'s
    /// own optional un-normalize).</summary>
    public Tensor Encode(IBackend backend, Tensor rgb)
    {
        if ((int)rgb.Shape[1] != _inChannels)
            throw new ArgumentException($"input channels {rgb.Shape[1]} != {_inChannels}.", nameof(rgb));

        Tensor patched = PixelShuffleDown(rgb);
        Tensor h = _convIn!.Forward(backend, patched);
        patched.Dispose();

        foreach (DownStage s in _downStages)
        {
            foreach (LtxVaeResnetBlock3d r in s.Resnets) { Tensor n = r.Forward(backend, h, null); h.Dispose(); h = n; }
            if (s.Downsampler is not null) { Tensor n = s.Downsampler.Forward(backend, h); h.Dispose(); h = n; }
            if (s.ConvOut is not null) { Tensor n = s.ConvOut.Forward(backend, h, null); h.Dispose(); h = n; }
        }
        foreach (LtxVaeResnetBlock3d r in _midResnets) { Tensor n = r.Forward(backend, h, null); h.Dispose(); h = n; }

        Tensor normed = ChannelRms(h, _lastChannel);
        h.Dispose();
        backend.Silu(normed, normed);
        Tensor raw = _convOut!.Forward(backend, normed);   // [1, latentChannels+1, F, H, W] (mean + shared logvar)
        normed.Dispose();

        Tensor latent = TakeMeanChannels(raw);
        raw.Dispose();
        Normalize(latent);
        return latent;
    }

    /// <summary>Per-channel latent normalization (in-place): <c>latent[c] = (raw[c] − mean[c]) / std[c]</c> — the
    /// inverse of <see cref="LtxVideoVaeDecoder"/>'s <c>Denormalize</c>. No-op when the checkpoint carries no
    /// stats.</summary>
    private void Normalize(Tensor latent)
    {
        if (_latentsMean is null || _latentsStd is null) return;
        int b = (int)latent.Shape[0];
        int c = (int)latent.Shape[1];
        long spatial = latent.Shape.ElementCount / ((long)b * c);
        float* lp = (float*)latent.DataPointer;
        float* mean = (float*)_latentsMean.DataPointer;
        float* std = (float*)_latentsStd.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                float m = mean[ci], s = std[ci];
                long basePos = ((long)bi * c + ci) * spatial;
                for (long i = 0; i < spatial; i++) lp[basePos + i] = (lp[basePos + i] - m) / s;
            }
    }

    /// <summary>Drops the trailing "uniform logvar" channel(s) from <c>conv_out</c>'s output, keeping only the
    /// first <see cref="_latentChannels"/> (the mean) — see class doc.</summary>
    private Tensor TakeMeanChannels(Tensor x)
    {
        int b = (int)x.Shape[0], f = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor outT = new Tensor(new TensorShape([(long)b, _latentChannels, f, h, w]), DType.F32);
        long srcSpatial = (long)f * h * w;
        int srcC = (int)x.Shape[1];
        float* sp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < _latentChannels; ci++)
                Buffer.MemoryCopy(
                    sp + ((long)bi * srcC + ci) * srcSpatial,
                    op + ((long)bi * _latentChannels + ci) * srcSpatial,
                    srcSpatial * sizeof(float), srcSpatial * sizeof(float));
        return outT;
    }

    /// <summary>Spatial pixel-shuffle-down (patch p): <c>[1, ic, F, H, W] → [1, ic·p², F, H/p, W/p]</c>. The exact
    /// inverse index mapping of <see cref="LtxVideoVaeDecoder"/>'s <c>PixelUnshuffle</c> (channel =
    /// <c>c·p² + Wrem·p + Hrem</c>) — derived from and verified against diffusers' own reshape/permute
    /// (<c>reshape(...).permute(0,1,3,7,5,2,4,6).flatten(1,4)</c>), not assumed to be the naive inverse.</summary>
    private Tensor PixelShuffleDown(Tensor x)
    {
        int b = (int)x.Shape[0], ic = (int)x.Shape[1], f = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int p = _patch;
        int outH = h / p, outW = w / p;
        int outC = ic * p * p;
        Tensor outT = new Tensor(new TensorShape([(long)b, outC, f, outH, outW]), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        long srcFrame = (long)h * w, dstFrame = (long)outH * outW;
        for (int bi = 0; bi < b; bi++)
            for (int co = 0; co < outC; co++)
            {
                int oc = co / (p * p), rem = co % (p * p);
                int wRem = rem / p, hRem = rem % p;
                for (int fi = 0; fi < f; fi++)
                    for (int ho = 0; ho < outH; ho++)
                    {
                        int hi = ho * p + hRem;
                        for (int wo = 0; wo < outW; wo++)
                        {
                            int wi = wo * p + wRem;
                            long srcOff = (((long)bi * ic + oc) * f + fi) * srcFrame + (long)hi * w + wi;
                            long dstOff = (((long)bi * outC + co) * f + fi) * dstFrame + (long)ho * outW + wo;
                            op[dstOff] = sp[srcOff];
                        }
                    }
            }
        return outT;
    }

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

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string k) => w.TryGetValue(k, out Tensor? b) ? b : null;
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>LTX-Video VAE decoder (<c>LTXVideoDecoder3d</c> in <c>AutoencoderKLLTXVideo</c>), ported from diffusers for the base LTX-Video 0.9 config: <c>conv_in</c> → mid resnets → up blocks (optional channel-change <c>conv_in</c> + pixel-shuffle upsampler + resnets) → channel-RMS <c>norm_out</c> → SiLU → <c>conv_out</c> → spatial pixel-unshuffle (patch 4). Reuses <see cref="LtxVaeResnetBlock3d"/>/<see cref="LtxVaeUpsampler3d"/>/<see cref="CausalConv3d"/>.
///
/// <para>The released 0.9 VAE config has <c>timestep_conditioning=False</c> and <c>decoder_causal=False</c> (non-causal, symmetric edge-replicate temporal padding) — so this is a plain decoder (no decode timestep, no <c>feat_cache</c>). Output frames = <c>(T_lat − 1)·8 + 1</c>; spatial = <c>latent · 32</c>. The 0.9.1+/13B variant is timestep-conditioned (<c>timestepConditioned: true</c>): each mid/up block owns a PixArt timestep embedder (4·C modulation per resnet, via <see cref="LtxVaeTimeEmbedder"/>) and the decoder norm_out adds a 2·C (shift, scale); the decode timestep (e.g. 0.05) is scaled ×1000 internally. Numerics vs the real checkpoint are validation-pending.</para></summary>
public sealed unsafe class LtxVideoVaeDecoder
{
    private sealed class UpStage
    {
        public LtxVaeResnetBlock3d? ConvIn;     // channel change (in != out)
        public LtxVaeUpsampler3d? Upsampler;    // spatio-temporal ×2
        public LtxVaeResnetBlock3d[] Resnets = [];
        public LtxVaeTimeEmbedder? TimeEmbedder;   // per-block PixArt embedder (timestep-conditioned), emits 4·outC
    }

    private readonly int _latentChannels;
    private readonly int _outChannels;
    private readonly int _patch;
    private readonly bool _isCausal;
    private readonly bool _timestepCond;
    private readonly int[] _blockOutRev;        // reversed block_out_channels
    private readonly bool[] _scalingRev;        // reversed spatio_temporal_scaling
    private readonly int[] _layersRev;          // reversed layers_per_block

    private CausalConv3d? _convIn, _convOut;
    private LtxVaeResnetBlock3d[] _midResnets = [];
    private LtxVaeTimeEmbedder? _midTimeEmbedder;   // mid_block PixArt embedder, emits 4·midChannel
    private UpStage[] _upStages = [];
    private int _lastChannel;                   // channels into conv_out

    // norm_out timestep conditioning (decoder-level): embedder emits 2·lastChannel, plus scale_shift_table[2, C].
    private LtxVaeTimeEmbedder? _normOutTimeEmbedder;
    private Tensor? _normOutScaleShift;          // [2, lastChannel]
    // Per-channel latent normalization stats (LTX trains the diffusion model on normalized latents:
    // (raw − mean)/std). The decoder must un-normalize before decode: raw = latent·std + mean. [latentChannels] each.
    private Tensor? _latentsMean, _latentsStd;
    // VALIDATION-PENDING: verify vs diffusers LTXPipeline 0.9.7 — timestep_scale_multiplier registered as 1000.0.
    private const float TimestepScaleMultiplier = 1000.0f;

    public LtxVideoVaeDecoder(
        int latentChannels = 128, int outChannels = 3,
        int[]? blockOutChannels = null, bool[]? spatioTemporalScaling = null, int[]? layersPerBlock = null,
        int patchSize = 4, bool isCausal = false, bool timestepConditioned = false)
    {
        _latentChannels = latentChannels;
        _outChannels = outChannels;
        _patch = patchSize;
        _isCausal = isCausal;
        _timestepCond = timestepConditioned;
        blockOutChannels ??= [128, 256, 512, 512];
        spatioTemporalScaling ??= [true, true, true, false];
        layersPerBlock ??= [4, 3, 3, 3, 4];
        _blockOutRev = Reverse(blockOutChannels);
        _scalingRev = Reverse(spatioTemporalScaling);
        _layersRev = Reverse(layersPerBlock);
    }

    /// <summary>Output frames for a given latent frame count: <c>(T_lat − 1)·8 + 1</c> (3 temporal upsamplers ×2 each).</summary>
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
            _midResnets[j] = new LtxVaeResnetBlock3d(output, output, timestepCond: _timestepCond, isCausal: _isCausal);
            _midResnets[j].LoadWeights(w, $"decoder.mid_block.resnets.{j}");
        }
        // VALIDATION-PENDING: verify vs diffusers LTXPipeline 0.9.7 — each mid/up block owns a
        // PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels·4); its 4·C output feeds every resnet in the block.
        if (_timestepCond)
            _midTimeEmbedder = LtxVaeTimeEmbedder.Load(w, "decoder.mid_block.time_embedder", output * 4);

        _upStages = new UpStage[_blockOutRev.Length];
        for (int i = 0; i < _blockOutRev.Length; i++)
        {
            int inC = output;
            int outC = _blockOutRev[i];
            UpStage stage = new();
            string p = $"decoder.up_blocks.{i}";
            if (inC != outC)
            {
                stage.ConvIn = new LtxVaeResnetBlock3d(inC, outC, timestepCond: _timestepCond, isCausal: _isCausal);
                stage.ConvIn.LoadWeights(w, $"{p}.conv_in");
            }
            if (_scalingRev[i])
            {
                stage.Upsampler = new LtxVaeUpsampler3d(outC, (2, 2, 2), upscaleFactor: 1, residual: false, isCausal: _isCausal);
                stage.Upsampler.LoadWeights(w, $"{p}.upsamplers.0");
            }
            stage.Resnets = new LtxVaeResnetBlock3d[_layersRev[i + 1]];
            for (int j = 0; j < stage.Resnets.Length; j++)
            {
                stage.Resnets[j] = new LtxVaeResnetBlock3d(outC, outC, timestepCond: _timestepCond, isCausal: _isCausal);
                stage.Resnets[j].LoadWeights(w, $"{p}.resnets.{j}");
            }
            // VALIDATION-PENDING: verify vs diffusers LTXPipeline 0.9.7 — the up block time_embedder is sized
            // PixArtAlphaCombinedTimestepSizeEmbeddings(in_channels·4) (the block's INCOMING channels). All the
            // block's resnets are out_channels-sized and consume this 4·inC temb, so a timestep-conditioned VAE
            // only stays dimensionally consistent where inC == outC (the real LTX TC VAE has no channel-changing
            // up blocks; channel reduction is done by the pixel-shuffle upsampler, so conv_in is absent here).
            if (_timestepCond)
            {
                if (inC != outC)
                    throw new NotSupportedException($"Timestep-conditioned LTX-Video VAE up block {i} changes channels ({inC}→{outC}); not supported (no real-weight config does this).");
                stage.TimeEmbedder = LtxVaeTimeEmbedder.Load(w, $"{p}.time_embedder", inC * 4);
            }
            _upStages[i] = stage;
            output = outC;
        }

        _lastChannel = output;
        // VALIDATION-PENDING: verify vs diffusers LTXPipeline 0.9.7 — decoder-level norm_out conditioning:
        // time_embedder emits 2·lastChannel, summed with scale_shift_table[2, lastChannel] → (shift, scale).
        if (_timestepCond)
        {
            _normOutTimeEmbedder = LtxVaeTimeEmbedder.Load(w, "decoder.time_embedder", output * 2);
            _normOutScaleShift = LoadF32(w, "decoder.scale_shift_table");
        }
        _convOut = new CausalConv3d(w["decoder.conv_out.conv.weight"], Bias(w, "decoder.conv_out.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal);

        // Latent normalization stats (optional — absent on synthetic tests). LTX names them
        // per_channel_statistics.mean-of-means / std-of-means; the converter renames to latents_mean/std.
        if (w.TryGetValue("latents_mean", out Tensor? lm)) _latentsMean = lm.DType == DType.F32 ? lm : lm.CastTo(DType.F32);
        if (w.TryGetValue("latents_std", out Tensor? ls)) _latentsStd = ls.DType == DType.F32 ? ls : ls.CastTo(DType.F32);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convIn is not null) foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        foreach (LtxVaeResnetBlock3d r in _midResnets) foreach (Tensor t in r.EnumerateWeights()) yield return t;
        if (_midTimeEmbedder is not null) foreach (Tensor t in _midTimeEmbedder.EnumerateWeights()) yield return t;
        foreach (UpStage s in _upStages)
        {
            if (s.ConvIn is not null) foreach (Tensor t in s.ConvIn.EnumerateWeights()) yield return t;
            if (s.Upsampler is not null) foreach (Tensor t in s.Upsampler.EnumerateWeights()) yield return t;
            foreach (LtxVaeResnetBlock3d r in s.Resnets) foreach (Tensor t in r.EnumerateWeights()) yield return t;
            if (s.TimeEmbedder is not null) foreach (Tensor t in s.TimeEmbedder.EnumerateWeights()) yield return t;
        }
        if (_normOutTimeEmbedder is not null) foreach (Tensor t in _normOutTimeEmbedder.EnumerateWeights()) yield return t;
        if (_normOutScaleShift is not null) yield return _normOutScaleShift;
        if (_convOut is not null) foreach (Tensor t in _convOut.EnumerateWeights()) yield return t;
    }

    /// <summary>Decodes a latent <c>[1, latentChannels, T_lat, H, W]</c> to RGB <c>[1, 3, (T_lat−1)·8+1, H·32, W·32]</c> in [-1,1]. (The base config has identity latent normalization; scaling_factor 1.0.)
    ///
    /// <para>For the timestep-conditioned VAE (0.9.1+/13B) pass <paramref name="decodeTimestep"/> (e.g. 0.05). The scalar is scaled by 1000 (<c>timestep_scale_multiplier</c>), then each mid/up block embeds it (PixArt embedder → 4·C modulation) and the decoder norm_out applies its own (shift, scale). For the base 0.9.0 VAE leave it null for plain decode.</para></summary>
    public Tensor Decode(IBackend backend, Tensor latent, float? decodeTimestep = null)
    {
        if ((int)latent.Shape[1] != _latentChannels)
            throw new ArgumentException($"latent channels {latent.Shape[1]} != {_latentChannels}.", nameof(latent));
        if (_timestepCond && decodeTimestep is null)
            throw new ArgumentException("This LTX-Video VAE is timestep-conditioned; a decode timestep is required.", nameof(decodeTimestep));

        // VALIDATION-PENDING: verify vs diffusers LTXPipeline 0.9.7 — temb = decode_timestep · 1000 feeds the
        // PixArt embedders; the un-embedded scalar (not the embedding) is what flows between blocks upstream.
        float scaledTimestep = (decodeTimestep ?? 0f) * TimestepScaleMultiplier;

        // Un-normalize: the diffusion model works in normalized latent space (raw−mean)/std, so
        // raw = latent·std + mean per channel. In-place on the caller's latent (disposed right after decode).
        Denormalize(latent);

        Tensor h = _convIn!.Forward(backend, latent);

        Tensor? midTemb = _midTimeEmbedder?.Embed(backend, scaledTimestep);
        foreach (LtxVaeResnetBlock3d r in _midResnets) { Tensor n = r.Forward(backend, h, midTemb); h.Dispose(); h = n; }
        midTemb?.Dispose();

        foreach (UpStage s in _upStages)
        {
            Tensor? upTemb = s.TimeEmbedder?.Embed(backend, scaledTimestep);
            if (s.ConvIn is not null) { Tensor n = s.ConvIn.Forward(backend, h, upTemb); h.Dispose(); h = n; }
            if (s.Upsampler is not null) { Tensor n = s.Upsampler.Forward(backend, h); h.Dispose(); h = n; }
            foreach (LtxVaeResnetBlock3d r in s.Resnets) { Tensor n = r.Forward(backend, h, upTemb); h.Dispose(); h = n; }
            upTemb?.Dispose();
        }

        Tensor normed = ChannelRms(h, _lastChannel);
        h.Dispose();
        if (_normOutTimeEmbedder is not null)
        {
            // VALIDATION-PENDING: verify vs diffusers LTXPipeline 0.9.7 — norm_out modulation:
            // (shift, scale) = embed(temb)[2,C] + scale_shift_table[2,C]; h = h·(1+scale) + shift.
            Tensor normTemb = _normOutTimeEmbedder.Embed(backend, scaledTimestep);
            ApplyNormOutModulation(normed, _lastChannel, normTemb, _normOutScaleShift!);
            normTemb.Dispose();
        }
        backend.Silu(normed, normed);
        Tensor patched = _convOut!.Forward(backend, normed);   // [1, outChannels·p², F, 8H, 8W]
        normed.Dispose();

        Tensor rgb = PixelUnshuffle(patched);
        patched.Dispose();
        return rgb;
    }

    /// <summary>Per-channel latent un-normalization (in-place): <c>latent[c] = latent[c]·std[c] + mean[c]</c>
    /// over <c>[1, C, T, H, W]</c>. No-op when the checkpoint carries no stats (identity normalization).</summary>
    private void Denormalize(Tensor latent)
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
                for (long i = 0; i < spatial; i++) lp[basePos + i] = lp[basePos + i] * s + m;
            }
    }

    /// <summary>norm_out (shift, scale) conditioning: <c>temb [2·C]</c> + <c>scale_shift_table [2, C]</c> → per-channel <c>x·(1+scale) + shift</c>.</summary>
    private static void ApplyNormOutModulation(Tensor x, int c, Tensor temb, Tensor scaleShift)
    {
        int b = (int)x.Shape[0];
        long spatial = x.Shape.ElementCount / ((long)b * c);
        float* xp = (float*)x.DataPointer;
        float* tp = (float*)temb.DataPointer;     // [2*C] = [shift(C), scale(C)]
        float* ss = (float*)scaleShift.DataPointer; // [2, C] = [shift(C), scale(C)]
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                float shift = tp[ci] + ss[ci];
                float scale = tp[c + ci] + ss[c + ci];
                long basePos = ((long)bi * c + ci) * spatial;
                float sc = 1f + scale;
                for (long s = 0; s < spatial; s++) xp[basePos + s] = xp[basePos + s] * sc + shift;
            }
    }

    /// <summary>Spatial pixel-unshuffle (patch p): <c>[1, oc·p², F, H, W] → [1, oc, F, H·p, W·p]</c>, matching the upstream reshape/permute (channel = <c>oc·p² + p_a·p + p_b</c>, with p_b→H, p_a→W).</summary>
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
                            int ch = oc > 0 ? (c * p + pa) * p + pb : 0;
                            long srcOff = (((long)bi * srcC + ch) * f + fi) * srcFrame + (long)hi * w + wi;
                            long dstOff = (((long)bi * oc + c) * f + fi) * dstFrame + (long)ho * outW + wo;
                            op[dstOff] = sp[srcOff];
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

    private static int[] Reverse(int[] a) { int[] r = (int[])a.Clone(); Array.Reverse(r); return r; }
    private static bool[] Reverse(bool[] a) { bool[] r = (bool[])a.Clone(); Array.Reverse(r); return r; }
    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string k) => w.TryGetValue(k, out Tensor? b) ? b : null;
    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string k) { Tensor t = w[k]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }
}

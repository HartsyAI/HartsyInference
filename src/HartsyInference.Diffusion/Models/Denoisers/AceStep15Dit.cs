using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Music;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>ACE-Step v1.5 turbo DiT (<c>AceStepDiTModel</c>, 2B) — flow-matching velocity head over 25 Hz / 64-ch
/// Oobleck latents. Structure verbatim from <c>modeling_acestep_v15_turbo.py</c>: Conv1d patchify
/// (<c>proj_in.1</c>, 192→2048, k2 s2; input = concat([src ‖ chunk_mask ‖ noisy], ch)), dual
/// <c>TimestepEmbedding</c> (sinusoid 256 ×1000 cos-first → 2048; the second one embeds <c>t − r</c>, turbo passes
/// r = t so it adds a constant), 24 Qwen3 blocks — GQA 16:8 q/k-norm self-attention with alternating
/// sliding(128)/full bidirectional masks and RoPE θ=1e6, un-modulated cross-attention to the packed condition
/// sequence (run through <c>condition_embedder</c> once), SwiGLU 6144 — each AdaLN-6-modulated by its own
/// <c>scale_shift_table</c> [1,6,2048] plus the shared <c>time_proj</c> output (NOT one global table; §5's
/// 19-keys-per-layer count includes the per-layer table), then RMSNorm <c>norm_out</c> AdaLN'd by the model-level
/// 2-chunk <c>scale_shift_table</c> + temb and ConvTranspose1d de-patchify (<c>proj_out.1</c>, 2048→64, k2 s2).
/// Numerically verified vs the f32 torch oracle on the real turbo checkpoint (velocity corr 1.0, maxAbs 5.1e-6 —
/// see <c>AceStep15DitParityTests</c>).</summary>
public sealed unsafe class AceStep15Dit : IDisposable
{
    private readonly AceStep15Config _config;
    private Layer[] _layers = [];
    private int _disposed;

    private Tensor? _projInW, _projInB, _projOutW, _projOutB;
    private Tensor? _tL1W, _tL1B, _tL2W, _tL2B, _tProjW, _tProjB;
    private Tensor? _rL1W, _rL1B, _rL2W, _rL2B, _rProjW, _rProjB;
    private Tensor? _condEmbW, _condEmbB, _normOutW, _globalScaleShift, _globalScaleShiftFlat;

    public AceStep15Dit(AceStep15Config config) => _config = config;

    public AceStep15Config Config => _config;

    /// <summary>Loads the <c>decoder.*</c> keys of the main v1.5 safetensors (§5 key map, header-verified).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _projInW = w["decoder.proj_in.1.weight"];
        _projInB = w["decoder.proj_in.1.bias"];
        _tL1W = w["decoder.time_embed.linear_1.weight"]; _tL1B = w["decoder.time_embed.linear_1.bias"];
        _tL2W = w["decoder.time_embed.linear_2.weight"]; _tL2B = w["decoder.time_embed.linear_2.bias"];
        _tProjW = w["decoder.time_embed.time_proj.weight"]; _tProjB = w["decoder.time_embed.time_proj.bias"];
        _rL1W = w["decoder.time_embed_r.linear_1.weight"]; _rL1B = w["decoder.time_embed_r.linear_1.bias"];
        _rL2W = w["decoder.time_embed_r.linear_2.weight"]; _rL2B = w["decoder.time_embed_r.linear_2.bias"];
        _rProjW = w["decoder.time_embed_r.time_proj.weight"]; _rProjB = w["decoder.time_embed_r.time_proj.bias"];
        _condEmbW = w["decoder.condition_embedder.weight"];
        _condEmbB = w["decoder.condition_embedder.bias"];

        _layers = new Layer[_config.NumLayers];
        for (int i = 0; i < _config.NumLayers; i++)
        {
            _layers[i] = new Layer(_config);
            _layers[i].LoadWeights(w, $"decoder.layers.{i}");
        }

        _normOutW = EnsureF32(w["decoder.norm_out.weight"]);
        _globalScaleShift = EnsureF32(w["decoder.scale_shift_table"]);
        // Flattened [1, 2·dim] view for the device SliceRows split in FinalLayer (host indexing i·dim unchanged).
        _globalScaleShiftFlat = _globalScaleShift.Reshape(new TensorShape(1, 2 * _config.HiddenSize));
        _projOutW = w["decoder.proj_out.1.weight"];
        _projOutB = w["decoder.proj_out.1.bias"];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _projInW, _projInB, _tL1W, _tL1B, _tL2W, _tL2B, _tProjW, _tProjB,
            _rL1W, _rL1B, _rL2W, _rL2B, _rProjW, _rProjB, _condEmbW, _condEmbB,
            _normOutW, _globalScaleShift, _projOutW, _projOutB })
            if (t is not null) yield return t;
        foreach (Layer l in _layers)
            foreach (Tensor t in l.EnumerateWeights()) yield return t;
    }

    /// <summary>Velocity prediction <c>v = noise − data</c>: noisy latents <c>[1, T, 64]</c> (T even) +
    /// <c>context_latents</c> <c>[1, T, 128]</c> (src ‖ chunk mask) + packed conditions <c>[1, L, 2048]</c> →
    /// <c>[1, T, 64]</c>. Timesteps are flow σ in [0, 1]; turbo passes <paramref name="timestepR"/> = t.</summary>
    public Tensor Forward(IBackend backend, Tensor noisy, Tensor contextLatents, Tensor conditions,
        float timestep, float timestepR)
    {
        int dim = _config.HiddenSize, latCh = _config.LatentChannels, patch = _config.PatchSize;
        int t = (int)noisy.Shape[1];
        if (noisy.Shape.Rank != 3 || (int)noisy.Shape[2] != latCh || t % patch != 0)
            throw new ArgumentException($"noisy latents must be [1, T·{patch}, {latCh}]; got {noisy.Shape}.", nameof(noisy));
        if (contextLatents.Shape.Rank != 3 || contextLatents.Shape[1] != t ||
            (int)contextLatents.Shape[2] != _config.InChannels - latCh)
            throw new ArgumentException($"context latents must be [1, {t}, {_config.InChannels - latCh}]; got {contextLatents.Shape}.",
                nameof(contextLatents));
        int s = t / patch;

        Tensor tokens = PatchEmbed(backend, noisy, contextLatents, t, s);

        (Tensor tembT, Tensor projT) = TimeEmbed(backend, timestep, _tL1W!, _tL1B!, _tL2W!, _tL2B!, _tProjW!, _tProjB!);
        (Tensor tembR, Tensor projR) = TimeEmbed(backend, timestep - timestepR, _rL1W!, _rL1B!, _rL2W!, _rL2B!, _rProjW!, _rProjB!);
        Tensor temb = new Tensor(tembT.Shape, DType.F32);
        backend.Add(temb, tembT, tembR);
        Tensor proj6 = new Tensor(projT.Shape, DType.F32);
        backend.Add(proj6, projT, projR);
        tembT.Dispose(); tembR.Dispose(); projT.Dispose(); projR.Dispose();

        int l = (int)conditions.Shape[1];
        Tensor cond = new Tensor(new TensorShape(1, l, dim), DType.F32);
        backend.Linear(cond, conditions, _condEmbW!, _condEmbB);

        (float[] cos, float[] sin) = AceStep15Attention.BuildRopeTables(s, _config.HeadDim, _config.RopeTheta);
        Tensor? slidingMask = s > _config.SlidingWindow + 1
            ? AceStep15Attention.BuildSlidingMask(s, _config.SlidingWindow) : null;

        // Device-resident layer chain: each layer returns a fresh tensor (no in-place host mutation), so the whole
        // stack stays on-device between GEMMs — the previous per-op DataPointer host loops synced every step.
        Tensor cur = tokens;
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, cur, cond, proj6, cos, sin,
                _config.IsSlidingLayer(i) ? slidingMask : null);
            cur.Dispose();
            cur = next;
        }
        slidingMask?.Dispose();
        cond.Dispose();
        proj6.Dispose();

        Tensor velocity = FinalLayer(backend, cur, temb, s, t);
        cur.Dispose();
        temb.Dispose();
        return velocity;
    }

    /// <summary>Channel-concat [src ‖ mask ‖ noisy] (context first, per the reference) → Conv1d k2 s2 → tokens
    /// <c>[1, S, 2048]</c> at 12.5 Hz.</summary>
    private Tensor PatchEmbed(IBackend backend, Tensor noisy, Tensor contextLatents, int t, int s)
    {
        int latCh = _config.LatentChannels, inCh = _config.InChannels, dim = _config.HiddenSize;
        int ctxCh = inCh - latCh;
        Tensor rows = new Tensor(new TensorShape(1, t, inCh), DType.F32);
        float* rp = (float*)rows.DataPointer;
        float* cp = (float*)contextLatents.DataPointer;
        float* np = (float*)noisy.DataPointer;
        for (int i = 0; i < t; i++)
        {
            Buffer.MemoryCopy(cp + (long)i * ctxCh, rp + (long)i * inCh, ctxCh * 4, ctxCh * 4);
            Buffer.MemoryCopy(np + (long)i * latCh, rp + (long)i * inCh + ctxCh, latCh * 4, latCh * 4);
        }

        Tensor channelsFirst = new Tensor(new TensorShape(1, inCh, t), DType.F32);
        backend.Transpose2D(channelsFirst, rows, t, inCh);
        rows.Dispose();
        Tensor conv = new Tensor(new TensorShape(1, dim, s), DType.F32);
        backend.Conv1d(conv, channelsFirst, _projInW!, _projInB, _config.PatchSize, 0, 0, 1, 1);
        channelsFirst.Dispose();
        Tensor tokens = new Tensor(new TensorShape(1, s, dim), DType.F32);
        backend.Transpose2D(tokens, conv, dim, s);
        conv.Dispose();
        return tokens;
    }

    /// <summary>One reference <c>TimestepEmbedding</c>: sinusoid(σ·1000, 256, cos-first) → linear_1 → SiLU →
    /// linear_2 = temb <c>[1, 2048]</c>; <c>time_proj(silu(temb))</c> = the AdaLN-6 vector <c>[1, 6·2048]</c>.
    /// Parameterized by weights so <c>time_embed</c> and <c>time_embed_r</c> share one implementation.</summary>
    private (Tensor Temb, Tensor Proj6) TimeEmbed(IBackend backend, float sigma,
        Tensor l1W, Tensor l1B, Tensor l2W, Tensor l2B, Tensor projW, Tensor projB)
    {
        int freq = _config.FreqDim, dim = _config.HiddenSize, half = freq / 2;
        float scaled = sigma * 1000f;
        Tensor sinusoid = new Tensor(new TensorShape(1, freq), DType.F32);
        float* sp = (float*)sinusoid.DataPointer;
        for (int i = 0; i < half; i++)
        {
            double f = Math.Exp(-Math.Log(10000.0) * i / half);
            sp[i] = (float)Math.Cos(scaled * f);
            sp[half + i] = (float)Math.Sin(scaled * f);
        }
        Tensor h1 = new Tensor(new TensorShape(1, dim), DType.F32);
        backend.Linear(h1, sinusoid, l1W, l1B);
        sinusoid.Dispose();
        backend.Silu(h1, h1);
        Tensor temb = new Tensor(new TensorShape(1, dim), DType.F32);
        backend.Linear(temb, h1, l2W, l2B);
        h1.Dispose();

        Tensor act = new Tensor(temb.Shape, DType.F32);
        backend.Silu(act, temb);
        Tensor proj6 = new Tensor(new TensorShape(1, 6 * dim), DType.F32);
        backend.Linear(proj6, act, projW, projB);
        act.Dispose();
        return (temb, proj6);
    }

    /// <summary>norm_out → AdaLN from the model-level table (chunk 2: shift, scale; both + temb) →
    /// ConvTranspose1d de-patchify back to <c>[1, T, 64]</c>.</summary>
    private Tensor FinalLayer(IBackend backend, Tensor tokens, Tensor temb, int s, int t)
    {
        int dim = _config.HiddenSize, latCh = _config.LatentChannels;
        Tensor normed = new Tensor(tokens.Shape, DType.F32);
        backend.RmsNorm(normed, tokens, _normOutW!, _config.RmsNormEps);
        // AdaLN from the model-level table (chunk 2: shift, scale), device-resident. Both shift and scale add the
        // SAME temb; out = normed·(1+scale)+shift. Bit-identical to the old host loop.
        Tensor g0 = new Tensor(new TensorShape(1, dim), DType.F32);
        Tensor g1 = new Tensor(new TensorShape(1, dim), DType.F32);
        backend.SliceRows(g0, _globalScaleShiftFlat!, 0);
        backend.SliceRows(g1, _globalScaleShiftFlat!, 1);
        Tensor shiftVec = new Tensor(new TensorShape(1, dim), DType.F32);
        Tensor scaleVec = new Tensor(new TensorShape(1, dim), DType.F32);
        backend.Add(shiftVec, g0, temb);
        backend.Add(scaleVec, g1, temb);
        g0.Dispose(); g1.Dispose();
        Tensor modded = DiTUtils.Modulate(backend, normed, shiftVec, scaleVec, normed.Shape);
        normed.Dispose(); shiftVec.Dispose(); scaleVec.Dispose();

        Tensor channelsFirst = new Tensor(new TensorShape(1, dim, s), DType.F32);
        backend.Transpose2D(channelsFirst, modded, s, dim);
        modded.Dispose();
        Tensor conv = new Tensor(new TensorShape(1, latCh, t), DType.F32);
        backend.ConvTranspose1d(conv, channelsFirst, _projOutW!, _projOutB, _config.PatchSize, 0, 0, 1, 1);
        channelsFirst.Dispose();
        Tensor output = new Tensor(new TensorShape(1, t, latCh), DType.F32);
        backend.Transpose2D(output, conv, latCh, t);
        conv.Dispose();
        return output;
    }

    private static Tensor EnsureF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) _layers = [];
        GC.SuppressFinalize(this);
    }

    /// <summary>One <c>AceStepDiTLayer</c>: AdaLN-6 (per-layer table + time_proj) gating the self-attention and the
    /// SwiGLU MLP; cross-attention to the conditions is plain pre-norm with an ungated residual.</summary>
    private sealed class Layer
    {
        private readonly AceStep15Config _c;
        private readonly AceStep15Attention _selfAttn;
        private readonly AceStep15Attention _crossAttn;
        private readonly SwiGluFfn _mlp;
        private Tensor? _scaleShift, _selfNorm, _crossNorm, _mlpNorm;

        public Layer(AceStep15Config c)
        {
            _c = c;
            _selfAttn = new AceStep15Attention(c);
            _crossAttn = new AceStep15Attention(c);
            _mlp = new SwiGluFfn(c.HiddenSize, c.IntermediateSize);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            // Flattened [1, 6·dim] for the device modulation split (SplitModulation); host indexing i·dim unchanged.
            _scaleShift = EnsureF32(w[$"{p}.scale_shift_table"]).Reshape(new TensorShape(1, 6 * _c.HiddenSize));
            _selfNorm = EnsureF32(w[$"{p}.self_attn_norm.weight"]);
            _crossNorm = EnsureF32(w[$"{p}.cross_attn_norm.weight"]);
            _mlpNorm = EnsureF32(w[$"{p}.mlp_norm.weight"]);
            _selfAttn.LoadWeights(w, $"{p}.self_attn");
            _crossAttn.LoadWeights(w, $"{p}.cross_attn");
            _mlp.LoadSwiGluWeights(
                w[$"{p}.mlp.gate_proj.weight"], null,
                w[$"{p}.mlp.up_proj.weight"], null,
                w[$"{p}.mlp.down_proj.weight"], null);
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor? t in new[] { _scaleShift, _selfNorm, _crossNorm, _mlpNorm })
                if (t is not null) yield return t;
            foreach (Tensor t in _selfAttn.EnumerateWeights()) yield return t;
            foreach (Tensor t in _crossAttn.EnumerateWeights()) yield return t;
            foreach (Tensor t in _mlp.EnumerateWeights()) yield return t;
        }

        /// <summary>Returns a fresh <c>[1, S, H]</c> tensor (input untouched) — device-resident: the AdaLN-6 split,
        /// the two (1+scale)·norm+shift affines, the two gated residuals and the ungated cross-attn add all run as
        /// IBackend ops (no per-op DataPointer host loops). Bit-identical F32 math to the old in-place host path.</summary>
        public Tensor Forward(IBackend backend, Tensor x, Tensor cond, Tensor proj6,
            float[] cos, float[] sin, Tensor? mask)
        {
            int s = (int)x.Shape[1], dim = _c.HiddenSize;
            DType act = x.DType;
            TensorShape shape = new TensorShape(1, s, dim);

            // 6 modulation vectors [1, dim] = scale_shift_table + proj6, split on-device:
            // [shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp].
            Tensor[] mod = SplitModulation(backend, proj6, dim);

            Tensor n1 = new Tensor(shape, act);
            backend.RmsNorm(n1, x, _selfNorm!, _c.RmsNormEps);
            Tensor preAttn = DiTUtils.Modulate(backend, n1, mod[0], mod[1], shape);   // (1+scale_sa)·n1 + shift_sa
            n1.Dispose();
            Tensor attn = _selfAttn.Forward(backend, preAttn, crossKv: null, cos, sin, mask);
            preAttn.Dispose();
            Tensor h1 = new Tensor(shape, act);
            backend.GatedResidualLastDim(h1, x, attn, mod[2]);                        // x + gate_sa·attn
            attn.Dispose();

            Tensor n2 = new Tensor(shape, act);
            backend.RmsNorm(n2, h1, _crossNorm!, _c.RmsNormEps);
            Tensor cross = _crossAttn.Forward(backend, n2, cond, ropeCos: null, ropeSin: null, mask: null);
            n2.Dispose();
            Tensor h2 = new Tensor(shape, act);
            backend.Add(h2, h1, cross);                                              // ungated cross-attn residual
            h1.Dispose(); cross.Dispose();

            Tensor n3 = new Tensor(shape, act);
            backend.RmsNorm(n3, h2, _mlpNorm!, _c.RmsNormEps);
            Tensor preMlp = DiTUtils.Modulate(backend, n3, mod[3], mod[4], shape);    // (1+scale_mlp)·n3 + shift_mlp
            n3.Dispose();
            Tensor ff = _mlp.Forward(backend, preMlp, 1, s);
            preMlp.Dispose();
            Tensor h3 = new Tensor(shape, act);
            backend.GatedResidualLastDim(h3, h2, ff, mod[5]);                        // h2 + gate_mlp·ff
            h2.Dispose(); ff.Dispose();

            foreach (Tensor m in mod) m.Dispose();
            return h3;
        }

        /// <summary>Splits <c>proj6 [1, 6·dim] + scale_shift_table [1, 6·dim]</c> into 6 <c>[1, dim]</c> vectors,
        /// device-resident (one Add over the whole width + 6 row slices), mirroring Krea2Block.SplitModulation.</summary>
        private Tensor[] SplitModulation(IBackend backend, Tensor proj6, int dim)
        {
            Tensor packed = new Tensor(new TensorShape(1, 6 * dim), DType.F32);
            backend.Add(packed, proj6, _scaleShift!);
            Tensor[] mod = new Tensor[6];
            for (int i = 0; i < 6; i++)
            {
                mod[i] = new Tensor(new TensorShape(1, dim), DType.F32);
                backend.SliceRows(mod[i], packed, i);
            }
            packed.Dispose();
            return mod;
        }
    }
}

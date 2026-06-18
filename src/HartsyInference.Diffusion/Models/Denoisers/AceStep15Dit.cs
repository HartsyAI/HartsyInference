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
/// Numerics validation-pending vs the Python reference.</summary>
public sealed unsafe class AceStep15Dit : IDisposable
{
    private readonly AceStep15Config _config;
    private Layer[] _layers = [];
    private int _disposed;

    private Tensor? _projInW, _projInB, _projOutW, _projOutB;
    private Tensor? _tL1W, _tL1B, _tL2W, _tL2B, _tProjW, _tProjB;
    private Tensor? _rL1W, _rL1B, _rL2W, _rL2B, _rProjW, _rProjB;
    private Tensor? _condEmbW, _condEmbB, _normOutW, _globalScaleShift;

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

        for (int i = 0; i < _layers.Length; i++)
            _layers[i].Forward(backend, tokens, cond, proj6, cos, sin,
                _config.IsSlidingLayer(i) ? slidingMask : null);
        slidingMask?.Dispose();
        cond.Dispose();
        proj6.Dispose();

        Tensor velocity = FinalLayer(backend, tokens, temb, s, t);
        tokens.Dispose();
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
        float* gp = (float*)_globalScaleShift!.DataPointer;
        float* tp = (float*)temb.DataPointer;
        float* np = (float*)normed.DataPointer;
        for (int i = 0; i < s; i++)
            for (int d = 0; d < dim; d++)
            {
                float shift = gp[d] + tp[d];
                float scale = gp[dim + d] + tp[d];
                long off = (long)i * dim + d;
                np[off] = np[off] * (1f + scale) + shift;
            }

        Tensor channelsFirst = new Tensor(new TensorShape(1, dim, s), DType.F32);
        backend.Transpose2D(channelsFirst, normed, s, dim);
        normed.Dispose();
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
            _scaleShift = EnsureF32(w[$"{p}.scale_shift_table"]);
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

        /// <summary>Updates <paramref name="x"/> <c>[1, S, H]</c> in place.</summary>
        public void Forward(IBackend backend, Tensor x, Tensor cond, Tensor proj6,
            float[] cos, float[] sin, Tensor? mask)
        {
            int s = (int)x.Shape[1], dim = _c.HiddenSize;
            float* table = (float*)_scaleShift!.DataPointer;
            float* proj = (float*)proj6.DataPointer;
            TensorShape shape = new TensorShape(1, s, dim);

            Tensor n1 = new Tensor(shape, DType.F32);
            backend.RmsNorm(n1, x, _selfNorm!, _c.RmsNormEps);
            Modulate(n1, table, proj, dim, shiftIdx: 0, scaleIdx: 1, rows: s);
            Tensor attn = _selfAttn.Forward(backend, n1, crossKv: null, cos, sin, mask);
            n1.Dispose();
            GatedAdd(x, attn, table, proj, dim, gateIdx: 2, rows: s);
            attn.Dispose();

            Tensor n2 = new Tensor(shape, DType.F32);
            backend.RmsNorm(n2, x, _crossNorm!, _c.RmsNormEps);
            Tensor cross = _crossAttn.Forward(backend, n2, cond, ropeCos: null, ropeSin: null, mask: null);
            n2.Dispose();
            AddInPlace(x, cross);
            cross.Dispose();

            Tensor n3 = new Tensor(shape, DType.F32);
            backend.RmsNorm(n3, x, _mlpNorm!, _c.RmsNormEps);
            Modulate(n3, table, proj, dim, shiftIdx: 3, scaleIdx: 4, rows: s);
            Tensor ff = _mlp.Forward(backend, n3, 1, s);
            n3.Dispose();
            GatedAdd(x, ff, table, proj, dim, gateIdx: 5, rows: s);
            ff.Dispose();
        }

        private static void Modulate(Tensor x, float* table, float* proj, int dim, int shiftIdx, int scaleIdx, int rows)
        {
            float* xp = (float*)x.DataPointer;
            for (int i = 0; i < rows; i++)
                for (int d = 0; d < dim; d++)
                {
                    float shift = table[shiftIdx * dim + d] + proj[shiftIdx * dim + d];
                    float scale = table[scaleIdx * dim + d] + proj[scaleIdx * dim + d];
                    long off = (long)i * dim + d;
                    xp[off] = xp[off] * (1f + scale) + shift;
                }
        }

        private static void GatedAdd(Tensor target, Tensor value, float* table, float* proj, int dim, int gateIdx, int rows)
        {
            float* tp = (float*)target.DataPointer;
            float* vp = (float*)value.DataPointer;
            for (int i = 0; i < rows; i++)
                for (int d = 0; d < dim; d++)
                {
                    float gate = table[gateIdx * dim + d] + proj[gateIdx * dim + d];
                    tp[(long)i * dim + d] += gate * vp[(long)i * dim + d];
                }
        }

        private static void AddInPlace(Tensor target, Tensor value)
        {
            float* tp = (float*)target.DataPointer;
            float* vp = (float*)value.DataPointer;
            for (long i = 0; i < target.Shape.ElementCount; i++) tp[i] += vp[i];
        }

        private static Tensor EnsureF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);
    }
}

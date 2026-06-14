using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Oasis spatio-temporal DiT block (<c>SpatioTemporalDiTBlock</c>, Latte-style): a <b>spatial half</b>
/// (bidirectional self-attention over each frame's tokens with 2-D axial RoPE, then MLP) followed by a
/// <b>temporal half</b> (causal self-attention over the time axis at each spatial location with 1-D RoPE, then MLP).
/// Each half has its own adaLN-zero modulation (<c>SiLU → Linear(dim → 6·dim)</c>) conditioned per frame on
/// timestep-embedding + projected action. Pre-norms are no-affine LayerNorm; MLPs are GELU-tanh ×4; fused QKV without
/// bias, output projection with bias. See <c>docs/Research/OASIS_ARCHITECTURE.md</c> § 3.1-3.3.</summary>
public sealed unsafe class OasisSpatioTemporalBlock
{
    private readonly int _dim;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _mlpHidden;

    private Tensor? _sModW, _sModB, _tModW, _tModB;       // adaLN: [6·dim, dim]
    private Tensor? _sQkvW, _sQkvB, _sOutW, _sOutB;
    private Tensor? _tQkvW, _tQkvB, _tOutW, _tOutB;
    private Tensor? _sFc1W, _sFc1B, _sFc2W, _sFc2B;
    private Tensor? _tFc1W, _tFc1B, _tFc2W, _tFc2B;

    public OasisSpatioTemporalBlock(OasisDitConfig c)
    {
        _dim = c.HiddenSize;
        _heads = c.NumHeads;
        _headDim = c.HiddenSize / c.NumHeads;
        _mlpHidden = (int)(c.HiddenSize * c.MlpRatio);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _sModW = w[$"{p}.s_adaLN_modulation.1.weight"]; w.TryGetValue($"{p}.s_adaLN_modulation.1.bias", out _sModB);
        _tModW = w[$"{p}.t_adaLN_modulation.1.weight"]; w.TryGetValue($"{p}.t_adaLN_modulation.1.bias", out _tModB);
        _sQkvW = w[$"{p}.s_attn.to_qkv.weight"]; w.TryGetValue($"{p}.s_attn.to_qkv.bias", out _sQkvB);
        _sOutW = w[$"{p}.s_attn.to_out.weight"]; w.TryGetValue($"{p}.s_attn.to_out.bias", out _sOutB);
        _tQkvW = w[$"{p}.t_attn.to_qkv.weight"]; w.TryGetValue($"{p}.t_attn.to_qkv.bias", out _tQkvB);
        _tOutW = w[$"{p}.t_attn.to_out.weight"]; w.TryGetValue($"{p}.t_attn.to_out.bias", out _tOutB);
        _sFc1W = w[$"{p}.s_mlp.fc1.weight"]; w.TryGetValue($"{p}.s_mlp.fc1.bias", out _sFc1B);
        _sFc2W = w[$"{p}.s_mlp.fc2.weight"]; w.TryGetValue($"{p}.s_mlp.fc2.bias", out _sFc2B);
        _tFc1W = w[$"{p}.t_mlp.fc1.weight"]; w.TryGetValue($"{p}.t_mlp.fc1.bias", out _tFc1B);
        _tFc2W = w[$"{p}.t_mlp.fc2.weight"]; w.TryGetValue($"{p}.t_mlp.fc2.bias", out _tFc2B);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _sModW, _sModB, _tModW, _tModB, _sQkvW, _sQkvB, _sOutW, _sOutB,
            _tQkvW, _tQkvB, _tOutW, _tOutB, _sFc1W, _sFc1B, _sFc2W, _sFc2B, _tFc1W, _tFc1B, _tFc2W, _tFc2B })
            if (t is not null) yield return t;
    }

    /// <summary>Forward over frame-major tokens <c>[T·S, dim]</c> (frame f's S tokens contiguous).
    /// <paramref name="cond"/> is the per-frame conditioning <c>[T, dim]</c>; <paramref name="spatialCos"/>/Sin are
    /// the per-token axial RoPE <c>[S, rotS]</c>; <paramref name="temporalCos"/>/Sin the per-frame 1-D RoPE
    /// <c>[T, headDim]</c>; <paramref name="causalMask"/> is the additive <c>[1, 1, T, T]</c> mask.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor cond, int frames, int tokensPerFrame,
        Tensor spatialCos, Tensor spatialSin, Tensor temporalCos, Tensor temporalSin, Tensor causalMask)
    {
        Tensor cur = Clone(x);
        Half(backend, cur, cond, frames, tokensPerFrame, spatial: true, spatialCos, spatialSin, temporalCos, temporalSin, causalMask);
        Half(backend, cur, cond, frames, tokensPerFrame, spatial: false, spatialCos, spatialSin, temporalCos, temporalSin, causalMask);
        return cur;
    }

    private void Half(IBackend backend, Tensor x, Tensor cond, int frames, int sp, bool spatial,
        Tensor spatialCos, Tensor spatialSin, Tensor temporalCos, Tensor temporalSin, Tensor causalMask)
    {
        int total = frames * sp;
        (Tensor modW, Tensor? modB) = spatial ? (_sModW!, _sModB) : (_tModW!, _tModB);
        Tensor mod = Modulation(backend, cond, frames, modW, modB);   // [T, 6·dim]
        float* mp = (float*)mod.DataPointer;

        // Attention sub-half.
        Tensor n1 = Clone(x);
        DiTUtils.LayerNormNoAffine(n1, x, 1, total, _dim, 1e-6f);
        ModulatePerFrame(n1, mp, shiftSlot: 0, scaleSlot: 1, frames, sp);
        Tensor attn = spatial
            ? SpatialAttention(backend, n1, frames, sp, spatialCos, spatialSin)
            : TemporalAttention(backend, n1, frames, sp, temporalCos, temporalSin, causalMask);
        n1.Dispose();
        GatedAddPerFrame(x, attn, mp, gateSlot: 2, frames, sp);
        attn.Dispose();

        // MLP sub-half.
        (Tensor fc1W, Tensor? fc1B, Tensor fc2W, Tensor? fc2B) = spatial
            ? (_sFc1W!, _sFc1B, _sFc2W!, _sFc2B)
            : (_tFc1W!, _tFc1B, _tFc2W!, _tFc2B);
        Tensor n2 = Clone(x);
        DiTUtils.LayerNormNoAffine(n2, x, 1, total, _dim, 1e-6f);
        ModulatePerFrame(n2, mp, shiftSlot: 3, scaleSlot: 4, frames, sp);
        Tensor mid = new Tensor(new TensorShape(total, _mlpHidden), DType.F32);
        backend.Linear(mid, n2, fc1W, fc1B);
        n2.Dispose();
        Tensor act = new Tensor(mid.Shape, DType.F32);
        backend.Gelu(act, mid);
        mid.Dispose();
        Tensor mlpOut = new Tensor(new TensorShape(total, _dim), DType.F32);
        backend.Linear(mlpOut, act, fc2W, fc2B);
        act.Dispose();
        GatedAddPerFrame(x, mlpOut, mp, gateSlot: 5, frames, sp);
        mlpOut.Dispose();
        mod.Dispose();
    }

    /// <summary>Per-frame bidirectional attention over each frame's tokens, batched <c>[T, heads, S, headDim]</c> with axial RoPE.</summary>
    private Tensor SpatialAttention(IBackend backend, Tensor n, int frames, int sp, Tensor cos, Tensor sin)
    {
        (Tensor qkvW, Tensor? qkvB, Tensor outW, Tensor? outB) = (_sQkvW!, _sQkvB, _sOutW!, _sOutB);
        Tensor qkv = new Tensor(new TensorShape(frames * sp, 3 * _dim), DType.F32);
        backend.Linear(qkv, n, qkvW, qkvB);

        Tensor q = SplitHeads(qkv, frames, sp, part: 0, seqIsTemporal: false);
        Tensor k = SplitHeads(qkv, frames, sp, part: 1, seqIsTemporal: false);
        Tensor v = SplitHeads(qkv, frames, sp, part: 2, seqIsTemporal: false);
        qkv.Dispose();
        RotateBhsd(q, cos, sin);
        RotateBhsd(k, cos, sin);

        Tensor attn = new Tensor(new TensorShape([(long)frames, _heads, sp, _headDim]), DType.F32);
        backend.ScaledDotProductAttention(attn, q, k, v, null, 1.0f / MathF.Sqrt(_headDim));
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor merged = MergeHeads(attn, frames, sp, seqIsTemporal: false);
        attn.Dispose();
        Tensor projected = new Tensor(new TensorShape(frames * sp, _dim), DType.F32);
        backend.Linear(projected, merged, outW, outB);
        merged.Dispose();
        return projected;
    }

    /// <summary>Per-spatial-location causal attention over time, batched <c>[S, heads, T, headDim]</c> with 1-D RoPE.</summary>
    private Tensor TemporalAttention(IBackend backend, Tensor n, int frames, int sp, Tensor cos, Tensor sin, Tensor causalMask)
    {
        (Tensor qkvW, Tensor? qkvB, Tensor outW, Tensor? outB) = (_tQkvW!, _tQkvB, _tOutW!, _tOutB);
        Tensor qkv = new Tensor(new TensorShape(frames * sp, 3 * _dim), DType.F32);
        backend.Linear(qkv, n, qkvW, qkvB);

        Tensor q = SplitHeads(qkv, frames, sp, part: 0, seqIsTemporal: true);
        Tensor k = SplitHeads(qkv, frames, sp, part: 1, seqIsTemporal: true);
        Tensor v = SplitHeads(qkv, frames, sp, part: 2, seqIsTemporal: true);
        qkv.Dispose();
        RotateBhsd(q, cos, sin);
        RotateBhsd(k, cos, sin);

        Tensor attn = new Tensor(new TensorShape([(long)sp, _heads, frames, _headDim]), DType.F32);
        backend.ScaledDotProductAttention(attn, q, k, v, causalMask, 1.0f / MathF.Sqrt(_headDim));
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor merged = MergeHeads(attn, frames, sp, seqIsTemporal: true);
        attn.Dispose();
        Tensor projected = new Tensor(new TensorShape(frames * sp, _dim), DType.F32);
        backend.Linear(projected, merged, outW, outB);
        merged.Dispose();
        return projected;
    }

    /// <summary>SiLU(cond) → Linear(dim → 6·dim) per frame.</summary>
    private Tensor Modulation(IBackend backend, Tensor cond, int frames, Tensor modW, Tensor? modB)
    {
        Tensor act = new Tensor(new TensorShape(frames, _dim), DType.F32);
        backend.Silu(act, cond);
        Tensor mod = new Tensor(new TensorShape(frames, 6 * _dim), DType.F32);
        backend.Linear(mod, act, modW, modB);
        act.Dispose();
        return mod;
    }

    private void ModulatePerFrame(Tensor x, float* mod, int shiftSlot, int scaleSlot, int frames, int sp)
    {
        float* xp = (float*)x.DataPointer;
        for (int f = 0; f < frames; f++)
        {
            float* shift = mod + (long)f * 6 * _dim + (long)shiftSlot * _dim;
            float* scale = mod + (long)f * 6 * _dim + (long)scaleSlot * _dim;
            for (int i = 0; i < sp; i++)
            {
                long off = ((long)f * sp + i) * _dim;
                for (int d = 0; d < _dim; d++) xp[off + d] = xp[off + d] * (1f + scale[d]) + shift[d];
            }
        }
    }

    private void GatedAddPerFrame(Tensor target, Tensor value, float* mod, int gateSlot, int frames, int sp)
    {
        float* tp = (float*)target.DataPointer;
        float* vp = (float*)value.DataPointer;
        for (int f = 0; f < frames; f++)
        {
            float* gate = mod + (long)f * 6 * _dim + (long)gateSlot * _dim;
            for (int i = 0; i < sp; i++)
            {
                long off = ((long)f * sp + i) * _dim;
                for (int d = 0; d < _dim; d++) tp[off + d] += gate[d] * vp[off + d];
            }
        }
    }

    /// <summary>Frame-major token rows → <c>[batch, heads, seq, headDim]</c>; spatial batches frames (seq = tokens),
    /// temporal batches spatial positions (seq = frames).</summary>
    private Tensor SplitHeads(Tensor qkv, int frames, int sp, int part, bool seqIsTemporal)
    {
        int batch = seqIsTemporal ? sp : frames;
        int seq = seqIsTemporal ? frames : sp;
        Tensor o = new Tensor(new TensorShape([(long)batch, _heads, seq, _headDim]), DType.F32);
        float* src = (float*)qkv.DataPointer;
        float* dst = (float*)o.DataPointer;
        for (int f = 0; f < frames; f++)
            for (int i = 0; i < sp; i++)
            {
                long token = (long)f * sp + i;
                int b = seqIsTemporal ? i : f;
                int s = seqIsTemporal ? f : i;
                for (int h = 0; h < _heads; h++)
                    Buffer.MemoryCopy(
                        src + token * 3 * _dim + (long)part * _dim + (long)h * _headDim,
                        dst + (((long)b * _heads + h) * seq + s) * _headDim,
                        (long)_headDim * 4, (long)_headDim * 4);
            }
        return o;
    }

    private Tensor MergeHeads(Tensor attn, int frames, int sp, bool seqIsTemporal)
    {
        int seq = seqIsTemporal ? frames : sp;
        Tensor o = new Tensor(new TensorShape(frames * sp, _dim), DType.F32);
        float* src = (float*)attn.DataPointer;
        float* dst = (float*)o.DataPointer;
        for (int f = 0; f < frames; f++)
            for (int i = 0; i < sp; i++)
            {
                long token = (long)f * sp + i;
                int b = seqIsTemporal ? i : f;
                int s = seqIsTemporal ? f : i;
                for (int h = 0; h < _heads; h++)
                    Buffer.MemoryCopy(
                        src + (((long)b * _heads + h) * seq + s) * _headDim,
                        dst + token * _dim + (long)h * _headDim,
                        (long)_headDim * 4, (long)_headDim * 4);
            }
        return o;
    }

    /// <summary>Rotates the first <c>rotDim = cos.Shape[1]</c> dims of every head; remaining dims pass through.</summary>
    private void RotateBhsd(Tensor x, Tensor cos, Tensor sin)
    {
        int batch = (int)x.Shape[0], seq = (int)x.Shape[2];
        int rotDim = (int)cos.Shape[1];
        int pairs = rotDim / 2;
        float* xp = (float*)x.DataPointer;
        float* cp = (float*)cos.DataPointer;
        float* sp = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < _heads; h++)
                for (int s = 0; s < seq; s++)
                {
                    long cOff = (long)s * rotDim;
                    long xOff = (((long)b * _heads + h) * seq + s) * _headDim;
                    for (int i = 0; i < pairs; i++)
                    {
                        int i0 = 2 * i;
                        float re = xp[xOff + i0], im = xp[xOff + i0 + 1];
                        float c = cp[cOff + i0], sn = sp[cOff + i0];
                        xp[xOff + i0] = re * c - im * sn;
                        xp[xOff + i0 + 1] = re * sn + im * c;
                    }
                }
    }

    private static Tensor Clone(Tensor x)
    {
        Tensor o = new Tensor(x.Shape, DType.F32);
        long bytes = x.Shape.ElementCount * 4;
        Buffer.MemoryCopy((float*)x.DataPointer, (float*)o.DataPointer, bytes, bytes);
        return o;
    }
}

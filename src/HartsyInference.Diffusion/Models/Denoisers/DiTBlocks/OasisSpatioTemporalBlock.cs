using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Rope;
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
    // ── Diagnostic phase timers (diagnostics.oasisPhase=true, eager path only) — Sync-bracketed GPU time per phase. ──
    internal static bool Prof => EngineKnobs.OasisPhase.Value;
    internal static double TSdpa, TAttnRest, TMlp, TModNorm;
    private static double Now() => System.Diagnostics.Stopwatch.GetTimestamp() * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

    private readonly int _dim;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _mlpHidden;
    private readonly AttentionLayer _attention;

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
        _attention = new AttentionLayer(new AttentionLayerConfig
        {
            Heads = _heads,
            KvHeads = _heads,
            HeadDim = _headDim,
            ComputeDType = DType.F32,
            AllowF16 = true,
        });
    }

    internal AttentionLayer AttentionStage => _attention;

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
        // Residual carrier is rank-3 [T, sp, dim] so the device per-frame modulation/gated-residual ops index
        // the frame axis (b) correctly; attention's flat-token host code reads the same memory unaffected.
        Tensor cur = new(new TensorShape([(long)frames, tokensPerFrame, _dim]), DType.F32);
        backend.CopyInto(cur, x);
        Half(backend, cur, cond, frames, tokensPerFrame, spatial: true, spatialCos, spatialSin, temporalCos, temporalSin, causalMask);
        Half(backend, cur, cond, frames, tokensPerFrame, spatial: false, spatialCos, spatialSin, temporalCos, temporalSin, causalMask);
        return cur;
    }

    private void Half(IBackend backend, Tensor x, Tensor cond, int frames, int sp, bool spatial,
        Tensor spatialCos, Tensor spatialSin, Tensor temporalCos, Tensor temporalSin, Tensor causalMask)
    {
        if (Prof) { backend.Sync(); TModNorm -= Now(); }
        (Tensor modW, Tensor? modB) = spatial ? (_sModW!, _sModB) : (_tModW!, _tModB);
        Tensor mod = Modulation(backend, cond, frames, modW, modB);   // [T, 6·dim] device

        // Attention sub-half: adaLN(x) = shift + (1+scale)·LayerNorm(x) — one fused kernel (per-frame slice+affine).
        Tensor n1 = new(new TensorShape([(long)frames, sp, _dim]), DType.F32);
        backend.NormModulate(NormModulateCall.Rows(n1, x, NormKind.LayerNorm, 1e-6f).WithModulation(ModulationSpec.Packed(mod, 6 * _dim, 1 * _dim, 0 * _dim, sp, ScaleMode.OnePlus)));
        if (Prof) { backend.Sync(); TModNorm += Now(); TAttnRest -= Now(); }
        Tensor attn = spatial ? SpatialAttention(backend, n1, frames, sp, spatialCos, spatialSin)
            : TemporalAttention(backend, n1, frames, sp, temporalCos, temporalSin, causalMask);
        n1.Dispose();
        GatedResidual(backend, x, attn, mod, frames, gateSlot: 2);   // x += gate·attn (in place, device)
        attn.Dispose();
        if (Prof) { backend.Sync(); TAttnRest += Now(); TMlp -= Now(); }

        // MLP sub-half.
        (Tensor fc1W, Tensor? fc1B, Tensor fc2W, Tensor? fc2B) = spatial ? (_sFc1W!, _sFc1B, _sFc2W!, _sFc2B)
            : (_tFc1W!, _tFc1B, _tFc2W!, _tFc2B);
        Tensor n2 = new(new TensorShape([(long)frames, sp, _dim]), DType.F32);
        backend.NormModulate(NormModulateCall.Rows(n2, x, NormKind.LayerNorm, 1e-6f).WithModulation(ModulationSpec.Packed(mod, 6 * _dim, 4 * _dim, 3 * _dim, sp, ScaleMode.OnePlus)));
        Tensor mid = new Tensor(new TensorShape(frames * sp, _mlpHidden), DType.F32);
        backend.Linear(mid, n2, fc1W, fc1B);
        n2.Dispose();
        Tensor act = new Tensor(mid.Shape, DType.F32);
        backend.Gelu(act, mid);
        mid.Dispose();
        Tensor mlpOut = new(new TensorShape([(long)frames, sp, _dim]), DType.F32);
        backend.Linear(mlpOut, act, fc2W, fc2B);
        act.Dispose();
        GatedResidual(backend, x, mlpOut, mod, frames, gateSlot: 5);
        mlpOut.Dispose();
        mod.Dispose();
        if (Prof) { backend.Sync(); TMlp += Now(); }
    }

    /// <summary>Device per-frame gated residual: <c>target[f,i,d] += gate[f,d]·value[f,i,d]</c> in place.</summary>
    private void GatedResidual(IBackend backend, Tensor target, Tensor value, Tensor mod, int frames, int gateSlot)
    {
        Tensor gate = new(new TensorShape(frames, _dim), DType.F32); backend.SliceLastDim(gate, mod, gateSlot * _dim);
        backend.GatedResidualLastDim(target, target, value, gate);
        gate.Dispose();
    }

    /// <summary>Per-frame bidirectional attention over each frame's tokens, batched <c>[T, heads, S, headDim]</c> with axial RoPE.</summary>
    private Tensor SpatialAttention(IBackend backend, Tensor n, int frames, int sp, Tensor cos, Tensor sin)
    {
        (Tensor qkvW, Tensor? qkvB, Tensor outW, Tensor? outB) = (_sQkvW!, _sQkvB, _sOutW!, _sOutB);
        Tensor qkv = new Tensor(new TensorShape(frames * sp, 3 * _dim), DType.F32);
        backend.Linear(qkv, n, qkvW, qkvB);

        Tensor q = SplitHeads(backend, qkv, frames, sp, part: 0, temporal: false);
        Tensor k = SplitHeads(backend, qkv, frames, sp, part: 1, temporal: false);
        Tensor v = SplitHeads(backend, qkv, frames, sp, part: 2, temporal: false);
        qkv.Dispose();
        int rotDim = (int)cos.Shape[1];
        backend.Rotary(RotaryCall.HeadMajor(q, cos, sin, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair, rotDim));
        backend.Rotary(RotaryCall.HeadMajor(k, cos, sin, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair, rotDim));

        if (Prof) { backend.Sync(); TSdpa -= Now(); }
        // cuDNN fused flash attention: mask-null, D=64, F16-tolerant (verified corr>0.9999) — collapses the
        // materialized QKᵀ→softmax→PV path that is catastrophic for these batched small-seq shapes.
        Tensor attn = _attention.AttendHeadMajorOperands(backend, q, k, v, default);
        if (Prof) { backend.Sync(); TSdpa += Now(); }

        Tensor merged = new(new TensorShape(frames * sp, _dim), DType.F32);
        MergeHeads(backend, merged, attn, frames, sp, temporal: false);
        attn.Dispose();
        Tensor projected = new(new TensorShape([(long)frames, sp, _dim]), DType.F32);
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

        Tensor q = SplitHeads(backend, qkv, frames, sp, part: 0, temporal: true);
        Tensor k = SplitHeads(backend, qkv, frames, sp, part: 1, temporal: true);
        Tensor v = SplitHeads(backend, qkv, frames, sp, part: 2, temporal: true);
        qkv.Dispose();
        int rotDim = (int)cos.Shape[1];
        backend.Rotary(RotaryCall.HeadMajor(q, cos, sin, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair, rotDim));
        backend.Rotary(RotaryCall.HeadMajor(k, cos, sin, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair, rotDim));

        if (Prof) { backend.Sync(); TSdpa -= Now(); }
        // Temporal causal mask is [1,1,T,T] — a cuDNN-eligible [B,1,Sq,Skv] additive F32 mask; F16-tolerant.
        Tensor attn = _attention.AttendHeadMajorOperands(backend, q, k, v, new AttentionCallContext { Bias = causalMask });
        if (Prof) { backend.Sync(); TSdpa += Now(); }

        Tensor merged = new(new TensorShape(frames * sp, _dim), DType.F32);
        MergeHeads(backend, merged, attn, frames, sp, temporal: true);
        attn.Dispose();
        Tensor projected = new(new TensorShape([(long)frames, sp, _dim]), DType.F32);
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

    /// <summary>Head split: frame-major qkv <c>[token, 3·dim]</c> → <c>[batch, heads, seq, headDim]</c>, batching frames
    /// (spatial, seq = positions) or positions (temporal, seq = frames). One strided copy of the chosen q/k/v slot.</summary>
    private Tensor SplitHeads(IBackend backend, Tensor qkv, int frames, int sp, int part, bool temporal)
    {
        int batch = temporal ? sp : frames;
        int seq = temporal ? frames : sp;
        Tensor o = new(new TensorShape([(long)batch, _heads, seq, _headDim]), DType.F32);
        StridedView slot = StridedView.Of(qkv.Shape).Reshape([frames, sp, 3, _heads, _headDim]).Select(2, part);
        backend.CopyStrided(o, qkv, StridedCopySpec.Gather(slot.Permute(HeadMajorOrder(temporal))));
        return o;
    }

    /// <summary>Inverse of <see cref="SplitHeads"/>: <c>[batch, heads, seq, headDim]</c> back into frame-major token rows <c>[token, dim]</c>.</summary>
    private void MergeHeads(IBackend backend, Tensor merged, Tensor attn, int frames, int sp, bool temporal)
    {
        StridedView rows = StridedView.Of(merged.Shape).Reshape([frames, sp, _heads, _headDim]).Permute(HeadMajorOrder(temporal));
        backend.CopyStrided(merged, attn, StridedCopySpec.Scatter(rows));
    }

    /// <summary>Axis order taking <c>[frame, position, head, dim]</c> to <c>[batch, head, seq, dim]</c>.</summary>
    private static ReadOnlySpan<int> HeadMajorOrder(bool temporal) => temporal ? [1, 2, 0, 3] : [0, 2, 1, 3];
}

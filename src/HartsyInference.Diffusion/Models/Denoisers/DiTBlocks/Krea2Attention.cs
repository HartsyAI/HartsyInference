using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Krea 2 self-attention (<c>Krea2Attention</c> + <c>Krea2AttnProcessor</c>). Grouped-query attention with
/// per-head zero-centered RMSNorm on Q and K, optional 3-axis RoPE, and a learned <b>sigmoid output gate</b>:
/// <c>out = to_out( sdpa(q, k, v) · sigmoid(to_gate(x)) )</c>. Shared by the main transformer blocks (with RoPE, GQA)
/// and the text-fusion blocks (no RoPE, full MHA).
///
/// <para>The zero-centered RMSNorm scales (<c>weight + 1</c>) are folded to plain RMSNorm weights at load time (the
/// caller adds 1.0), so the per-head norm runs through <see cref="IBackend.RmsNorm"/> unchanged.</para></summary>
public sealed unsafe class Krea2Attention
{
    private readonly int _hidden;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _kvGroup;
    private readonly float _eps;

    private Tensor? _toQ, _toK, _toV, _toGate, _toOut, _normQ, _normK;

    /// <summary>Creates an attention module. <paramref name="numKvHeads"/> &lt; <paramref name="numHeads"/> enables GQA.</summary>
    public Krea2Attention(int hidden, int numHeads, int numKvHeads, float eps = 1e-5f)
    {
        if (hidden % numHeads != 0)
            throw new ArgumentException($"hidden {hidden} must be divisible by numHeads {numHeads}.");
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException($"numHeads {numHeads} must be divisible by numKvHeads {numKvHeads}.");
        _hidden = hidden;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = hidden / numHeads;
        _kvGroup = numHeads / numKvHeads;
        _eps = eps;
    }

    /// <summary>Loads <c>{prefix}.to_q/to_k/to_v/to_gate.weight</c>, <c>{prefix}.to_out.0.weight</c> and the per-head
    /// <c>{prefix}.norm_q/norm_k.weight</c> (zero-centered: 1.0 is added at load).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _toQ = w[$"{p}.to_q.weight"];
        _toK = w[$"{p}.to_k.weight"];
        _toV = w[$"{p}.to_v.weight"];
        _toGate = w[$"{p}.to_gate.weight"];
        _toOut = w[$"{p}.to_out.0.weight"];
        _normQ = Krea2Norm.LoadZeroCentered(w[$"{p}.norm_q.weight"]);
        _normK = Krea2Norm.LoadZeroCentered(w[$"{p}.norm_k.weight"]);
    }

    /// <summary>Enumerates weight tensors for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _toQ, _toK, _toV, _toGate, _toOut, _normQ, _normK })
            if (t is not null) yield return t;
    }

    /// <summary>Runs attention over <paramref name="x"/> <c>[B, S, hidden]</c>. Pass <paramref name="rope"/> (already
    /// <c>Precompute</c>d for this sequence) for the main blocks; pass null for the text-fusion blocks.</summary>
    // GPU-residency rewrite (mirrors the verified FluxSingleStreamBlock / HunyuanImageSingleBlock): every glue op
    // (head split/merge, per-head QK-norm, GQA K/V repeat, the sigmoid output gate) runs as an IBackend op so the
    // activation chain stays device-resident — no per-op DataPointer D2H sync barriers (the old ReshapeTo/FromMultiHead
    // + host RepeatKvHeads loops D2H-synced every intermediate around every GEMM). Head split = declaring Q/K/V
    // directly as [B, S, H, D] (byte-identical to [B, S, hidden]) so RmsNorm normalizes over headDim with no reshape,
    // then Permute0213 to/from [B, H, S, D]. RoPE is GPU-resident for B=1 (the pipeline case) via
    // FluxRope.ApplyGpuGqa — WanRopeInterleaved is single-tensor with an explicit head count, so GQA
    // (numHeads != numKvHeads) rotates Q and K with their own counts on the pre-permute [B, S, H, D] layout,
    // bit-identical to the host ForwardSingle. B>1 keeps the host ForwardSingle fallback (post-permute).
    public Tensor Forward(IBackend backend, Tensor x, FluxRope? rope, int batch, int seqLen)
    {
        TensorShape qHeads = new TensorShape(batch, seqLen, _numHeads, _headDim);
        TensorShape kvHeads = new TensorShape(batch, seqLen, _numKvHeads, _headDim);
        TensorShape qMhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        TensorShape kvMhShape = new TensorShape(batch, _numKvHeads, seqLen, _headDim);
        TensorShape flatShape = new TensorShape(batch, seqLen, _hidden);

        // Activation dtype follows the INPUT: the main blocks feed preIn in Krea2Dtype.Act (F16 on the
        // HARTSY_KREA2_F16 path), while the text-fusion blocks (whose surrounding ops are F32 and cached
        // once per prompt) feed F32 and keep the baseline path. Weights stay packed fp8; the norm weights
        // (_normQ/_normK) are F32 — the F16 RmsNorm/RoPE/Sigmoid/RepeatKv kernels take F16 activation + F32 params.
        DType act = x.DType;

        // Q/K/V declared [B, S, H, D] (byte-identical to [B, S, H·D]) + gate [B, S, hidden].
        Tensor q = new Tensor(qHeads, act);
        backend.Linear(q, x, _toQ!, null);
        Tensor k = new Tensor(kvHeads, act);
        backend.Linear(k, x, _toK!, null);
        Tensor v = new Tensor(kvHeads, act);
        backend.Linear(v, x, _toV!, null);
        Tensor gate = new Tensor(flatShape, act);
        backend.Linear(gate, x, _toGate!, null);

        // Per-head zero-centered RMSNorm on Q and K (weight already folded +1 at load), over the last dim = headDim.
        Tensor qNorm = new Tensor(qHeads, act);
        backend.RmsNorm(qNorm, q, _normQ!, _eps);
        q.Dispose();
        Tensor kNorm = new Tensor(kvHeads, act);
        backend.RmsNorm(kNorm, k, _normK!, _eps);
        k.Dispose();

        // RoPE (main blocks only; text-fusion passes null). B=1 (the only case pipelines exercise): GPU-resident
        // on the pre-permute [B, S, H, D] layout (device WanRopeInterleaved with per-tensor GQA head counts —
        // bit-identical interleaved-pair rotation, no D2H). B>1: host ForwardSingle on the post-permute layout.
        bool gpuRope = rope is not null && batch == 1;
        if (gpuRope)
        {
            rope!.ApplyGpuGqa(backend, qNorm, kNorm, _numHeads, _numKvHeads);
        }

        // Permute [B, S, H, D] → [B, H, S, D] for SDPA.
        Tensor qMh = new Tensor(qMhShape, act);
        backend.Permute0213(qMh, qNorm, seqLen, _numHeads, _headDim);
        qNorm.Dispose();
        Tensor kMh = new Tensor(kvMhShape, act);
        backend.Permute0213(kMh, kNorm, seqLen, _numKvHeads, _headDim);
        kNorm.Dispose();
        Tensor vMh = new Tensor(kvMhShape, act);
        backend.Permute0213(vMh, v, seqLen, _numKvHeads, _headDim);
        v.Dispose();

        if (rope is not null && !gpuRope)
        {
            rope.ForwardSingle(qMh, batch, _numHeads, seqLen);
            rope.ForwardSingle(kMh, batch, _numKvHeads, seqLen);
        }

        // GQA K/V head repeat (device). group == 1 (text-fusion, full MHA) skips the copy.
        Tensor kRep, vRep;
        if (_kvGroup == 1)
        {
            kRep = kMh;
            vRep = vMh;
        }
        else
        {
            kRep = new Tensor(qMhShape, act);
            backend.RepeatKvHeads(kRep, kMh, _numKvHeads, _kvGroup);
            vRep = new Tensor(qMhShape, act);
            backend.RepeatKvHeads(vRep, vMh, _numKvHeads, _kvGroup);
            kMh.Dispose();
            vMh.Dispose();
        }

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnMh = new Tensor(qMhShape, act);
        // allowF16: Krea2 applies zero-centered RMSNorm to Q and K (per-head, over headDim) before attention, so
        // pre-softmax scores are bounded and the F16 SDPA path (half the score-matrix HBM traffic + F16 tensor
        // cores) is numerically safe — same condition Wan/LTX use. SDPA is ~55% of Krea2 GPU time (sync-profiled).
        backend.ScaledDotProductAttention(attnMh, qMh, kRep, vRep, null, scale, allowF16: true);
        qMh.Dispose();
        kRep.Dispose();   // when _kvGroup == 1, kRep/vRep alias kMh/vMh — disposed exactly once here.
        vRep.Dispose();

        // Permute [B, H, S, D] → [B, S, hidden].
        Tensor attnFlat = new Tensor(flatShape, act);
        backend.Permute0213(attnFlat, attnMh, _numHeads, seqLen, _headDim);
        attnMh.Dispose();

        // out = attn · sigmoid(gate)
        Tensor sig = new Tensor(flatShape, act);
        backend.Sigmoid(sig, gate);
        gate.Dispose();
        Tensor gated = new Tensor(flatShape, act);
        backend.Mul(gated, attnFlat, sig);
        attnFlat.Dispose();
        sig.Dispose();

        Tensor outp = new Tensor(flatShape, act);
        backend.Linear(outp, gated, _toOut!, null);
        gated.Dispose();
        return outp;
    }
}

/// <summary>Per-process activation dtype for the Krea2 DiT block/attention hot path. <c>HARTSY_KREA2_F16=1</c> runs the
/// 28-block loop in F16 (half the HBM traffic of the bandwidth-bound norm/modulate/gate/attention kernels — the measured
/// Krea2 bottleneck) while the once-per-forward text/image/timestep paths and the tiny per-channel modulation vectors
/// stay F32. Default F32 reproduces the baseline exactly. Weights stay packed fp8 (activation-quant GEMM), so VRAM is
/// unchanged. QK-norm makes F16 attention numerically safe; the one risk is SwiGLU exceeding F16's 65504 ceiling.</summary>
public static class Krea2Dtype
{
    public static readonly DType Act =
        Environment.GetEnvironmentVariable("HARTSY_KREA2_F16") == "1" ? DType.F16 : DType.F32;
}

/// <summary>Helper for Krea 2's zero-centered RMSNorm scales (<c>F.rms_norm(x, weight = weight + 1)</c>): loads a norm
/// weight as F32 and folds the <c>+1</c> so the runtime can use plain <see cref="IBackend.RmsNorm"/>.</summary>
public static unsafe class Krea2Norm
{
    /// <summary>Returns an F32 copy of <paramref name="raw"/> with 1.0 added to every element.</summary>
    public static Tensor LoadZeroCentered(Tensor raw)
    {
        Tensor f32 = raw.DType == DType.F32 ? CopyF32(raw) : raw.CastTo(DType.F32);
        float* p = (float*)f32.DataPointer;
        long n = f32.Shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] += 1.0f;
        return f32;
    }

    private static Tensor CopyF32(Tensor raw)
    {
        Tensor copy = new Tensor(raw.Shape, DType.F32);
        long bytes = raw.Shape.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)raw.DataPointer, (void*)copy.DataPointer, bytes, bytes);
        return copy;
    }
}

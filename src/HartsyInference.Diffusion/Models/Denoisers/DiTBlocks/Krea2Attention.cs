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
    public Tensor Forward(IBackend backend, Tensor x, FluxRope? rope, int batch, int seqLen)
    {
        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;

        Tensor q = Linear(backend, x, _toQ!, batch, seqLen, qDim);
        Tensor k = Linear(backend, x, _toK!, batch, seqLen, kvDim);
        Tensor v = Linear(backend, x, _toV!, batch, seqLen, kvDim);
        Tensor gate = Linear(backend, x, _toGate!, batch, seqLen, _hidden);

        Tensor qMh = DiTUtils.ReshapeToMultiHead(q, batch, seqLen, _numHeads, _headDim);
        Tensor kMh = DiTUtils.ReshapeToMultiHead(k, batch, seqLen, _numKvHeads, _headDim);
        Tensor vMh = DiTUtils.ReshapeToMultiHead(v, batch, seqLen, _numKvHeads, _headDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor qNorm = new Tensor(qMh.Shape, DType.F32);
        Tensor kNorm = new Tensor(kMh.Shape, DType.F32);
        backend.RmsNorm(qNorm, qMh, _normQ!, _eps);
        backend.RmsNorm(kNorm, kMh, _normK!, _eps);
        qMh.Dispose(); kMh.Dispose();

        if (rope is not null)
        {
            rope.ForwardSingle(qNorm, batch, _numHeads, seqLen);
            rope.ForwardSingle(kNorm, batch, _numKvHeads, seqLen);
        }

        Tensor kRep = RepeatKvHeads(kNorm, batch, _numKvHeads, _kvGroup, seqLen, _headDim);
        Tensor vRep = RepeatKvHeads(vMh, batch, _numKvHeads, _kvGroup, seqLen, _headDim);
        kNorm.Dispose(); vMh.Dispose();

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnMh = new Tensor(new TensorShape(batch, _numHeads, seqLen, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attnMh, qNorm, kRep, vRep, null, scale);
        qNorm.Dispose(); kRep.Dispose(); vRep.Dispose();

        Tensor attnFlat = DiTUtils.ReshapeFromMultiHead(attnMh, batch, seqLen, _numHeads, _headDim);
        attnMh.Dispose();

        // out = attn · sigmoid(gate)
        Tensor sig = new Tensor(gate.Shape, DType.F32);
        backend.Sigmoid(sig, gate);
        gate.Dispose();
        Tensor gated = new Tensor(attnFlat.Shape, DType.F32);
        backend.Mul(gated, attnFlat, sig);
        attnFlat.Dispose(); sig.Dispose();

        Tensor outp = Linear(backend, gated, _toOut!, batch, seqLen, _hidden);
        gated.Dispose();
        return outp;
    }

    private static Tensor Linear(IBackend backend, Tensor input, Tensor weight, int batch, int seqLen, int outDim)
    {
        Tensor output = new Tensor(new TensorShape(batch, seqLen, outDim), DType.F32);
        backend.Linear(output, input, weight, null);
        return output;
    }

    private static Tensor RepeatKvHeads(Tensor input, int batch, int kvHeads, int groupSize, int seqLen, int headDim)
    {
        if (groupSize == 1)
        {
            Tensor copy = new Tensor(new TensorShape(batch, kvHeads, seqLen, headDim), DType.F32);
            long bytes = (long)batch * kvHeads * seqLen * headDim * sizeof(float);
            Buffer.MemoryCopy((void*)input.DataPointer, (void*)copy.DataPointer, bytes, bytes);
            return copy;
        }
        int qHeads = kvHeads * groupSize;
        Tensor output = new Tensor(new TensorShape(batch, qHeads, seqLen, headDim), DType.F32);
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long perHead = (long)seqLen * headDim * sizeof(float);
        for (int b = 0; b < batch; b++)
            for (int kv = 0; kv < kvHeads; kv++)
            {
                long src = ((long)b * kvHeads + kv) * seqLen * headDim;
                for (int g = 0; g < groupSize; g++)
                {
                    long dst = ((long)b * qHeads + kv * groupSize + g) * seqLen * headDim;
                    Buffer.MemoryCopy(inPtr + src, outPtr + dst, perHead, perHead);
                }
            }
        return output;
    }
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

using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Lumina-Image-2.0 context refiner block. Refines Gemma-encoded caption tokens before they enter the main DiT. Same shape as <see cref="Lumina2Block"/> minus AdaLN — caption refinement is timestep-independent. Uses separate Q/K/V projections (not fused), grouped-query attention (numKvHeads), and Lumina 2.0 diffusers key naming. <c>norm1.weight</c> here is a plain RMSNorm scale, no <c>norm1.linear</c> sub-module.</summary>
public sealed unsafe class Lumina2ContextRefinerBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _qDim;
    private readonly int _kvDim;
    private readonly int _ffnDim;
    private readonly float _eps;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    private Tensor? _norm1Weight;
    private Tensor? _norm2Weight;

    private Tensor? _toQWeight;
    private Tensor? _toKWeight;
    private Tensor? _toVWeight;
    private Tensor? _toOutWeight;

    private Tensor? _ffnNorm1Weight;
    private Tensor? _ffnNorm2Weight;

    private Tensor? _ffWeight1;
    private Tensor? _ffWeight2;
    private Tensor? _ffWeight3;

    public Lumina2ContextRefinerBlock(int hiddenSize, int numHeads, int numKvHeads, int headDim, int ffnDim, float eps = 1e-5f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _qDim = numHeads * headDim;
        _kvDim = numKvHeads * headDim;
        _ffnDim = ffnDim;
        _eps = eps;

        _normQ = new QkNorm(_headDim, eps);
        _normK = new QkNorm(_headDim, eps);
    }

    /// <summary>Loads weights using Lumina 2.0's diffusers naming under the given prefix (e.g., "context_refiner.0").</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _norm1Weight = TensorCasts.LoadF32(weights, $"{prefix}.norm1.weight");
        _norm2Weight = TensorCasts.LoadF32(weights, $"{prefix}.norm2.weight");

        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _toOutWeight = weights[$"{prefix}.attn.to_out.0.weight"];

        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);

        _ffnNorm1Weight = TensorCasts.LoadF32(weights, $"{prefix}.ffn_norm1.weight");
        _ffnNorm2Weight = TensorCasts.LoadF32(weights, $"{prefix}.ffn_norm2.weight");

        _ffWeight1 = weights[$"{prefix}.feed_forward.linear_1.weight"];
        _ffWeight2 = weights[$"{prefix}.feed_forward.linear_2.weight"];
        _ffWeight3 = weights[$"{prefix}.feed_forward.linear_3.weight"];
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_norm1Weight is not null) yield return _norm1Weight;
        if (_norm2Weight is not null) yield return _norm2Weight;
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toOutWeight is not null) yield return _toOutWeight;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        if (_ffnNorm1Weight is not null) yield return _ffnNorm1Weight;
        if (_ffnNorm2Weight is not null) yield return _ffnNorm2Weight;
        if (_ffWeight1 is not null) yield return _ffWeight1;
        if (_ffWeight2 is not null) yield return _ffWeight2;
        if (_ffWeight3 is not null) yield return _ffWeight3;
    }

    /// <summary>Forward pass on caption tokens (no AdaLN). x: [B, capLen, hidden]. Diffusers Lumina 2.0 applies <c>norm1</c> RMSNorm before attention with no modulation; <c>norm2</c> RMSNorm post-attention; FFN sub-block mirrors the modulated path minus the (1+scale_mlp) and gate_mlp factors. RoPE is applied per-token if provided.</summary>
    public Tensor Forward(IBackend backend, Tensor x, ZImageRope? rope)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);

        // ── Attention sub-block: x = x + norm2(attn(norm1(x))) ──
        Tensor pre = new Tensor(shape, x.DType);
        backend.RmsNorm(pre, x, _norm1Weight!, _eps);

        TensorShape qShape = new TensorShape(batch, seqLen, _qDim);
        TensorShape kvShape = new TensorShape(batch, seqLen, _kvDim);

        Tensor q = new Tensor(qShape, x.DType);
        Tensor k = new Tensor(kvShape, x.DType);
        Tensor v = new Tensor(kvShape, x.DType);
        backend.Linear(q, pre, _toQWeight!, null);
        backend.Linear(k, pre, _toKWeight!, null);
        backend.Linear(v, pre, _toVWeight!, null);
        pre.Dispose();

        int qVecs = batch * seqLen * _numHeads;
        int kvVecs = batch * seqLen * _numKvHeads;
        Tensor qN = new Tensor(qShape, DType.F32);
        Tensor kN = new Tensor(kvShape, DType.F32);
        _normQ.Forward(qN, q, qVecs);
        _normK.Forward(kN, k, kvVecs);
        q.Dispose();
        k.Dispose();

        TensorShape qMhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        TensorShape kvMhShape = new TensorShape(batch, _numKvHeads, seqLen, _headDim);
        Tensor qMh = new Tensor(qMhShape, DType.F32);
        Tensor kMh = new Tensor(kvMhShape, DType.F32);
        Tensor vMh = new Tensor(kvMhShape, DType.F32);
        ReshapeToMultiHead(qMh, qN, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead(kMh, kN, batch, seqLen, _numKvHeads, _headDim);
        ReshapeToMultiHead(vMh, v, batch, seqLen, _numKvHeads, _headDim);
        qN.Dispose();
        kN.Dispose();
        v.Dispose();

        if (rope is not null)
        {
            rope.ForwardSingle(qMh, batch, _numHeads, seqLen);
            rope.ForwardSingle(kMh, batch, _numKvHeads, seqLen);
        }

        Tensor kFull, vFull;
        if (_numKvHeads != _numHeads)
        {
            int groupSize = _numHeads / _numKvHeads;
            kFull = new Tensor(qMhShape, DType.F32);
            vFull = new Tensor(qMhShape, DType.F32);
            RepeatKvHeads(kFull, kMh, batch, _numKvHeads, groupSize, seqLen, _headDim);
            RepeatKvHeads(vFull, vMh, batch, _numKvHeads, groupSize, seqLen, _headDim);
            kMh.Dispose();
            vMh.Dispose();
        }
        else
        {
            kFull = kMh;
            vFull = vMh;
        }

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(qMhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kFull, vFull, null, scale);
        qMh.Dispose();
        kFull.Dispose();
        vFull.Dispose();

        TensorShape qFlatShape = new TensorShape(batch, seqLen, _qDim);
        Tensor attnFlat = new Tensor(qFlatShape, DType.F32);
        ReshapeFromMultiHead(attnFlat, attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(shape, x.DType);
        backend.Linear(projected, attnFlat, _toOutWeight!, null);
        attnFlat.Dispose();

        Tensor postAttnNorm = new Tensor(shape, x.DType);
        backend.RmsNorm(postAttnNorm, projected, _norm2Weight!, _eps);
        projected.Dispose();

        Tensor afterAttn = new Tensor(shape, x.DType);
        backend.Add(afterAttn, x, postAttnNorm);
        postAttnNorm.Dispose();

        // ── FFN sub-block: x = x + ffn_norm2(ffn(ffn_norm1(x))) ──
        Tensor preFfn = new Tensor(shape, x.DType);
        backend.RmsNorm(preFfn, afterAttn, _ffnNorm1Weight!, _eps);

        Tensor ffnOut = ForwardSwiGlu(backend, preFfn, batch, seqLen);
        preFfn.Dispose();

        Tensor postFfnNorm = new Tensor(shape, x.DType);
        backend.RmsNorm(postFfnNorm, ffnOut, _ffnNorm2Weight!, _eps);
        ffnOut.Dispose();

        Tensor result = new Tensor(shape, x.DType);
        backend.Add(result, afterAttn, postFfnNorm);
        afterAttn.Dispose();
        postFfnNorm.Dispose();

        return result;
    }

    private Tensor ForwardSwiGlu(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnDim);

        Tensor gate = new Tensor(ffShape, input.DType);
        backend.Linear(gate, input, _ffWeight1!, null);
        Tensor gateActivated = new Tensor(ffShape, input.DType);
        backend.Silu(gateActivated, gate);
        gate.Dispose();

        Tensor up = new Tensor(ffShape, input.DType);
        backend.Linear(up, input, _ffWeight3!, null);

        Tensor gated = new Tensor(ffShape, input.DType);
        backend.Mul(gated, gateActivated, up);
        gateActivated.Dispose();
        up.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor output = new Tensor(outShape, input.DType);
        backend.Linear(output, gated, _ffWeight2!, null);
        gated.Dispose();

        return output;
    }

    private static void ReshapeToMultiHead(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    int outOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    private static void ReshapeFromMultiHead(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    int outOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    private static void RepeatKvHeads(Tensor output, Tensor input, int batch, int kvHeads, int groupSize, int seqLen, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        long perHead = (long)seqLen * headDim;
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < kvHeads; h++)
            {
                long srcOff = ((long)b * kvHeads + h) * perHead;
                for (int g = 0; g < groupSize; g++)
                {
                    int qHead = h * groupSize + g;
                    long dstOff = ((long)b * (kvHeads * groupSize) + qHead) * perHead;
                    Buffer.MemoryCopy(inPtr + srcOff, outPtr + dstOff, perHead * sizeof(float), perHead * sizeof(float));
                }
            }
        }
    }
}

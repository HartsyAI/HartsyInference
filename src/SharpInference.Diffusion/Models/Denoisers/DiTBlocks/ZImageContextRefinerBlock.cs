using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Z-Image context refiner block. Refines Qwen3-encoded caption tokens before they enter the main DiT. Same shape as <see cref="ZImageBlock"/> minus AdaLN — caption refinement is timestep-independent. Fused QKV like the main block. No attention biases.</summary>
public sealed unsafe class ZImageContextRefinerBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffnDim;
    private readonly float _eps;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    private Tensor? _qkvWeight;
    private Tensor? _attnOutWeight;

    private Tensor? _attnNorm1Weight;
    private Tensor? _attnNorm2Weight;
    private Tensor? _ffnNorm1Weight;
    private Tensor? _ffnNorm2Weight;

    private Tensor? _w1Weight, _w2Weight, _w3Weight;

    public ZImageContextRefinerBlock(int hiddenSize, int numHeads, int ffnDim, float eps = 1e-5f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _ffnDim = ffnDim;
        _eps = eps;

        _normQ = new QkNorm(_headDim, eps);
        _normK = new QkNorm(_headDim, eps);
    }

    /// <summary>Loads weights using Z-Image's Lumina-style naming under the given prefix (e.g., "context_refiner.0").</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _qkvWeight = weights[$"{prefix}.attention.qkv.weight"];
        _attnOutWeight = weights[$"{prefix}.attention.out.weight"];

        _normQ.LoadWeights(weights[$"{prefix}.attention.q_norm.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attention.k_norm.weight"]);

        _attnNorm1Weight = weights[$"{prefix}.attention_norm1.weight"];
        _attnNorm2Weight = weights[$"{prefix}.attention_norm2.weight"];
        _ffnNorm1Weight = weights[$"{prefix}.ffn_norm1.weight"];
        _ffnNorm2Weight = weights[$"{prefix}.ffn_norm2.weight"];

        _w1Weight = weights[$"{prefix}.feed_forward.w1.weight"];
        _w2Weight = weights[$"{prefix}.feed_forward.w2.weight"];
        _w3Weight = weights[$"{prefix}.feed_forward.w3.weight"];
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_qkvWeight is not null) yield return _qkvWeight;
        if (_attnOutWeight is not null) yield return _attnOutWeight;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        if (_attnNorm1Weight is not null) yield return _attnNorm1Weight;
        if (_attnNorm2Weight is not null) yield return _attnNorm2Weight;
        if (_ffnNorm1Weight is not null) yield return _ffnNorm1Weight;
        if (_ffnNorm2Weight is not null) yield return _ffnNorm2Weight;
        if (_w1Weight is not null) yield return _w1Weight;
        if (_w2Weight is not null) yield return _w2Weight;
        if (_w3Weight is not null) yield return _w3Weight;
    }

    /// <summary>Forward pass on caption tokens (no AdaLN). x: [B, capLen, hidden]. <paramref name="rope"/> is the caption-token RoPE — diffusers' ZImageTransformerBlock always applies the freqs_cis it receives, even with modulation=False, so context_refiner DOES apply RoPE to caption tokens. Caption pos IDs run frame=1..capPaddedLen on axis 0 (h=w=0).</summary>
    public Tensor Forward(IBackend backend, Tensor x, ZImageRope? rope)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);

        // ── Attention sub-block: x = x + AttnNorm2(Attn(AttnNorm1(x))) ──
        Tensor pre = new Tensor(shape, x.DType);
        backend.RmsNorm(pre, x, _attnNorm1Weight!, _eps);

        TensorShape qkvShape = new TensorShape(batch, seqLen, 3 * _hiddenSize);
        Tensor qkv = new Tensor(qkvShape, x.DType);
        backend.Linear(qkv, pre, _qkvWeight!, null);
        pre.Dispose();

        Tensor q = new Tensor(shape, x.DType);
        Tensor k = new Tensor(shape, x.DType);
        Tensor v = new Tensor(shape, x.DType);
        SplitQkv(qkv, q, k, v, batch, seqLen, _hiddenSize);
        qkv.Dispose();

        int totalVecs = batch * seqLen * _numHeads;
        Tensor qN = new Tensor(q.Shape, DType.F32);
        Tensor kN = new Tensor(k.Shape, DType.F32);
        _normQ.Forward(qN, q, totalVecs);
        _normK.Forward(kN, k, totalVecs);
        q.Dispose();
        k.Dispose();

        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        Tensor qMh = new Tensor(mhShape, DType.F32);
        Tensor kMh = new Tensor(mhShape, DType.F32);
        Tensor vMh = new Tensor(mhShape, DType.F32);
        ReshapeToMultiHead(qMh, qN, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead(kMh, kN, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead(vMh, v, batch, seqLen, _numHeads, _headDim);
        qN.Dispose();
        kN.Dispose();
        v.Dispose();

        rope?.Forward(qMh, kMh, batch, _numHeads, seqLen);

        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, null, scale);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        Tensor attnFlat = new Tensor(shape, DType.F32);
        ReshapeFromMultiHead(attnFlat, attnOut, batch, seqLen, _numHeads, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(shape, x.DType);
        backend.Linear(projected, attnFlat, _attnOutWeight!, null);
        attnFlat.Dispose();

        Tensor postAttnNorm = new Tensor(shape, x.DType);
        backend.RmsNorm(postAttnNorm, projected, _attnNorm2Weight!, _eps);
        projected.Dispose();

        Tensor afterAttn = new Tensor(shape, x.DType);
        backend.Add(afterAttn, x, postAttnNorm);
        postAttnNorm.Dispose();

        // ── FFN sub-block ──
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

    private static void SplitQkv(Tensor qkv, Tensor q, Tensor k, Tensor v, int batch, int seqLen, int hidden)
    {
        float* srcPtr = (float*)qkv.DataPointer;
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;
        float* vPtr = (float*)v.DataPointer;

        long bytesPerSlice = (long)hidden * sizeof(float);
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                long srcBase = ((long)b * seqLen + s) * 3 * hidden;
                long dstBase = ((long)b * seqLen + s) * hidden;
                Buffer.MemoryCopy(srcPtr + srcBase, qPtr + dstBase, bytesPerSlice, bytesPerSlice);
                Buffer.MemoryCopy(srcPtr + srcBase + hidden, kPtr + dstBase, bytesPerSlice, bytesPerSlice);
                Buffer.MemoryCopy(srcPtr + srcBase + 2 * hidden, vPtr + dstBase, bytesPerSlice, bytesPerSlice);
            }
        }
    }

    private Tensor ForwardSwiGlu(IBackend backend, Tensor input, int batch, int seqLen)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnDim);

        Tensor gate = new Tensor(ffShape, input.DType);
        backend.Linear(gate, input, _w1Weight!, null);
        Tensor gateActivated = new Tensor(ffShape, input.DType);
        backend.Silu(gateActivated, gate);
        gate.Dispose();

        Tensor linear = new Tensor(ffShape, input.DType);
        backend.Linear(linear, input, _w3Weight!, null);

        Tensor gated = new Tensor(ffShape, input.DType);
        backend.Mul(gated, gateActivated, linear);
        gateActivated.Dispose();
        linear.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor output = new Tensor(outShape, input.DType);
        backend.Linear(output, gated, _w2Weight!, null);
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
}

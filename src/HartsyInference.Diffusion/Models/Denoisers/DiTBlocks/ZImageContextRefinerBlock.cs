using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Z-Image context refiner block. Refines Qwen3-encoded caption tokens before they enter the main DiT. Same shape as <see cref="ZImageBlock"/> minus AdaLN — caption refinement is timestep-independent. Fused QKV like the main block. No attention biases.</summary>
public sealed unsafe class ZImageContextRefinerBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffnDim;
    private readonly float _eps;
    private readonly bool _allowF16Attention;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    // Fused QKV split into separate Q/K/V weights at load (GPU-residency rewrite); see ZImageBlock.SplitQkv.
    private Tensor? _toQWeight, _toKWeight, _toVWeight;
    private Tensor? _attnOutWeight;

    private Tensor? _attnNorm1Weight;
    private Tensor? _attnNorm2Weight;
    private Tensor? _ffnNorm1Weight;
    private Tensor? _ffnNorm2Weight;

    private Tensor? _w1Weight, _w2Weight, _w3Weight;

    public ZImageContextRefinerBlock(int hiddenSize, int numHeads, int ffnDim, float eps = 1e-5f,
        bool allowF16Attention = true)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _ffnDim = ffnDim;
        _eps = eps;
        _allowF16Attention = allowF16Attention;

        _normQ = new QkNorm(_headDim, eps);
        _normK = new QkNorm(_headDim, eps);
    }

    /// <summary>Loads weights using Z-Image's Lumina-style naming under the given prefix (e.g., "context_refiner.0").</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        (_toQWeight, _toKWeight, _toVWeight) = ZImageBlock.SplitQkv(weights[$"{prefix}.attention.qkv.weight"], _hiddenSize);
        _attnOutWeight = weights[$"{prefix}.attention.out.weight"];

        _normQ.LoadWeights(weights[$"{prefix}.attention.q_norm.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attention.k_norm.weight"]);

        // RMSNorm scales must be F32 (CudaBackend.RmsNorm reads weight as float* directly).
        _attnNorm1Weight = TensorCasts.LoadF32(weights, $"{prefix}.attention_norm1.weight");
        _attnNorm2Weight = TensorCasts.LoadF32(weights, $"{prefix}.attention_norm2.weight");
        _ffnNorm1Weight = TensorCasts.LoadF32(weights, $"{prefix}.ffn_norm1.weight");
        _ffnNorm2Weight = TensorCasts.LoadF32(weights, $"{prefix}.ffn_norm2.weight");

        _w1Weight = weights[$"{prefix}.feed_forward.w1.weight"];
        _w2Weight = weights[$"{prefix}.feed_forward.w2.weight"];
        _w3Weight = weights[$"{prefix}.feed_forward.w3.weight"];
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toVWeight is not null) yield return _toVWeight;
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
        TensorShape headsShape = new TensorShape(batch, seqLen, _numHeads, _headDim);
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);

        // GPU-residency rewrite — see ZImageBlock.Forward for the full rationale. This block has no AdaLN
        // (caption refinement is timestep-independent), so the residuals stay plain backend.Add.

        // ── Attention sub-block: x = x + AttnNorm2(Attn(AttnNorm1(x))) ──
        Tensor pre = new Tensor(shape, DType.F32);
        backend.RmsNorm(pre, x, _attnNorm1Weight!, _eps);

        Tensor qHeads = new Tensor(headsShape, DType.F32);
        Tensor kHeads = new Tensor(headsShape, DType.F32);
        Tensor v = new Tensor(headsShape, DType.F32);
        backend.Linear(qHeads, pre, _toQWeight!, null);
        backend.Linear(kHeads, pre, _toKWeight!, null);
        backend.Linear(v, pre, _toVWeight!, null);
        pre.Dispose();

        Tensor qNormed = new Tensor(headsShape, DType.F32);
        Tensor kNormed = new Tensor(headsShape, DType.F32);
        backend.RmsNorm(qNormed, qHeads, _normQ.Weight, _normQ.Eps);
        backend.RmsNorm(kNormed, kHeads, _normK.Weight, _normK.Eps);
        qHeads.Dispose();
        kHeads.Dispose();

        // Device RoPE on the pre-permute layout for B=1 (see ZImageBlock — bit-identical, kills the host
        // D2H round-trip of Q/K); B>1 keeps the host path after the permute.
        bool gpuRope = rope is not null && batch == 1;
        if (gpuRope)
        {
            rope!.ApplyGpu(backend, qNormed, kNormed, _numHeads);
        }

        Tensor qMh = new Tensor(mhShape, DType.F32);
        Tensor kMh = new Tensor(mhShape, DType.F32);
        Tensor vMh = new Tensor(mhShape, DType.F32);
        backend.Permute0213(qMh, qNormed, seqLen, _numHeads, _headDim);
        backend.Permute0213(kMh, kNormed, seqLen, _numHeads, _headDim);
        backend.Permute0213(vMh, v, seqLen, _numHeads, _headDim);
        qNormed.Dispose();
        kNormed.Dispose();
        v.Dispose();

        if (rope is not null && !gpuRope)
        {
            rope.Forward(qMh, kMh, batch, _numHeads, seqLen);
        }

        // Q/K normalization bounds scores but not V. Base disables internal F16 narrowing for every attention
        // site; Turbo retains its validated fused path.
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, null, scale,
            allowF16: _allowF16Attention);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();

        Tensor attnFlat = new Tensor(shape, DType.F32);
        backend.Permute0213(attnFlat, attnOut, _numHeads, seqLen, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(shape, DType.F32);
        backend.Linear(projected, attnFlat, _attnOutWeight!, null);
        attnFlat.Dispose();

        Tensor postAttnNorm = new Tensor(shape, DType.F32);
        backend.RmsNorm(postAttnNorm, projected, _attnNorm2Weight!, _eps);
        projected.Dispose();

        Tensor afterAttn = new Tensor(shape, DType.F32);
        backend.Add(afterAttn, x, postAttnNorm);
        postAttnNorm.Dispose();

        // ── FFN sub-block ──
        Tensor preFfn = new Tensor(shape, DType.F32);
        backend.RmsNorm(preFfn, afterAttn, _ffnNorm1Weight!, _eps);

        Tensor ffnOut = ForwardSwiGlu(backend, preFfn, batch, seqLen);
        preFfn.Dispose();

        Tensor postFfnNorm = new Tensor(shape, DType.F32);
        backend.RmsNorm(postFfnNorm, ffnOut, _ffnNorm2Weight!, _eps);
        ffnOut.Dispose();

        Tensor result = new Tensor(shape, DType.F32);
        backend.Add(result, afterAttn, postFfnNorm);
        afterAttn.Dispose();
        postFfnNorm.Dispose();

        return result;
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

}

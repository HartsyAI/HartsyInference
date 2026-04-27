using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>
/// Flux.2 single-stream block. <b>Parallel</b> transformer block (ViT-22B style): the QKV
/// projections and the SwiGLU MLP gate/up projections all share a single fused input linear
/// (<c>linear1</c>); the attention output and the SwiGLU MLP output are concatenated and projected
/// back to hidden via a single fused output linear (<c>linear2</c>).
/// <para>The block runs on the concatenated <c>[txt, img]</c> sequence (concatenation is done by
/// the transformer before the first single block). Modulation is shared across all single blocks
/// and produces 3 params <c>(shift, scale, gate)</c>.</para>
/// <para>Per-token flow:
/// <c>norm(LayerNorm) → modulate → linear1 → split [Q,K,V,gate,up] → QK-RMSNorm → RoPE → SDPA → concat[attn, silu(gate)*up] → linear2 → gated residual</c>.
/// </para>
/// In the BFL checkpoint the fused weights are stored as:
/// <c>single_blocks.{i}.linear1.weight: [3*hidden + 2*mlp_inner, hidden]</c> (rows 0..3*hidden = QKV, then gate, then up)
/// and <c>single_blocks.{i}.linear2.weight: [hidden, hidden + mlp_inner]</c>.
/// The converter splits <c>linear1</c> into 5 separate weights so QKV and SwiGLU can be computed
/// as independent matmuls; <c>linear2</c> is kept fused.
/// </summary>
public sealed unsafe class Flux2SingleBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpInner;
    private readonly float _layerNormEps;
    private readonly bool _qkvBias;

    // Split from linear1 by the converter (no bias for Flux.2)
    private Tensor? _toQWeight, _toQBias;
    private Tensor? _toKWeight, _toKBias;
    private Tensor? _toVWeight, _toVBias;
    private Tensor? _gateWeight, _gateBias;
    private Tensor? _upWeight, _upBias;

    // linear2: fused [attn_out (hidden) || swiglu_out (mlp_inner)] → hidden
    private Tensor? _toOutWeight, _toOutBias;

    // Per-head Q/K RMSNorm (pre-RoPE)
    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    public Flux2SingleBlock(int hiddenSize, int numHeads, int mlpInner,
        bool qkvBias = false, float qkNormEps = 1e-6f, float layerNormEps = 1e-6f)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _mlpInner = mlpInner;
        _layerNormEps = layerNormEps;
        _qkvBias = qkvBias;

        _normQ = new QkNorm(_headDim, qkNormEps);
        _normK = new QkNorm(_headDim, qkNormEps);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        // Split-from-linear1 projections (converter responsibility — see Flux2CheckpointConverter)
        _toQWeight = weights[$"{prefix}.attn.to_q.weight"];
        _toKWeight = weights[$"{prefix}.attn.to_k.weight"];
        _toVWeight = weights[$"{prefix}.attn.to_v.weight"];
        _gateWeight = weights[$"{prefix}.attn.linear_in_gate.weight"];
        _upWeight = weights[$"{prefix}.attn.linear_in_up.weight"];

        if (_qkvBias)
        {
            _toQBias = weights[$"{prefix}.attn.to_q.bias"];
            _toKBias = weights[$"{prefix}.attn.to_k.bias"];
            _toVBias = weights[$"{prefix}.attn.to_v.bias"];
            _gateBias = weights[$"{prefix}.attn.linear_in_gate.bias"];
            _upBias = weights[$"{prefix}.attn.linear_in_up.bias"];
        }

        // Fused linear2 (kept whole — input dim = hidden + mlp_inner, output = hidden)
        _toOutWeight = weights[$"{prefix}.attn.to_out.weight"];
        if (_qkvBias)
            _toOutBias = weights[$"{prefix}.attn.to_out.bias"];

        // Per-head Q/K RMSNorm
        _normQ.LoadWeights(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attn.norm_k.weight"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toQBias is not null) yield return _toQBias;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toKBias is not null) yield return _toKBias;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toVBias is not null) yield return _toVBias;
        if (_gateWeight is not null) yield return _gateWeight;
        if (_gateBias is not null) yield return _gateBias;
        if (_upWeight is not null) yield return _upWeight;
        if (_upBias is not null) yield return _upBias;
        if (_toOutWeight is not null) yield return _toOutWeight;
        if (_toOutBias is not null) yield return _toOutBias;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
    }

    /// <summary>
    /// Forward pass on the concatenated <c>[txt, img]</c> sequence. <paramref name="mod"/> has 3
    /// elements: <c>(shift, scale, gate)</c>, shape <c>[B, hidden]</c>, shared across all single
    /// blocks (computed once at the top level).
    /// </summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor[] mod, FluxRope rope)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];

        TensorShape hiddenShape = new TensorShape(batch, seqLen, _hiddenSize);
        TensorShape mlpShape = new TensorShape(batch, seqLen, _mlpInner);
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);

        // ── 1. LayerNorm + modulate ──
        Tensor normed = new Tensor(hiddenShape, DType.F32);
        LayerNormNoAffine(normed, hidden, batch, seqLen, _hiddenSize, _layerNormEps);
        Tensor modulated = AdaLNModulation.ApplyModulation(normed, mod[0], mod[1], batch, seqLen, _hiddenSize);
        normed.Dispose();

        // ── 2. Q/K/V projections + gate/up MLP projections (each from same modulated input) ──
        Tensor q = new Tensor(hiddenShape, DType.F32);
        backend.Linear(q, modulated, _toQWeight!, _toQBias);
        Tensor k = new Tensor(hiddenShape, DType.F32);
        backend.Linear(k, modulated, _toKWeight!, _toKBias);
        Tensor v = new Tensor(hiddenShape, DType.F32);
        backend.Linear(v, modulated, _toVWeight!, _toVBias);
        Tensor gate = new Tensor(mlpShape, DType.F32);
        backend.Linear(gate, modulated, _gateWeight!, _gateBias);
        Tensor up = new Tensor(mlpShape, DType.F32);
        backend.Linear(up, modulated, _upWeight!, _upBias);
        modulated.Dispose();

        // ── 3. QK-Norm (per-head RMSNorm pre-RoPE) ──
        int vectors = batch * seqLen * _numHeads;
        Tensor qNormed = new Tensor(q.Shape, DType.F32);
        Tensor kNormed = new Tensor(k.Shape, DType.F32);
        _normQ.Forward(qNormed, q, vectors);
        _normK.Forward(kNormed, k, vectors);
        q.Dispose();
        k.Dispose();

        // ── 4. Reshape to multi-head ──
        Tensor qMh = new Tensor(mhShape, DType.F32);
        Tensor kMh = new Tensor(mhShape, DType.F32);
        Tensor vMh = new Tensor(mhShape, DType.F32);
        ReshapeToMultiHead(qMh, qNormed, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead(kMh, kNormed, batch, seqLen, _numHeads, _headDim);
        ReshapeToMultiHead(vMh, v, batch, seqLen, _numHeads, _headDim);
        qNormed.Dispose(); kNormed.Dispose(); v.Dispose();

        // ── 5. RoPE (4-axis pairwise rotation on Q/K) ──
        rope.Forward(qMh, kMh, batch, _numHeads, seqLen);

        // ── 6. SDPA ──
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOutMh = new Tensor(mhShape, DType.F32);
        backend.ScaledDotProductAttention(attnOutMh, qMh, kMh, vMh, null, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        // ── 7. Reshape attn output back to [B, S, hidden] ──
        Tensor attnOut = new Tensor(hiddenShape, DType.F32);
        ReshapeFromMultiHead(attnOut, attnOutMh, batch, seqLen, _numHeads, _headDim);
        attnOutMh.Dispose();

        // ── 8. SwiGLU activation: silu(gate) * up ──
        Tensor swigluOut = new Tensor(mlpShape, DType.F32);
        SwiGluActivation(swigluOut, gate, up, batch * seqLen * _mlpInner);
        gate.Dispose(); up.Dispose();

        // ── 9. Concatenate [attn_out (hidden) || swiglu_out (mlp_inner)] along last dim ──
        TensorShape concatShape = new TensorShape(batch, seqLen, _hiddenSize + _mlpInner);
        Tensor concat = new Tensor(concatShape, DType.F32);
        ConcatLastDim(concat, attnOut, swigluOut, batch * seqLen, _hiddenSize, _mlpInner);
        attnOut.Dispose(); swigluOut.Dispose();

        // ── 10. Fused output projection: [hidden + mlp_inner] → hidden ──
        Tensor proj = new Tensor(hiddenShape, DType.F32);
        backend.Linear(proj, concat, _toOutWeight!, _toOutBias);
        concat.Dispose();

        // ── 11. Gated residual ──
        Tensor result = AdaLNModulation.ApplyGatedResidual(hidden, proj, mod[2], batch, seqLen, _hiddenSize);
        proj.Dispose();
        return result;
    }

    private static void SwiGluActivation(Tensor output, Tensor gate, Tensor up, int totalElements)
    {
        float* gPtr = (float*)gate.DataPointer;
        float* uPtr = (float*)up.DataPointer;
        float* oPtr = (float*)output.DataPointer;
        for (int i = 0; i < totalElements; i++)
        {
            float g = gPtr[i];
            float silu = g / (1.0f + MathF.Exp(-g));
            oPtr[i] = silu * uPtr[i];
        }
    }

    private static void ConcatLastDim(Tensor output, Tensor first, Tensor second,
        int rowCount, int firstDim, int secondDim)
    {
        float* fPtr = (float*)first.DataPointer;
        float* sPtr = (float*)second.DataPointer;
        float* oPtr = (float*)output.DataPointer;
        int totalDim = firstDim + secondDim;
        for (int r = 0; r < rowCount; r++)
        {
            Buffer.MemoryCopy(fPtr + r * firstDim, oPtr + r * totalDim,
                firstDim * sizeof(float), firstDim * sizeof(float));
            Buffer.MemoryCopy(sPtr + r * secondDim, oPtr + r * totalDim + firstDim,
                secondDim * sizeof(float), secondDim * sizeof(float));
        }
    }

    private static void LayerNormNoAffine(Tensor output, Tensor input, int batch, int seqLen, int dim, float eps)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int offset = (b * seqLen + s) * dim;
                float mean = 0f;
                for (int d = 0; d < dim; d++) mean += inPtr[offset + d];
                mean /= dim;
                float variance = 0f;
                for (int d = 0; d < dim; d++) { float diff = inPtr[offset + d] - mean; variance += diff * diff; }
                variance /= dim;
                float invStd = 1.0f / MathF.Sqrt(variance + eps);
                for (int d = 0; d < dim; d++) outPtr[offset + d] = (inPtr[offset + d] - mean) * invStd;
            }
        }
    }

    private static void ReshapeToMultiHead(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    int outOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
    }

    private static void ReshapeFromMultiHead(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    int outOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
    }
}

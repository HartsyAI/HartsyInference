using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Flux.2 single-stream parallel block (ViT-22B style): QKV and SwiGLU gate/up share one fused input linear, attn-out and MLP-out share one fused output linear. Runs on concatenated [txt,img]; modulation shared across all single blocks (3 params: shift/scale/gate). Flow: LayerNorm → modulate → linear1 → split [Q,K,V,gate,up] → QK-RMSNorm → RoPE → SDPA → concat[attn, silu(gate)*up] → linear2 → gated residual. The converter splits BFL's fused linear1 into 5 independent weights; linear2 is kept fused.</summary>
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

    /// <param name="branchDamp">Residual-stream damp for the F16 activation path (see <see cref="ChromaF16"/>); damps the fused output projection (the only write into the residual stream). 1.0 = off.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix, float branchDamp = 1.0f)
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

        if (branchDamp != 1.0f)
        {
            // Weight damp rides the GEMM alpha; bias (qkvBias variants only) gets a value-scaled copy.
            _toOutWeight!.Fp8ScaleFactor *= branchDamp;
            if (_toOutBias is not null) _toOutBias = ChromaF16.DampBias(_toOutBias, branchDamp);
        }
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

    /// <summary>Forward pass on the concatenated <c>[txt, img]</c> sequence. <paramref name="mod"/> has 3 elements <c>(shift, scale, gate)</c>, shape <c>[B, hidden]</c>, shared across all single blocks (computed once at the top level).</summary>
    // GPU-residency rewrite (mirrors the verified FluxSingleStreamBlock / the Flux2DoubleBlock port): every glue
    // op (LayerNorm / modulation / QK-norm / head reshape / SwiGLU activation / last-dim concat / gated residual)
    // runs as an IBackend op so the activation stays device-resident across the whole block (Flux2SingleBlock
    // cpu=29 in the residency audit). Q/K/V declared [B, S, H, D] so QK-norm's RmsNorm runs over headDim with no
    // reshape; RoPE rotates pre-permute on-device (B=1; host fallback for batch).
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor[] mod, FluxRope rope, Tensor? attnBias = null)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];
        float scale = 1.0f / MathF.Sqrt(_headDim);

        // F16 activation path: activations follow the incoming stream dtype (see Flux2DoubleBlock).
        DType act = hidden.DType;
        TensorShape hiddenShape = new TensorShape(batch, seqLen, _hiddenSize);
        TensorShape mlpShape = new TensorShape(batch, seqLen, _mlpInner);
        TensorShape headsShape = new TensorShape(batch, seqLen, _numHeads, _headDim);
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);

        // ── 1. LayerNorm (no affine) + modulate x*(1+scale)+shift ──
        Tensor modulated = DiTUtils.NormModulate(backend, hidden, mod[0], mod[1], hiddenShape, _layerNormEps);

        // ── 2. Q/K/V projections ([B, S, H, D]) + gate/up MLP projections (same modulated input) ──
        Tensor q = new Tensor(headsShape, act);
        backend.Linear(q, modulated, _toQWeight!, _toQBias);
        Tensor k = new Tensor(headsShape, act);
        backend.Linear(k, modulated, _toKWeight!, _toKBias);
        Tensor v = new Tensor(headsShape, act);
        backend.Linear(v, modulated, _toVWeight!, _toVBias);
        Tensor gate = new Tensor(mlpShape, act);
        backend.Linear(gate, modulated, _gateWeight!, _gateBias);
        Tensor up = new Tensor(mlpShape, act);
        backend.Linear(up, modulated, _upWeight!, _upBias);
        modulated.Dispose();

        // ── 3. QK-Norm (per-head RMSNorm over the last dim = headDim, pre-RoPE) ──
        Tensor qNormed = new Tensor(headsShape, act);
        backend.RmsNorm(qNormed, q, _normQ.Weight, _normQ.Eps);
        q.Dispose();
        Tensor kNormed = new Tensor(headsShape, act);
        backend.RmsNorm(kNormed, k, _normK.Weight, _normK.Eps);
        k.Dispose();

        // ── 4. RoPE BEFORE the head permute (B=1 GPU path; identical pairwise rotation) ──
        if (batch == 1)
            rope.ApplyGpu(backend, qNormed, kNormed, _numHeads);

        // ── 5. Permute [B, S, H, D] → [B, H, S, D] ──
        Tensor qMh = new Tensor(mhShape, act);
        backend.Permute0213(qMh, qNormed, seqLen, _numHeads, _headDim);
        qNormed.Dispose();
        Tensor kMh = new Tensor(mhShape, act);
        backend.Permute0213(kMh, kNormed, seqLen, _numHeads, _headDim);
        kNormed.Dispose();
        Tensor vMh = new Tensor(mhShape, act);
        backend.Permute0213(vMh, v, seqLen, _numHeads, _headDim);
        v.Dispose();

        // Host RoPE fallback (batched inference only; B=1 runs the GPU path at 4) ──
        if (batch != 1)
            rope.Forward(qMh, kMh, batch, _numHeads, seqLen);

        // ── 6. SDPA ──
        Tensor attnOutMh = new Tensor(mhShape, act);
        backend.ScaledDotProductAttention(attnOutMh, qMh, kMh, vMh, attnBias, scale, allowF16: true);   // QK RMS-norm bounds scores; enables the cuDNN fused path (bias rides fp32 in-engine)
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        // ── 7. Permute attn output back [B, H, S, D] → [B, S, hidden] ──
        Tensor attnOut = new Tensor(hiddenShape, act);
        backend.Permute0213(attnOut, attnOutMh, _numHeads, seqLen, _headDim);
        attnOutMh.Dispose();

        // ── 8. SwiGLU activation: silu(gate) * up ──
        Tensor gateActivated = new Tensor(mlpShape, act);
        backend.Silu(gateActivated, gate);
        gate.Dispose();
        Tensor swigluOut = new Tensor(mlpShape, act);
        backend.Mul(swigluOut, gateActivated, up);
        gateActivated.Dispose();
        up.Dispose();

        // ── 9. Concatenate [attn_out (hidden) || swiglu_out (mlp_inner)] along the last dim ──
        TensorShape concatShape = new TensorShape(batch, seqLen, _hiddenSize + _mlpInner);
        Tensor concat = new Tensor(concatShape, act);
        backend.Concat(concat, new Tensor[] { attnOut, swigluOut }, 2);
        attnOut.Dispose(); swigluOut.Dispose();

        // ── 10. Fused output projection: [hidden + mlp_inner] → hidden ──
        Tensor proj = new Tensor(hiddenShape, act);
        backend.Linear(proj, concat, _toOutWeight!, _toOutBias);
        concat.Dispose();

        // ── 11. Gated residual ──
        Tensor result = new Tensor(hiddenShape, act);
        backend.GatedResidualLastDim(result, hidden, proj, mod[2]);
        proj.Dispose();
        return result;
    }

}

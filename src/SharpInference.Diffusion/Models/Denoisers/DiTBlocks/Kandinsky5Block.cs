using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Kandinsky 5 transformer encoder block (text stream).
/// Per <c>Kandinsky5TransformerEncoderBlock.forward</c>: produces 6 modulation params from a single
/// <c>Linear(time_embed → 6 * model_dim)</c> after SiLU; chunks them into
/// <c>(self_attn_params, ff_params)</c>, then each into <c>(shift, scale, gate)</c>; runs
/// non-affine LayerNorm then modulates pre-attention/pre-FFN; the post-attention/post-FFN combine
/// is <c>x + gate * sub_out</c>. Self-attention only — Q/K/V/Out all biased; QK norm is RMSNorm with
/// learnable scale. Self-attention applies 1D RoPE on Q and K.</summary>
public sealed unsafe class Kandinsky5EncoderBlock
{
    private readonly int _modelDim;
    private readonly int _timeDim;
    private readonly int _ffDim;
    private readonly int _headDim;
    private readonly int _numHeads;
    private readonly float _qkNormEps;

    private Tensor? _modWeight, _modBias;
    private Tensor? _qWeight, _qBias;
    private Tensor? _kWeight, _kBias;
    private Tensor? _vWeight, _vBias;
    private Tensor? _outWeight, _outBias;
    private Tensor? _qNormWeight;
    private Tensor? _kNormWeight;
    private Tensor? _ffIn, _ffOut;

    public Kandinsky5EncoderBlock(int modelDim, int timeDim, int ffDim, int headDim, float qkNormEps = 1e-5f)
    {
        if (modelDim % headDim != 0)
            throw new ArgumentException($"modelDim ({modelDim}) must be divisible by headDim ({headDim}).");

        _modelDim = modelDim;
        _timeDim = timeDim;
        _ffDim = ffDim;
        _headDim = headDim;
        _numHeads = modelDim / headDim;
        _qkNormEps = qkNormEps;
    }

    /// <summary>Loads weights with the diffusers naming convention: <c>{prefix}.{text_modulation,
    /// self_attention, feed_forward}.*</c>. Self-attention uses <c>to_query/to_key/to_value/out_layer</c>
    /// and RMSNorm <c>query_norm/key_norm</c>. Feed-forward uses <c>in_layer/out_layer</c> with no biases.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _modWeight = weights[$"{prefix}.text_modulation.out_layer.weight"];
        _modBias   = weights[$"{prefix}.text_modulation.out_layer.bias"];

        _qWeight   = weights[$"{prefix}.self_attention.to_query.weight"];
        _qBias     = weights[$"{prefix}.self_attention.to_query.bias"];
        _kWeight   = weights[$"{prefix}.self_attention.to_key.weight"];
        _kBias     = weights[$"{prefix}.self_attention.to_key.bias"];
        _vWeight   = weights[$"{prefix}.self_attention.to_value.weight"];
        _vBias     = weights[$"{prefix}.self_attention.to_value.bias"];
        _outWeight = weights[$"{prefix}.self_attention.out_layer.weight"];
        _outBias   = weights[$"{prefix}.self_attention.out_layer.bias"];

        _qNormWeight = weights[$"{prefix}.self_attention.query_norm.weight"];
        _kNormWeight = weights[$"{prefix}.self_attention.key_norm.weight"];

        _ffIn  = weights[$"{prefix}.feed_forward.in_layer.weight"];
        _ffOut = weights[$"{prefix}.feed_forward.out_layer.weight"];
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_modWeight is not null) yield return _modWeight;
        if (_modBias is not null) yield return _modBias;
        if (_qWeight is not null) yield return _qWeight;
        if (_qBias is not null) yield return _qBias;
        if (_kWeight is not null) yield return _kWeight;
        if (_kBias is not null) yield return _kBias;
        if (_vWeight is not null) yield return _vWeight;
        if (_vBias is not null) yield return _vBias;
        if (_outWeight is not null) yield return _outWeight;
        if (_outBias is not null) yield return _outBias;
        if (_qNormWeight is not null) yield return _qNormWeight;
        if (_kNormWeight is not null) yield return _kNormWeight;
        if (_ffIn is not null) yield return _ffIn;
        if (_ffOut is not null) yield return _ffOut;
    }

    /// <summary>Forward pass on text tokens <c>x = [B, S_text, D]</c> with timestep embedding
    /// <c>temb = [B, time_dim]</c> and 1D RoPE <c>rope</c> already precomputed for the same seq_len.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor temb, Kandinsky5Rope rope)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        int dim = _modelDim;

        Tensor[] mod = ProduceModulation(backend, temb, batch, dim);
        Tensor saShift = mod[0], saScale = mod[1], saGate = mod[2];
        Tensor ffShift = mod[3], ffScale = mod[4], ffGate = mod[5];

        // ── Self-attention sub-block ──
        Tensor saIn = LayerNormModulate(x, saShift, saScale, batch, seqLen, dim);

        Tensor q = NewTensor(batch, seqLen, dim);
        Tensor k = NewTensor(batch, seqLen, dim);
        Tensor v = NewTensor(batch, seqLen, dim);
        backend.Linear(q, saIn, _qWeight!, _qBias);
        backend.Linear(k, saIn, _kWeight!, _kBias);
        backend.Linear(v, saIn, _vWeight!, _vBias);
        saIn.Dispose();

        Tensor qMh = DiTUtils.ReshapeToMultiHead(q, batch, seqLen, _numHeads, _headDim);
        Tensor kMh = DiTUtils.ReshapeToMultiHead(k, batch, seqLen, _numHeads, _headDim);
        Tensor vMh = DiTUtils.ReshapeToMultiHead(v, batch, seqLen, _numHeads, _headDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        // QK-norm (RMSNorm) — apply across the head_dim axis on each [B,H,S] vector.
        ApplyRmsNormPerHead(backend, qMh, _qNormWeight!, batch, _numHeads, seqLen);
        ApplyRmsNormPerHead(backend, kMh, _kNormWeight!, batch, _numHeads, seqLen);

        // Apply RoPE on Q and K.
        rope.Apply(qMh, kMh, batch, _numHeads, seqLen);

        Tensor attnMh = NewTensor4D(batch, _numHeads, seqLen, _headDim);
        float scale = 1.0f / MathF.Sqrt(_headDim);
        backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMh, null, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor attnFlat = DiTUtils.ReshapeFromMultiHead(attnMh, batch, seqLen, _numHeads, _headDim);
        attnMh.Dispose();

        Tensor attnOut = NewTensor(batch, seqLen, dim);
        backend.Linear(attnOut, attnFlat, _outWeight!, _outBias);
        attnFlat.Dispose();

        Tensor afterAttn = AdaLNModulation.ApplyGatedResidual(x, attnOut, saGate, batch, seqLen, dim);
        attnOut.Dispose();

        // ── Feed-forward sub-block ──
        Tensor ffIn = LayerNormModulate(afterAttn, ffShift, ffScale, batch, seqLen, dim);

        Tensor ffMid = NewTensor(batch, seqLen, _ffDim);
        backend.Linear(ffMid, ffIn, _ffIn!, null);
        ffIn.Dispose();

        Tensor ffAct = NewTensor(batch, seqLen, _ffDim);
        backend.Gelu(ffAct, ffMid);
        ffMid.Dispose();

        Tensor ffOut = NewTensor(batch, seqLen, dim);
        backend.Linear(ffOut, ffAct, _ffOut!, null);
        ffAct.Dispose();

        Tensor result = AdaLNModulation.ApplyGatedResidual(afterAttn, ffOut, ffGate, batch, seqLen, dim);
        ffOut.Dispose();
        afterAttn.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();
        return result;
    }

    /// <summary>Computes <c>Linear(SiLU(temb)) → 6 * model_dim</c> then chunks into 6 <c>[B, model_dim]</c> tensors
    /// in the order produced by diffusers' two-level chunk: <c>(self_attn_shift, self_attn_scale, self_attn_gate,
    /// ff_shift, ff_scale, ff_gate)</c>.</summary>
    private Tensor[] ProduceModulation(IBackend backend, Tensor temb, int batch, int dim)
    {
        int outDim = 6 * dim;

        Tensor activated = new Tensor(new TensorShape(batch, _timeDim), DType.F32);
        backend.Silu(activated, temb);

        Tensor projected = new Tensor(new TensorShape(batch, outDim), DType.F32);
        backend.Linear(projected, activated, _modWeight!, _modBias);
        activated.Dispose();

        Tensor[] result = new Tensor[6];
        float* projPtr = (float*)projected.DataPointer;
        for (int p = 0; p < 6; p++)
        {
            Tensor slot = new Tensor(new TensorShape(batch, dim), DType.F32);
            float* slotPtr = (float*)slot.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                int srcOffset = b * outDim + p * dim;
                int dstOffset = b * dim;
                for (int d = 0; d < dim; d++)
                    slotPtr[dstOffset + d] = projPtr[srcOffset + d];
            }
            result[p] = slot;
        }

        projected.Dispose();
        return result;
    }

    /// <summary>Non-affine LayerNorm followed by <c>x * (1 + scale) + shift</c>.</summary>
    private static Tensor LayerNormModulate(Tensor x, Tensor shift, Tensor scale, int batch, int seqLen, int dim)
    {
        Tensor normed = new Tensor(new TensorShape(batch, seqLen, dim), DType.F32);
        DiTUtils.LayerNormNoAffine(normed, x, batch, seqLen, dim);
        Tensor modulated = AdaLNModulation.ApplyModulation(normed, shift, scale, batch, seqLen, dim);
        normed.Dispose();
        return modulated;
    }

    private static Tensor NewTensor(int batch, int seqLen, int dim) =>
        new Tensor(new TensorShape(batch, seqLen, dim), DType.F32);

    private static Tensor NewTensor4D(int b, int h, int s, int d) =>
        new Tensor(new TensorShape(b, h, s, d), DType.F32);

    /// <summary>Applies RMSNorm across the last dim of a <c>[B, H, S, D]</c> tensor in-place using the
    /// per-head learnable scale weight. Uses the backend RmsNorm op by reshaping the leading dims as a
    /// single "row" dimension — RmsNorm normalizes the last axis only.</summary>
    private void ApplyRmsNormPerHead(IBackend backend, Tensor input, Tensor weight, int batch, int numHeads, int seqLen)
    {
        Tensor output = new Tensor(input.Shape, DType.F32);
        backend.RmsNorm(output, input, weight, _qkNormEps);

        long bytes = input.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)output.DataPointer, (void*)input.DataPointer, bytes, bytes);
        output.Dispose();
    }
}

/// <summary>Kandinsky 5 transformer decoder block (visual stream).
/// Per <c>Kandinsky5TransformerDecoderBlock.forward</c>: produces 9 modulation params from a single
/// <c>Linear(time_embed → 9 * model_dim)</c> after SiLU; chunks into
/// <c>(self_attn_params, cross_attn_params, ff_params)</c>; each into <c>(shift, scale, gate)</c>.
///
/// Three sub-blocks: (1) self-attention with 3D RoPE on Q/K, (2) cross-attention to text (no RoPE,
/// no QK-norm scaling on the rope path), (3) FFN. All sub-LayerNorms are non-affine; QKV/out linears
/// are biased; FFN is bias-free <c>Linear → GELU → Linear</c>; QK norm is RMSNorm.</summary>
public sealed unsafe class Kandinsky5DecoderBlock
{
    private readonly int _modelDim;
    private readonly int _timeDim;
    private readonly int _ffDim;
    private readonly int _headDim;
    private readonly int _numHeads;
    private readonly float _qkNormEps;

    private Tensor? _modWeight, _modBias;

    private Tensor? _saQ, _saQB, _saK, _saKB, _saV, _saVB, _saO, _saOB;
    private Tensor? _saQNorm, _saKNorm;

    private Tensor? _xaQ, _xaQB, _xaK, _xaKB, _xaV, _xaVB, _xaO, _xaOB;
    private Tensor? _xaQNorm, _xaKNorm;

    private Tensor? _ffIn, _ffOut;

    public Kandinsky5DecoderBlock(int modelDim, int timeDim, int ffDim, int headDim, float qkNormEps = 1e-5f)
    {
        if (modelDim % headDim != 0)
            throw new ArgumentException($"modelDim ({modelDim}) must be divisible by headDim ({headDim}).");

        _modelDim = modelDim;
        _timeDim = timeDim;
        _ffDim = ffDim;
        _headDim = headDim;
        _numHeads = modelDim / headDim;
        _qkNormEps = qkNormEps;
    }

    /// <summary>Loads weights with the diffusers convention <c>{prefix}.{visual_modulation,
    /// self_attention, cross_attention, feed_forward}.*</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _modWeight = weights[$"{prefix}.visual_modulation.out_layer.weight"];
        _modBias   = weights[$"{prefix}.visual_modulation.out_layer.bias"];

        _saQ  = weights[$"{prefix}.self_attention.to_query.weight"];
        _saQB = weights[$"{prefix}.self_attention.to_query.bias"];
        _saK  = weights[$"{prefix}.self_attention.to_key.weight"];
        _saKB = weights[$"{prefix}.self_attention.to_key.bias"];
        _saV  = weights[$"{prefix}.self_attention.to_value.weight"];
        _saVB = weights[$"{prefix}.self_attention.to_value.bias"];
        _saO  = weights[$"{prefix}.self_attention.out_layer.weight"];
        _saOB = weights[$"{prefix}.self_attention.out_layer.bias"];
        _saQNorm = weights[$"{prefix}.self_attention.query_norm.weight"];
        _saKNorm = weights[$"{prefix}.self_attention.key_norm.weight"];

        _xaQ  = weights[$"{prefix}.cross_attention.to_query.weight"];
        _xaQB = weights[$"{prefix}.cross_attention.to_query.bias"];
        _xaK  = weights[$"{prefix}.cross_attention.to_key.weight"];
        _xaKB = weights[$"{prefix}.cross_attention.to_key.bias"];
        _xaV  = weights[$"{prefix}.cross_attention.to_value.weight"];
        _xaVB = weights[$"{prefix}.cross_attention.to_value.bias"];
        _xaO  = weights[$"{prefix}.cross_attention.out_layer.weight"];
        _xaOB = weights[$"{prefix}.cross_attention.out_layer.bias"];
        _xaQNorm = weights[$"{prefix}.cross_attention.query_norm.weight"];
        _xaKNorm = weights[$"{prefix}.cross_attention.key_norm.weight"];

        _ffIn  = weights[$"{prefix}.feed_forward.in_layer.weight"];
        _ffOut = weights[$"{prefix}.feed_forward.out_layer.weight"];
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _modWeight, _modBias,
            _saQ, _saQB, _saK, _saKB, _saV, _saVB, _saO, _saOB,
            _saQNorm, _saKNorm,
            _xaQ, _xaQB, _xaK, _xaKB, _xaV, _xaVB, _xaO, _xaOB,
            _xaQNorm, _xaKNorm,
            _ffIn, _ffOut,
        ];
        for (int i = 0; i < all.Length; i++)
        {
            Tensor? t = all[i];
            if (t is not null) yield return t;
        }
    }

    /// <summary>Forward pass on visual tokens <c>visual = [B, S_v, D]</c> with text <c>text = [B, S_t, D]</c>,
    /// timestep <c>temb = [B, time_dim]</c>, and 3D RoPE <c>rope</c> already precomputed for the visual seq.</summary>
    public Tensor Forward(IBackend backend, Tensor visual, Tensor text, Tensor temb, Kandinsky5Rope rope)
    {
        int batch = (int)visual.Shape[0];
        int sV = (int)visual.Shape[1];
        int sT = (int)text.Shape[1];
        int dim = _modelDim;

        Tensor[] mod = ProduceModulation(backend, temb, batch, dim);
        // Order: (sa_shift, sa_scale, sa_gate, xa_shift, xa_scale, xa_gate, ff_shift, ff_scale, ff_gate).
        Tensor saShift = mod[0], saScale = mod[1], saGate = mod[2];
        Tensor xaShift = mod[3], xaScale = mod[4], xaGate = mod[5];
        Tensor ffShift = mod[6], ffScale = mod[7], ffGate = mod[8];

        // ── 1. Self-attention with 3D RoPE on Q/K ──
        Tensor saIn = LayerNormModulate(visual, saShift, saScale, batch, sV, dim);

        Tensor q = NewTensor(batch, sV, dim);
        Tensor k = NewTensor(batch, sV, dim);
        Tensor v = NewTensor(batch, sV, dim);
        backend.Linear(q, saIn, _saQ!, _saQB);
        backend.Linear(k, saIn, _saK!, _saKB);
        backend.Linear(v, saIn, _saV!, _saVB);
        saIn.Dispose();

        Tensor qMh = DiTUtils.ReshapeToMultiHead(q, batch, sV, _numHeads, _headDim);
        Tensor kMh = DiTUtils.ReshapeToMultiHead(k, batch, sV, _numHeads, _headDim);
        Tensor vMh = DiTUtils.ReshapeToMultiHead(v, batch, sV, _numHeads, _headDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        ApplyRmsNormPerHead(backend, qMh, _saQNorm!);
        ApplyRmsNormPerHead(backend, kMh, _saKNorm!);
        rope.Apply(qMh, kMh, batch, _numHeads, sV);

        Tensor attnMh = NewTensor4D(batch, _numHeads, sV, _headDim);
        float saScaleVal = 1.0f / MathF.Sqrt(_headDim);
        backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMh, null, saScaleVal);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor attnFlat = DiTUtils.ReshapeFromMultiHead(attnMh, batch, sV, _numHeads, _headDim);
        attnMh.Dispose();

        Tensor saOut = NewTensor(batch, sV, dim);
        backend.Linear(saOut, attnFlat, _saO!, _saOB);
        attnFlat.Dispose();

        Tensor afterSa = AdaLNModulation.ApplyGatedResidual(visual, saOut, saGate, batch, sV, dim);
        saOut.Dispose();

        // ── 2. Cross-attention to text (no RoPE) ──
        Tensor xaIn = LayerNormModulate(afterSa, xaShift, xaScale, batch, sV, dim);

        Tensor qX = NewTensor(batch, sV, dim);
        backend.Linear(qX, xaIn, _xaQ!, _xaQB);
        xaIn.Dispose();

        Tensor kX = NewTensor(batch, sT, dim);
        Tensor vX = NewTensor(batch, sT, dim);
        backend.Linear(kX, text, _xaK!, _xaKB);
        backend.Linear(vX, text, _xaV!, _xaVB);

        Tensor qXMh = DiTUtils.ReshapeToMultiHead(qX, batch, sV, _numHeads, _headDim);
        Tensor kXMh = DiTUtils.ReshapeToMultiHead(kX, batch, sT, _numHeads, _headDim);
        Tensor vXMh = DiTUtils.ReshapeToMultiHead(vX, batch, sT, _numHeads, _headDim);
        qX.Dispose(); kX.Dispose(); vX.Dispose();

        ApplyRmsNormPerHead(backend, qXMh, _xaQNorm!);
        ApplyRmsNormPerHead(backend, kXMh, _xaKNorm!);

        Tensor xaMh = NewTensor4D(batch, _numHeads, sV, _headDim);
        float xaScaleVal = 1.0f / MathF.Sqrt(_headDim);
        backend.ScaledDotProductAttention(xaMh, qXMh, kXMh, vXMh, null, xaScaleVal);
        qXMh.Dispose(); kXMh.Dispose(); vXMh.Dispose();

        Tensor xaFlat = DiTUtils.ReshapeFromMultiHead(xaMh, batch, sV, _numHeads, _headDim);
        xaMh.Dispose();

        Tensor xaOut = NewTensor(batch, sV, dim);
        backend.Linear(xaOut, xaFlat, _xaO!, _xaOB);
        xaFlat.Dispose();

        Tensor afterXa = AdaLNModulation.ApplyGatedResidual(afterSa, xaOut, xaGate, batch, sV, dim);
        xaOut.Dispose();
        afterSa.Dispose();

        // ── 3. Feed-forward ──
        Tensor ffIn = LayerNormModulate(afterXa, ffShift, ffScale, batch, sV, dim);

        Tensor ffMid = NewTensor(batch, sV, _ffDim);
        backend.Linear(ffMid, ffIn, _ffIn!, null);
        ffIn.Dispose();

        Tensor ffAct = NewTensor(batch, sV, _ffDim);
        backend.Gelu(ffAct, ffMid);
        ffMid.Dispose();

        Tensor ffOut = NewTensor(batch, sV, dim);
        backend.Linear(ffOut, ffAct, _ffOut!, null);
        ffAct.Dispose();

        Tensor result = AdaLNModulation.ApplyGatedResidual(afterXa, ffOut, ffGate, batch, sV, dim);
        ffOut.Dispose();
        afterXa.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();
        return result;
    }

    private Tensor[] ProduceModulation(IBackend backend, Tensor temb, int batch, int dim)
    {
        int outDim = 9 * dim;

        Tensor activated = new Tensor(new TensorShape(batch, _timeDim), DType.F32);
        backend.Silu(activated, temb);

        Tensor projected = new Tensor(new TensorShape(batch, outDim), DType.F32);
        backend.Linear(projected, activated, _modWeight!, _modBias);
        activated.Dispose();

        Tensor[] result = new Tensor[9];
        float* projPtr = (float*)projected.DataPointer;
        for (int p = 0; p < 9; p++)
        {
            Tensor slot = new Tensor(new TensorShape(batch, dim), DType.F32);
            float* slotPtr = (float*)slot.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                int srcOffset = b * outDim + p * dim;
                int dstOffset = b * dim;
                for (int d = 0; d < dim; d++)
                    slotPtr[dstOffset + d] = projPtr[srcOffset + d];
            }
            result[p] = slot;
        }

        projected.Dispose();
        return result;
    }

    private static Tensor LayerNormModulate(Tensor x, Tensor shift, Tensor scale, int batch, int seqLen, int dim)
    {
        Tensor normed = new Tensor(new TensorShape(batch, seqLen, dim), DType.F32);
        DiTUtils.LayerNormNoAffine(normed, x, batch, seqLen, dim);
        Tensor modulated = AdaLNModulation.ApplyModulation(normed, shift, scale, batch, seqLen, dim);
        normed.Dispose();
        return modulated;
    }

    private static Tensor NewTensor(int batch, int seqLen, int dim) =>
        new Tensor(new TensorShape(batch, seqLen, dim), DType.F32);

    private static Tensor NewTensor4D(int b, int h, int s, int d) =>
        new Tensor(new TensorShape(b, h, s, d), DType.F32);

    private void ApplyRmsNormPerHead(IBackend backend, Tensor input, Tensor weight)
    {
        Tensor output = new Tensor(input.Shape, DType.F32);
        backend.RmsNorm(output, input, weight, _qkNormEps);

        long bytes = input.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)output.DataPointer, (void*)input.DataPointer, bytes, bytes);
        output.Dispose();
    }
}

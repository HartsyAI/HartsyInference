using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Kandinsky 5 transformer encoder block (text stream) — ports <c>Kandinsky5TransformerEncoderBlock.forward</c>: self-attention only (biased Q/K/V/Out, RMSNorm QK-norm), 1D RoPE on Q/K. GPU-resident end to end (mirrors QwenImageBlock/ChromaDoubleStreamBlock), including RoPE for batch 1 via <see cref="Kandinsky5Rope.ApplyGpu"/>; <see cref="Kandinsky5Rope.Apply"/> is the batched host fallback.</summary>
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

    /// <summary>Loads weights with the diffusers naming convention: <c>{prefix}.{text_modulation, self_attention, feed_forward}.*</c>. Self-attention uses <c>to_query/to_key/to_value/out_layer</c> and RMSNorm <c>query_norm/key_norm</c>. Feed-forward uses <c>in_layer/out_layer</c> with no biases.</summary>
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

    /// <summary>Forward pass on text tokens <c>x = [B, S_text, D]</c> with timestep embedding <c>temb = [B, time_dim]</c> and 1D RoPE <c>rope</c> already precomputed for the same seq_len.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor temb, Kandinsky5Rope rope)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        int dim = _modelDim;
        float attnScale = 1.0f / MathF.Sqrt(_headDim);

        Tensor[] mod = Kandinsky5BlockOps.ProduceModulation(backend, temb, _modWeight!, _modBias, batch, _timeDim, dim, 6);
        Tensor saShift = mod[0], saScale = mod[1], saGate = mod[2];
        Tensor ffShift = mod[3], ffScale = mod[4], ffGate = mod[5];

        TensorShape flat = new TensorShape(batch, seqLen, dim);
        TensorShape heads = new TensorShape(batch, seqLen, _numHeads, _headDim);
        TensorShape mh = new TensorShape(batch, _numHeads, seqLen, _headDim);

        // ── Self-attention sub-block ──
        Tensor saIn = Kandinsky5BlockOps.NormModulate(backend, x, saShift, saScale, flat);

        Tensor q = new Tensor(heads, DType.F32);
        backend.Linear(q, saIn, _qWeight!, _qBias);
        Tensor k = new Tensor(heads, DType.F32);
        backend.Linear(k, saIn, _kWeight!, _kBias);
        Tensor v = new Tensor(heads, DType.F32);
        backend.Linear(v, saIn, _vWeight!, _vBias);
        saIn.Dispose();

        // QK-norm (per-head RMSNorm over the last dim = headDim).
        Tensor qn = new Tensor(heads, DType.F32);
        backend.RmsNorm(qn, q, _qNormWeight!, _qkNormEps);
        q.Dispose();
        Tensor kn = new Tensor(heads, DType.F32);
        backend.RmsNorm(kn, k, _kNormWeight!, _qkNormEps);
        k.Dispose();

        // RoPE BEFORE the head permute: [B(=1), S, H, D] is exactly WanRopeInterleaved's [S, heads·headDim]
        // contract and the rotation is the same interleaved pairing — one GPU-resident op instead of the
        // per-block Q/K D2H → host-trig → H2D excursion (the dominant per-step cost).
        if (batch == 1)
            rope.ApplyGpu(backend, qn, kn, _numHeads, seqLen);

        // Permute [B, S, H, D] → [B, H, S, D] for SDPA.
        Tensor qMh = new Tensor(mh, DType.F32);
        backend.Permute0213(qMh, qn, seqLen, _numHeads, _headDim);
        qn.Dispose();
        Tensor kMh = new Tensor(mh, DType.F32);
        backend.Permute0213(kMh, kn, seqLen, _numHeads, _headDim);
        kn.Dispose();
        Tensor vMh = new Tensor(mh, DType.F32);
        backend.Permute0213(vMh, v, seqLen, _numHeads, _headDim);
        v.Dispose();

        if (batch != 1)   // host fallback (batched inference only)
            rope.Apply(qMh, kMh, batch, _numHeads, seqLen);

        // allowF16: Q/K are RMS-normed above → bounded scores → F16/cuDNN-fused attention is safe (43.23 gate).
        Tensor attnMh = new Tensor(mh, DType.F32);
        backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMh, null, attnScale, allowF16: true);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        // Permute back [B, H, S, D] → [B, S, hidden].
        Tensor attnFlat = new Tensor(flat, DType.F32);
        backend.Permute0213(attnFlat, attnMh, _numHeads, seqLen, _headDim);
        attnMh.Dispose();

        Tensor attnOut = new Tensor(flat, DType.F32);
        backend.Linear(attnOut, attnFlat, _outWeight!, _outBias);
        attnFlat.Dispose();

        Tensor afterAttn = new Tensor(flat, DType.F32);
        backend.GatedResidualLastDim(afterAttn, x, attnOut, saGate);
        attnOut.Dispose();

        // ── Feed-forward sub-block ──
        Tensor ffInT = Kandinsky5BlockOps.NormModulate(backend, afterAttn, ffShift, ffScale, flat);

        Tensor ffMid = new Tensor(new TensorShape(batch, seqLen, _ffDim), DType.F32);
        backend.Linear(ffMid, ffInT, _ffIn!, null);
        ffInT.Dispose();

        Tensor ffAct = new Tensor(new TensorShape(batch, seqLen, _ffDim), DType.F32);
        backend.Gelu(ffAct, ffMid);
        ffMid.Dispose();

        Tensor ffOut = new Tensor(flat, DType.F32);
        backend.Linear(ffOut, ffAct, _ffOut!, null);
        ffAct.Dispose();

        Tensor result = new Tensor(flat, DType.F32);
        backend.GatedResidualLastDim(result, afterAttn, ffOut, ffGate);
        ffOut.Dispose();
        afterAttn.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();
        return result;
    }
}

/// <summary>Kandinsky 5 transformer decoder block (visual stream) — ports <c>Kandinsky5TransformerDecoderBlock.forward</c>: self-attention with 3D RoPE on Q/K, then cross-attention to text (no RoPE), then FFN, each gated by its own modulation triple. GPU-resident; see <see cref="Kandinsky5EncoderBlock"/> for the rationale.</summary>
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

    /// <summary>Loads weights with the diffusers convention <c>{prefix}.{visual_modulation, self_attention, cross_attention, feed_forward}.*</c>.</summary>
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

    /// <summary>Forward pass on visual tokens <c>visual = [B, S_v, D]</c> with text <c>text = [B, S_t, D]</c>, timestep <c>temb = [B, time_dim]</c>, and 3D RoPE <c>rope</c> already precomputed for the visual seq.</summary>
    public Tensor Forward(IBackend backend, Tensor visual, Tensor text, Tensor temb, Kandinsky5Rope rope)
    {
        int batch = (int)visual.Shape[0];
        int sV = (int)visual.Shape[1];
        int sT = (int)text.Shape[1];
        int dim = _modelDim;
        float attnScale = 1.0f / MathF.Sqrt(_headDim);

        Tensor[] mod = Kandinsky5BlockOps.ProduceModulation(backend, temb, _modWeight!, _modBias, batch, _timeDim, dim, 9);
        // Order: (sa_shift, sa_scale, sa_gate, xa_shift, xa_scale, xa_gate, ff_shift, ff_scale, ff_gate).
        Tensor saShift = mod[0], saScale = mod[1], saGate = mod[2];
        Tensor xaShift = mod[3], xaScale = mod[4], xaGate = mod[5];
        Tensor ffShift = mod[6], ffScale = mod[7], ffGate = mod[8];

        TensorShape vFlat = new TensorShape(batch, sV, dim);
        TensorShape vHeads = new TensorShape(batch, sV, _numHeads, _headDim);
        TensorShape vMh = new TensorShape(batch, _numHeads, sV, _headDim);
        TensorShape tHeads = new TensorShape(batch, sT, _numHeads, _headDim);
        TensorShape tMh = new TensorShape(batch, _numHeads, sT, _headDim);

        // ── 1. Self-attention with 3D RoPE on Q/K ──
        Tensor saIn = Kandinsky5BlockOps.NormModulate(backend, visual, saShift, saScale, vFlat);

        Tensor q = new Tensor(vHeads, DType.F32);
        backend.Linear(q, saIn, _saQ!, _saQB);
        Tensor k = new Tensor(vHeads, DType.F32);
        backend.Linear(k, saIn, _saK!, _saKB);
        Tensor v = new Tensor(vHeads, DType.F32);
        backend.Linear(v, saIn, _saV!, _saVB);
        saIn.Dispose();

        Tensor qn = new Tensor(vHeads, DType.F32);
        backend.RmsNorm(qn, q, _saQNorm!, _qkNormEps);
        q.Dispose();
        Tensor kn = new Tensor(vHeads, DType.F32);
        backend.RmsNorm(kn, k, _saKNorm!, _qkNormEps);
        k.Dispose();

        // GPU RoPE pre-permute — see the encoder block note.
        if (batch == 1)
            rope.ApplyGpu(backend, qn, kn, _numHeads, sV);

        Tensor qMh = new Tensor(vMh, DType.F32);
        backend.Permute0213(qMh, qn, sV, _numHeads, _headDim);
        qn.Dispose();
        Tensor kMh = new Tensor(vMh, DType.F32);
        backend.Permute0213(kMh, kn, sV, _numHeads, _headDim);
        kn.Dispose();
        Tensor vMhT = new Tensor(vMh, DType.F32);
        backend.Permute0213(vMhT, v, sV, _numHeads, _headDim);
        v.Dispose();

        if (batch != 1)   // host fallback (batched inference only)
            rope.Apply(qMh, kMh, batch, _numHeads, sV);

        // allowF16: Q/K are RMS-normed above → bounded scores → F16/cuDNN-fused attention is safe (43.23 gate).
        Tensor attnMh = new Tensor(vMh, DType.F32);
        backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMhT, null, attnScale, allowF16: true);
        qMh.Dispose(); kMh.Dispose(); vMhT.Dispose();

        Tensor attnFlat = new Tensor(vFlat, DType.F32);
        backend.Permute0213(attnFlat, attnMh, _numHeads, sV, _headDim);
        attnMh.Dispose();

        Tensor saOut = new Tensor(vFlat, DType.F32);
        backend.Linear(saOut, attnFlat, _saO!, _saOB);
        attnFlat.Dispose();

        Tensor afterSa = new Tensor(vFlat, DType.F32);
        backend.GatedResidualLastDim(afterSa, visual, saOut, saGate);
        saOut.Dispose();

        // ── 2. Cross-attention to text (no RoPE) ──
        Tensor xaIn = Kandinsky5BlockOps.NormModulate(backend, afterSa, xaShift, xaScale, vFlat);

        Tensor qX = new Tensor(vHeads, DType.F32);
        backend.Linear(qX, xaIn, _xaQ!, _xaQB);
        xaIn.Dispose();

        Tensor kX = new Tensor(tHeads, DType.F32);
        backend.Linear(kX, text, _xaK!, _xaKB);
        Tensor vX = new Tensor(tHeads, DType.F32);
        backend.Linear(vX, text, _xaV!, _xaVB);

        Tensor qXn = new Tensor(vHeads, DType.F32);
        backend.RmsNorm(qXn, qX, _xaQNorm!, _qkNormEps);
        qX.Dispose();
        Tensor kXn = new Tensor(tHeads, DType.F32);
        backend.RmsNorm(kXn, kX, _xaKNorm!, _qkNormEps);
        kX.Dispose();

        Tensor qXMh = new Tensor(vMh, DType.F32);
        backend.Permute0213(qXMh, qXn, sV, _numHeads, _headDim);
        qXn.Dispose();
        Tensor kXMh = new Tensor(tMh, DType.F32);
        backend.Permute0213(kXMh, kXn, sT, _numHeads, _headDim);
        kXn.Dispose();
        Tensor vXMh = new Tensor(tMh, DType.F32);
        backend.Permute0213(vXMh, vX, sT, _numHeads, _headDim);
        vX.Dispose();

        // allowF16: Q/K are RMS-normed above → bounded scores → F16/cuDNN-fused attention is safe (43.23 gate).
        Tensor xaMh = new Tensor(vMh, DType.F32);
        backend.ScaledDotProductAttention(xaMh, qXMh, kXMh, vXMh, null, attnScale, allowF16: true);
        qXMh.Dispose(); kXMh.Dispose(); vXMh.Dispose();

        Tensor xaFlat = new Tensor(vFlat, DType.F32);
        backend.Permute0213(xaFlat, xaMh, _numHeads, sV, _headDim);
        xaMh.Dispose();

        Tensor xaOut = new Tensor(vFlat, DType.F32);
        backend.Linear(xaOut, xaFlat, _xaO!, _xaOB);
        xaFlat.Dispose();

        Tensor afterXa = new Tensor(vFlat, DType.F32);
        backend.GatedResidualLastDim(afterXa, afterSa, xaOut, xaGate);
        xaOut.Dispose();
        afterSa.Dispose();

        // ── 3. Feed-forward ──
        Tensor ffInT = Kandinsky5BlockOps.NormModulate(backend, afterXa, ffShift, ffScale, vFlat);

        Tensor ffMid = new Tensor(new TensorShape(batch, sV, _ffDim), DType.F32);
        backend.Linear(ffMid, ffInT, _ffIn!, null);
        ffInT.Dispose();

        Tensor ffAct = new Tensor(new TensorShape(batch, sV, _ffDim), DType.F32);
        backend.Gelu(ffAct, ffMid);
        ffMid.Dispose();

        Tensor ffOut = new Tensor(vFlat, DType.F32);
        backend.Linear(ffOut, ffAct, _ffOut!, null);
        ffAct.Dispose();

        Tensor result = new Tensor(vFlat, DType.F32);
        backend.GatedResidualLastDim(result, afterXa, ffOut, ffGate);
        ffOut.Dispose();
        afterXa.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();
        return result;
    }
}

/// <summary>Shared GPU-resident modulation helpers for the Kandinsky 5 encoder/decoder blocks. Keeps the AdaLN math on the device (no host DataPointer reads): the modulation projection is split with <c>SliceLastDim</c> and modulation is <c>LayerNormNoAffine → x*(1+scale)+shift</c> via <c>AddScalar</c> + <c>AffineBroadcastLastDim</c>, bit-identical to the prior CPU <c>DiTUtils</c>/<c>AdaLNModulation</c> path.</summary>
internal static class Kandinsky5BlockOps
{
    /// <summary>Computes <c>Linear(SiLU(temb)) → count * model_dim</c> then slices into <paramref name="count"/> <c>[B, dim]</c> tensors (chunk order matches diffusers' nested <c>torch.chunk</c>), all on the GPU.</summary>
    public static Tensor[] ProduceModulation(IBackend backend, Tensor temb, Tensor modWeight, Tensor? modBias,
        int batch, int timeDim, int dim, int count)
    {
        int outDim = count * dim;

        Tensor activated = new Tensor(new TensorShape(batch, timeDim), DType.F32);
        backend.Silu(activated, temb);

        Tensor projected = new Tensor(new TensorShape(batch, outDim), DType.F32);
        backend.Linear(projected, activated, modWeight, modBias);
        activated.Dispose();

        Tensor[] result = new Tensor[count];
        for (int p = 0; p < count; p++)
        {
            Tensor slot = new Tensor(new TensorShape(batch, dim), DType.F32);
            backend.SliceLastDim(slot, projected, p * dim);
            result[p] = slot;
        }
        projected.Dispose();
        return result;
    }

    /// <summary>Non-affine LayerNorm (eps 1e-6) followed by AdaLN modulation <c>out = x*(1+scale)+shift</c>, all on the GPU. <c>AffineBroadcastLastDim</c> computes <c>x*scale+shift</c>, so the scale tensor is pre-incremented by 1 via <c>AddScalar</c> to reproduce the <c>(1+scale)</c> factor.</summary>
    public static Tensor NormModulate(IBackend backend, Tensor x, Tensor shift, Tensor scale, TensorShape shape)
    {
        Tensor normed = new Tensor(shape, DType.F32);
        backend.LayerNormNoAffine(normed, x, 1e-6f);
        Tensor scalePlus1 = new Tensor(scale.Shape, DType.F32);
        backend.AddScalar(scalePlus1, scale, 1.0f);
        Tensor output = new Tensor(shape, DType.F32);
        backend.AffineBroadcastLastDim(output, normed, scalePlus1, shift);
        normed.Dispose();
        scalePlus1.Dispose();
        return output;
    }
}

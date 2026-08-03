using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>One ViT3D decoder block: affine RMSNorm → fused-QKV attention with non-affine per-head qk RMSNorm and
/// partial split-half 3-axis rope → per-channel-gated residual (<c>scale1</c>), then affine RMSNorm → gated-SiLU FFN
/// → per-channel-gated residual (<c>scale2</c>). <c>to_qkv</c> emits per-head <c>[q|k|v]</c> triples, so the row view
/// <c>[seq·heads, 3·headDim]</c> makes the split three contiguous last-dim slices.</summary>
internal sealed class MiniMaxH3VideoVaeVitBlock
{
    private readonly MiniMaxH3VideoVaeConfig _config;
    private Tensor? _norm1W, _norm2W, _scale1, _scale2;
    private Tensor? _qkvW, _qkvB, _outW, _outB;
    private Tensor? _w1W, _w1B, _w2W, _w2B;

    public MiniMaxH3VideoVaeVitBlock(MiniMaxH3VideoVaeConfig config) => _config = config;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, List<Tensor> owned)
    {
        _norm1W = MiniMaxH3VideoVaeWeights.F32(w, $"{prefix}.norm1.weight", owned);
        _norm2W = MiniMaxH3VideoVaeWeights.F32(w, $"{prefix}.norm2.weight", owned);
        _scale1 = MiniMaxH3VideoVaeWeights.F32(w, $"{prefix}.scale1", owned);
        _scale2 = MiniMaxH3VideoVaeWeights.F32(w, $"{prefix}.scale2", owned);
        _qkvW = MiniMaxH3VideoVaeWeights.Raw(w, $"{prefix}.attn.to_qkv.weight");
        _qkvB = MiniMaxH3VideoVaeWeights.OptionalF32(w, $"{prefix}.attn.to_qkv.bias", owned);
        _outW = MiniMaxH3VideoVaeWeights.Raw(w, $"{prefix}.attn.to_out.weight");
        _outB = MiniMaxH3VideoVaeWeights.OptionalF32(w, $"{prefix}.attn.to_out.bias", owned);
        _w1W = MiniMaxH3VideoVaeWeights.Raw(w, $"{prefix}.ff.w1.weight");
        _w1B = MiniMaxH3VideoVaeWeights.OptionalF32(w, $"{prefix}.ff.w1.bias", owned);
        _w2W = MiniMaxH3VideoVaeWeights.Raw(w, $"{prefix}.ff.w2.weight");
        _w2B = MiniMaxH3VideoVaeWeights.OptionalF32(w, $"{prefix}.ff.w2.bias", owned);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _norm1W, _norm2W, _scale1, _scale2, _qkvW, _qkvB, _outW, _outB, _w1W, _w1B, _w2W, _w2B })
        {
            if (t is not null) yield return t;
        }
    }

    /// <summary>Updates <paramref name="h"/> <c>[seq, dim]</c> in place. <paramref name="qkNormOnes"/> is the unit
    /// weight standing in for the reference's non-affine per-head RMSNorm.</summary>
    public void Forward(IBackend backend, Tensor h, Tensor cos, Tensor sin, Tensor qkNormOnes)
    {
        int dim = _config.Dim;
        int seq = (int)(h.ElementCount / dim);

        using (Tensor normed = new Tensor(new TensorShape(seq, dim), DType.F32))
        {
            backend.RmsNorm(normed, h, _norm1W!, _config.Eps);
            using Tensor attn = Attention(backend, normed, cos, sin, qkNormOnes, seq);
            backend.GatedResidualLastDim(h, h, attn, _scale1!);
        }

        using Tensor normed2 = new Tensor(new TensorShape(seq, dim), DType.F32);
        backend.RmsNorm(normed2, h, _norm2W!, _config.Eps);
        using Tensor ff = FeedForward(backend, normed2, seq);
        backend.GatedResidualLastDim(h, h, ff, _scale2!);
    }

    private Tensor Attention(IBackend backend, Tensor x, Tensor cos, Tensor sin, Tensor qkNormOnes, int seq)
    {
        int heads = _config.Heads, hd = _config.DimHead, dim = _config.Dim;
        TensorShape headShape = new TensorShape(1, seq, heads, hd);

        using Tensor qkv = new Tensor(new TensorShape((long)seq * heads, 3L * hd), DType.F32);
        backend.Linear(qkv, x, _qkvW!, _qkvB);

        using Tensor q = new Tensor(headShape, DType.F32);
        using Tensor k = new Tensor(headShape, DType.F32);
        using Tensor v = new Tensor(headShape, DType.F32);
        using Tensor qn = new Tensor(headShape, DType.F32);
        using Tensor kn = new Tensor(headShape, DType.F32);
        backend.SliceLastDim(q, qkv, 0);
        backend.SliceLastDim(k, qkv, hd);
        backend.SliceLastDim(v, qkv, 2 * hd);
        backend.RmsNorm(qn, q, qkNormOnes, _config.Eps);
        backend.RmsNorm(kn, k, qkNormOnes, _config.Eps);
        backend.ApplyRopeSingle(qn, cos, sin, _config.RopeDim);
        backend.ApplyRopeSingle(kn, cos, sin, _config.RopeDim);

        TensorShape bhsd = new TensorShape(1, heads, seq, hd);
        using Tensor qh = new Tensor(bhsd, DType.F32);
        using Tensor kh = new Tensor(bhsd, DType.F32);
        using Tensor vh = new Tensor(bhsd, DType.F32);
        using Tensor attn = new Tensor(bhsd, DType.F32);
        backend.Permute0213(qh, qn, seq, heads, hd);
        backend.Permute0213(kh, kn, seq, heads, hd);
        backend.Permute0213(vh, v, seq, heads, hd);
        backend.ScaledDotProductAttention(attn, qh, kh, vh, null, 1f / MathF.Sqrt(hd));

        using Tensor merged = new Tensor(new TensorShape(seq, dim), DType.F32);
        backend.Permute0213(merged, attn, heads, seq, hd);
        Tensor outT = new Tensor(new TensorShape(seq, dim), DType.F32);
        backend.Linear(outT, merged, _outW!, _outB);
        return outT;
    }

    private Tensor FeedForward(IBackend backend, Tensor x, int seq)
    {
        int inner = _config.FfnInner;
        using Tensor gateUp = new Tensor(new TensorShape(seq, 2L * inner), DType.F32);
        backend.Linear(gateUp, x, _w1W!, _w1B);
        using Tensor act = new Tensor(new TensorShape(seq, inner), DType.F32);
        backend.GluActivate(act, gateUp, inner, gelu: false);
        Tensor outT = new Tensor(new TensorShape(seq, _config.Dim), DType.F32);
        backend.Linear(outT, act, _w2W!, _w2B);
        return outT;
    }
}

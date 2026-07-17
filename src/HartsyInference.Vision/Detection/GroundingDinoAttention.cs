using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Reusable multi-head attention block for Grounding DINO, parameterized by separate query and key/value
/// widths so the same code serves same-dim self-attention, image↔text cross-attention (image 256 ↔ text 768), and
/// the decoder's query→memory cross-attentions. Projects query/key/value into a common <c>embedDim</c>, runs
/// <see cref="IBackend.ScaledDotProductAttention"/>, then projects the context back to the query width. Returns the
/// raw attention output; the caller applies the residual + norm.</summary>
public sealed class GroundingDinoAttention
{
    private readonly int _queryDim;
    private readonly int _kvDim;
    private readonly int _embedDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly float _scale;

    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB;

    /// <summary>Creates the block. <paramref name="embedDim"/> must be divisible by <paramref name="numHeads"/>.</summary>
    public GroundingDinoAttention(int queryDim, int kvDim, int embedDim, int numHeads)
    {
        if (embedDim % numHeads != 0)
            throw new ArgumentException($"embedDim {embedDim} must be divisible by numHeads {numHeads}.", nameof(embedDim));
        _queryDim = queryDim;
        _kvDim = kvDim;
        _embedDim = embedDim;
        _numHeads = numHeads;
        _headDim = embedDim / numHeads;
        _scale = 1f / MathF.Sqrt(_headDim);
    }

    public void Build(IGroundingWeightBag bag, string prefix)
    {
        _qW = bag.Get($"{prefix}.q_proj.weight", _embedDim, _queryDim);
        _qB = bag.Get($"{prefix}.q_proj.bias", _embedDim);
        _kW = bag.Get($"{prefix}.k_proj.weight", _embedDim, _kvDim);
        _kB = bag.Get($"{prefix}.k_proj.bias", _embedDim);
        _vW = bag.Get($"{prefix}.v_proj.weight", _embedDim, _kvDim);
        _vB = bag.Get($"{prefix}.v_proj.bias", _embedDim);
        _oW = bag.Get($"{prefix}.out_proj.weight", _queryDim, _embedDim);
        _oB = bag.Get($"{prefix}.out_proj.bias", _queryDim);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB];
        foreach (Tensor? t in all)
            if (t is not null) yield return t;
    }

    /// <summary>Attends <paramref name="query"/> <c>[1, Sq, queryDim]</c> over <paramref name="keyValue"/>
    /// <c>[1, Skv, kvDim]</c>; optional <paramref name="additiveMask"/> is <c>[Sq, Skv]</c> (0 keep / −inf drop,
    /// shared across heads). Returns <c>[1, Sq, queryDim]</c>; caller owns it.</summary>
    public Tensor Forward(IBackend backend, Tensor query, Tensor keyValue, Tensor? additiveMask)
    {
        if (_qW is null)
            throw new InvalidOperationException("GroundingDinoAttention weights not loaded.");
        int sq = (int)query.Shape[1];
        int skv = (int)keyValue.Shape[1];

        Tensor q = new Tensor(new TensorShape(1, sq, _embedDim), DType.F32);
        Tensor k = new Tensor(new TensorShape(1, skv, _embedDim), DType.F32);
        Tensor v = new Tensor(new TensorShape(1, skv, _embedDim), DType.F32);
        backend.Linear(q, query, _qW!, _qB);
        backend.Linear(k, keyValue, _kW!, _kB);
        backend.Linear(v, keyValue, _vW!, _vB);

        Tensor qH = ToHeads(backend, q, sq);
        Tensor kH = ToHeads(backend, k, skv);
        Tensor vH = ToHeads(backend, v, skv);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor attn = new Tensor(new TensorShape(1, _numHeads, sq, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attn, qH, kH, vH, additiveMask, _scale);
        qH.Dispose(); kH.Dispose(); vH.Dispose();

        Tensor merged = FromHeads(backend, attn, sq);
        attn.Dispose();

        Tensor output = new Tensor(new TensorShape(1, sq, _queryDim), DType.F32);
        backend.Linear(output, merged, _oW!, _oB);
        merged.Dispose();
        return output;
    }

    /// <summary>[1, S, embedDim] → [1, heads, S, headDim] (view as [1, S, heads, headDim] then swap S/heads).</summary>
    private Tensor ToHeads(IBackend backend, Tensor x, int s)
    {
        Tensor view = x.Reshape(new TensorShape(1, s, _numHeads, _headDim));
        Tensor heads = new Tensor(new TensorShape(1, _numHeads, s, _headDim), DType.F32);
        backend.Permute0213(heads, view, s, _numHeads, _headDim);
        view.Dispose();
        return heads;
    }

    /// <summary>[1, heads, S, headDim] → [1, S, embedDim] (swap heads/S back, then flatten). The permute writes
    /// through a reshaped view of the owning <c>merged</c> buffer so the caller gets a disposable owner, not a
    /// view whose parent would leak.</summary>
    private Tensor FromHeads(IBackend backend, Tensor heads, int s)
    {
        Tensor merged = new Tensor(new TensorShape(1, s, _embedDim), DType.F32);
        Tensor mergedView = merged.Reshape(new TensorShape(1, s, _numHeads, _headDim));
        backend.Permute0213(mergedView, heads, _numHeads, s, _headDim);
        mergedView.Dispose();
        return merged;
    }
}

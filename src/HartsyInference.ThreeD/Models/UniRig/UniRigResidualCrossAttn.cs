using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Post-norm cross-attention block of the skin head: <c>x = norm1(q + attn(q, kv))</c>, <c>x = norm2(x + ffn(x))</c> with a 4× GELU feed-forward.</summary>
internal sealed class UniRigResidualCrossAttn : IDisposable
{
    private readonly IBackend _backend;
    private readonly AttentionLayer _attention;
    private readonly Tensor _keyWeight;
    private readonly Tensor _valueWeight;
    private readonly Tensor _keyBias;
    private readonly Tensor _valueBias;
    private readonly Tensor _norm1Weight;
    private readonly Tensor _norm1Bias;
    private readonly Tensor _norm2Weight;
    private readonly Tensor _norm2Bias;
    private readonly Tensor _ffn1Weight;
    private readonly Tensor _ffn1Bias;
    private readonly Tensor _ffn2Weight;
    private readonly Tensor _ffn2Bias;

    public UniRigResidualCrossAttn(IBackend backend, IReadOnlyDictionary<string, Tensor> w, string prefix, int dim, int heads)
    {
        _backend = backend;
        long d = dim;
        Tensor kv = UniRigCheckpoint.Get(w, $"{prefix}.attention.Wkv.weight", 2 * d, d);
        Tensor kvBias = UniRigCheckpoint.Get(w, $"{prefix}.attention.Wkv.bias", 2 * d);
        _keyWeight = RowSlice(kv, 0, dim);
        _valueWeight = RowSlice(kv, dim, dim);
        _keyBias = RowSlice(kvBias, 0, dim);
        _valueBias = RowSlice(kvBias, dim, dim);
        _attention = new AttentionLayer(
            new AttentionLayerConfig { Heads = heads, KvHeads = heads, HeadDim = dim / heads, ComputeDType = DType.F32 },
            new AttentionLayerWeights
            {
                Query = UniRigCheckpoint.Get(w, $"{prefix}.attention.Wq.weight", d, d),
                QueryBias = UniRigCheckpoint.Get(w, $"{prefix}.attention.Wq.bias", d),
                Key = _keyWeight,
                KeyBias = _keyBias,
                Value = _valueWeight,
                ValueBias = _valueBias,
                Output = UniRigCheckpoint.Get(w, $"{prefix}.attention.out_proj.weight", d, d),
                OutputBias = UniRigCheckpoint.Get(w, $"{prefix}.attention.out_proj.bias", d),
            });
        _norm1Weight = UniRigCheckpoint.Get(w, $"{prefix}.norm1.weight", d);
        _norm1Bias = UniRigCheckpoint.Get(w, $"{prefix}.norm1.bias", d);
        _norm2Weight = UniRigCheckpoint.Get(w, $"{prefix}.norm2.weight", d);
        _norm2Bias = UniRigCheckpoint.Get(w, $"{prefix}.norm2.bias", d);
        _ffn1Weight = UniRigCheckpoint.Get(w, $"{prefix}.ffn.0.weight", 4 * d, d);
        _ffn1Bias = UniRigCheckpoint.Get(w, $"{prefix}.ffn.0.bias", 4 * d);
        _ffn2Weight = UniRigCheckpoint.Get(w, $"{prefix}.ffn.2.weight", d, 4 * d);
        _ffn2Bias = UniRigCheckpoint.Get(w, $"{prefix}.ffn.2.bias", d);
    }

    /// <summary>Attends <paramref name="q"/> <c>[1, Sq, D]</c> over <paramref name="kv"/> <c>[1, Skv, D]</c>; neither input is released.</summary>
    public Tensor Forward(Tensor q, Tensor kv)
    {
        using Tensor attended = _attention.Forward(_backend, q, kv, default);
        using Tensor sum = UniRigOps.Add(_backend, q, attended);
        using Tensor x = UniRigOps.LayerNorm(_backend, sum, _norm1Weight, _norm1Bias);
        using Tensor hidden = UniRigOps.Linear(_backend, x, _ffn1Weight, _ffn1Bias);
        using Tensor activated = UniRigOps.GeluErf(_backend, hidden);
        using Tensor ffn = UniRigOps.Linear(_backend, activated, _ffn2Weight, _ffn2Bias);
        using Tensor sum2 = UniRigOps.Add(_backend, x, ffn);
        return UniRigOps.LayerNorm(_backend, sum2, _norm2Weight, _norm2Bias);
    }

    public void Dispose()
    {
        _keyWeight.Dispose();
        _valueWeight.Dispose();
        _keyBias.Dispose();
        _valueBias.Dispose();
    }

    private static Tensor RowSlice(Tensor source, int start, int count)
    {
        long width = source.Shape.Rank == 1 ? 1 : source.Shape[1];
        Tensor slice = new(source.Shape.Rank == 1 ? new TensorShape(count) : new TensorShape(count, width), DType.F32);
        source.AsReadOnlySpan<float>().Slice((int)(start * width), (int)(count * width)).CopyTo(slice.AsSpan<float>());
        return slice;
    }
}

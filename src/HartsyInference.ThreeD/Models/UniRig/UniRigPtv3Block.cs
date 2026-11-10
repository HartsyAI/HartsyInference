using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Utonia;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>
/// One block of UniRig's point transformer: a submanifold-conv positional encoding (conv, linear, LayerNorm) added to the features,
/// then pre-norm patch attention and a pre-norm GELU MLP, each as a residual branch.
/// </summary>
internal sealed class UniRigPtv3Block : IDisposable
{
    private readonly IBackend _backend;
    private readonly Tensor[] _convSlices;
    private readonly Tensor _convBias;
    private readonly Tensor _cpeWeight;
    private readonly Tensor _cpeBias;
    private readonly Tensor _cpeNormWeight;
    private readonly Tensor _cpeNormBias;
    private readonly Tensor _norm1Weight;
    private readonly Tensor _norm1Bias;
    private readonly Tensor _norm2Weight;
    private readonly Tensor _norm2Bias;
    private readonly Tensor _fc1Weight;
    private readonly Tensor _fc1Bias;
    private readonly Tensor _fc2Weight;
    private readonly Tensor _fc2Bias;
    private readonly UniRigPtv3Attention _attention;
    private bool _disposed;

    public UniRigPtv3Block(IBackend backend, IReadOnlyDictionary<string, Tensor> w, string prefix, int channels, int heads, int patchSize, int orderIndex)
    {
        _backend = backend;
        long c = channels;
        Tensor conv = UniRigCheckpoint.Get(w, $"{prefix}.cpe.0.weight", c, 3, 3, 3, c);
        _convSlices = SparseOps.ConvWeightSlices(conv);
        _convBias = UniRigCheckpoint.Get(w, $"{prefix}.cpe.0.bias", c);
        _cpeWeight = UniRigCheckpoint.Get(w, $"{prefix}.cpe.1.weight", c, c);
        _cpeBias = UniRigCheckpoint.Get(w, $"{prefix}.cpe.1.bias", c);
        _cpeNormWeight = UniRigCheckpoint.Get(w, $"{prefix}.cpe.2.weight", c);
        _cpeNormBias = UniRigCheckpoint.Get(w, $"{prefix}.cpe.2.bias", c);
        _norm1Weight = UniRigCheckpoint.Get(w, $"{prefix}.norm1.0.weight", c);
        _norm1Bias = UniRigCheckpoint.Get(w, $"{prefix}.norm1.0.bias", c);
        _norm2Weight = UniRigCheckpoint.Get(w, $"{prefix}.norm2.0.weight", c);
        _norm2Bias = UniRigCheckpoint.Get(w, $"{prefix}.norm2.0.bias", c);
        _fc1Weight = UniRigCheckpoint.Get(w, $"{prefix}.mlp.0.fc1.weight", 4 * c, c);
        _fc1Bias = UniRigCheckpoint.Get(w, $"{prefix}.mlp.0.fc1.bias", 4 * c);
        _fc2Weight = UniRigCheckpoint.Get(w, $"{prefix}.mlp.0.fc2.weight", c, 4 * c);
        _fc2Bias = UniRigCheckpoint.Get(w, $"{prefix}.mlp.0.fc2.bias", c);
        _attention = new UniRigPtv3Attention(
            backend,
            UniRigCheckpoint.Get(w, $"{prefix}.attn.qkv.weight", 3 * c, c),
            UniRigCheckpoint.Get(w, $"{prefix}.attn.qkv.bias", 3 * c),
            UniRigCheckpoint.Get(w, $"{prefix}.attn.proj.weight", c, c),
            UniRigCheckpoint.Get(w, $"{prefix}.attn.proj.bias", c),
            channels, heads, patchSize, orderIndex);
    }

    /// <summary>Runs the block on <paramref name="x"/> <c>[N, C]</c>; the input is released and a new tensor returned. The point must be serialized and sparsified.</summary>
    public Tensor Forward(UtoniaPoint point, Tensor x)
    {
        Tensor x1;
        using (Tensor cpe = Positional(point, x))
        {
            x1 = UniRigOps.Add(_backend, x, cpe);
        }

        x.Dispose();
        using Tensor n1 = UniRigOps.LayerNorm(_backend, x1, _norm1Weight, _norm1Bias);
        using Tensor attended = _attention.Forward(n1, point);
        Tensor x2 = UniRigOps.Add(_backend, x1, attended);
        x1.Dispose();
        using Tensor n2 = UniRigOps.LayerNorm(_backend, x2, _norm2Weight, _norm2Bias);
        using Tensor hidden = UniRigOps.Linear(_backend, n2, _fc1Weight, _fc1Bias);
        using Tensor activated = UniRigOps.GeluErf(_backend, hidden);
        using Tensor mlp = UniRigOps.Linear(_backend, activated, _fc2Weight, _fc2Bias);
        Tensor x3 = UniRigOps.Add(_backend, x2, mlp);
        x2.Dispose();
        return x3;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (Tensor slice in _convSlices)
        {
            slice.Dispose();
        }
    }

    private Tensor Positional(UtoniaPoint point, Tensor x)
    {
        SparseTensor sparse = point.AsSparse(x);
        SparseTensor conv = SparseOps.SubmanifoldConv3dSparse(_backend, sparse, _convSlices, _convBias);
        using Tensor flat = new(x.Shape, DType.F32);
        conv.Feats.AsReadOnlySpan<float>().CopyTo(flat.AsSpan<float>());
        conv.Feats.Dispose();
        using Tensor linear = UniRigOps.Linear(_backend, flat, _cpeWeight, _cpeBias);
        return UniRigOps.LayerNorm(_backend, linear, _cpeNormWeight, _cpeNormBias);
    }
}

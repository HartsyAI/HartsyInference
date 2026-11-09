using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>
/// One Point Transformer block: submanifold-conv positional encoding, patch attention and an MLP, each with a
/// residual. Inference only, so drop-path is the identity.
/// </summary>
internal sealed class UtoniaBlock : IDisposable
{
    private readonly IBackend _backend;
    private readonly Tensor[] _convSlices;
    private readonly Tensor _convBias;
    private readonly UtoniaLinear _cpeLinear;
    private readonly UtoniaLayerNorm _cpeNorm;
    private readonly UtoniaLayerNorm _norm1;
    private readonly UtoniaLayerNorm _norm2;
    private readonly Tensor? _gamma1;
    private readonly Tensor? _gamma2;
    private readonly UtoniaSerializedAttention _attention;
    private readonly UtoniaLinear _fc1;
    private readonly UtoniaLinear _fc2;
    private readonly bool _preNorm;
    private bool _disposed;

    public UtoniaBlock(IBackend backend, UtoniaWeights weights, string prefix, UtoniaConfig config, int channels, int heads,
        int patchSize, int orderIndex)
    {
        _backend = backend;
        _preNorm = config.PreNorm;
        long c = channels;
        long hidden = (long)(channels * config.MlpRatio);

        Tensor convWeight = weights.Get(UtoniaCheckpointKeys.ParameterName(prefix, UtoniaCheckpointKeys.CpeConv, UtoniaCheckpointKeys.Weight), c, 3, 3, 3, c);
        _convSlices = SparseOps.ConvWeightSlices(convWeight);
        _convBias = weights.Get(UtoniaCheckpointKeys.ParameterName(prefix, UtoniaCheckpointKeys.CpeConv, UtoniaCheckpointKeys.Bias), c);
        _cpeLinear = Linear(weights, prefix, UtoniaCheckpointKeys.CpeLinear, c, c, true);
        _cpeNorm = Norm(weights, prefix, UtoniaCheckpointKeys.CpeNorm, c);
        _norm1 = Norm(weights, prefix, UtoniaCheckpointKeys.Norm1, c);
        _norm2 = Norm(weights, prefix, UtoniaCheckpointKeys.Norm2, c);
        if (config.LayerScaleInit is not null)
        {
            _gamma1 = weights.Get(UtoniaCheckpointKeys.ParameterName(prefix, UtoniaCheckpointKeys.LayerScale1, UtoniaCheckpointKeys.Gamma), c);
            _gamma2 = weights.Get(UtoniaCheckpointKeys.ParameterName(prefix, UtoniaCheckpointKeys.LayerScale2, UtoniaCheckpointKeys.Gamma), c);
        }

        UtoniaLinear qkv = Linear(weights, prefix, UtoniaCheckpointKeys.AttnQkv, 3 * c, c, config.QkvBias);
        UtoniaLinear proj = Linear(weights, prefix, UtoniaCheckpointKeys.AttnProj, c, c, true);
        _attention = new UtoniaSerializedAttention(backend, qkv, proj, channels, heads, patchSize, config.RopeBase, orderIndex);
        _fc1 = Linear(weights, prefix, UtoniaCheckpointKeys.MlpFc1, hidden, c, true);
        _fc2 = Linear(weights, prefix, UtoniaCheckpointKeys.MlpFc2, c, hidden, true);
    }

    /// <summary>Runs the block on <paramref name="point"/>'s features and replaces them. Requires a serialized, sparsified point.</summary>
    public void Forward(UtoniaPoint point)
    {
        Tensor x0 = point.Feat;
        Tensor cpe = CpeBranch(point, x0);
        Tensor x1 = UtoniaTensors.Add(_backend, x0, cpe);
        cpe.Dispose();

        Tensor attnIn = _preNorm ? _norm1.Forward(x1) : x1;
        Tensor attn = _attention.Forward(attnIn, point);
        if (_preNorm)
        {
            attnIn.Dispose();
        }

        Tensor x2 = Residual(x1, attn, _gamma1);
        x1.Dispose();
        if (!_preNorm)
        {
            Tensor normed1 = _norm1.Forward(x2);
            x2.Dispose();
            x2 = normed1;
        }

        Tensor mlpIn = _preNorm ? _norm2.Forward(x2) : x2;
        Tensor hidden = _fc1.Forward(mlpIn);
        if (_preNorm)
        {
            mlpIn.Dispose();
        }

        Tensor activated = new(hidden.Shape, DType.F32);
        _backend.GeluErf(activated, hidden);
        hidden.Dispose();
        Tensor mlp = _fc2.Forward(activated);
        activated.Dispose();
        Tensor x3 = Residual(x2, mlp, _gamma2);
        x2.Dispose();
        if (!_preNorm)
        {
            Tensor normed2 = _norm2.Forward(x3);
            x3.Dispose();
            x3 = normed2;
        }

        point.ReplaceFeat(x3);
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

    private Tensor CpeBranch(UtoniaPoint point, Tensor feat)
    {
        SparseTensor input = point.AsSparse(feat);
        SparseTensor conv = SparseOps.SubmanifoldConv3dSparse(_backend, input, _convSlices, _convBias);
        Tensor flat = new(feat.Shape, DType.F32);
        _backend.CopyTo(flat, conv.Feats);
        conv.Feats.Dispose();
        Tensor linear = _cpeLinear.Forward(flat);
        flat.Dispose();
        Tensor normed = _cpeNorm.Forward(linear);
        linear.Dispose();
        return normed;
    }

    /// <summary>Returns <c>shortcut + gamma * branch</c> as a new tensor; <paramref name="branch"/> is released.</summary>
    private Tensor Residual(Tensor shortcut, Tensor branch, Tensor? gamma)
    {
        Tensor scaled = branch;
        if (gamma is not null)
        {
            scaled = new Tensor(branch.Shape, DType.F32);
            _backend.AffineBroadcastLastDim(scaled, branch, gamma, null);
            branch.Dispose();
        }

        Tensor sum = UtoniaTensors.Add(_backend, shortcut, scaled);
        scaled.Dispose();
        return sum;
    }

    private UtoniaLinear Linear(UtoniaWeights weights, string prefix, string module, long outDim, long inDim, bool bias)
    {
        Tensor weight = weights.Get(UtoniaCheckpointKeys.ParameterName(prefix, module, UtoniaCheckpointKeys.Weight), outDim, inDim);
        Tensor? biasTensor = bias ? weights.Get(UtoniaCheckpointKeys.ParameterName(prefix, module, UtoniaCheckpointKeys.Bias), outDim) : null;
        return new UtoniaLinear(_backend, weight, biasTensor);
    }

    private UtoniaLayerNorm Norm(UtoniaWeights weights, string prefix, string module, long dim)
    {
        Tensor weight = weights.Get(UtoniaCheckpointKeys.ParameterName(prefix, module, UtoniaCheckpointKeys.Weight), dim);
        Tensor bias = weights.Get(UtoniaCheckpointKeys.ParameterName(prefix, module, UtoniaCheckpointKeys.Bias), dim);
        return new UtoniaLayerNorm(_backend, weight, bias);
    }
}

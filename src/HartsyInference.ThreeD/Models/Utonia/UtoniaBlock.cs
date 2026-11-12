using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>
/// One Point Transformer block: submanifold-conv positional encoding, patch attention and an MLP, each with a
/// residual. Inference only, so drop-path is the identity. Every full-width operator runs in row chunks, so the
/// block's features live in host memory and only one chunk at a time occupies VRAM.
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
    private readonly int _channels;
    private readonly long _chunkBytes;
    private readonly int _rowsPerChunk;
    private bool _disposed;

    public UtoniaBlock(IBackend backend, UtoniaWeights weights, string prefix, UtoniaConfig config, int channels, int heads,
        int patchSize, int orderIndex, long chunkBytes = UtoniaRowChunks.ChunkBytes)
    {
        _backend = backend;
        _preNorm = config.PreNorm;
        _channels = channels;
        _chunkBytes = chunkBytes;
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
        _attention = new UtoniaSerializedAttention(backend, qkv, proj, channels, heads, patchSize, config.UseRope ? config.RopeBase : 0f, orderIndex, chunkBytes);
        _fc1 = Linear(weights, prefix, UtoniaCheckpointKeys.MlpFc1, hidden, c, true);
        _fc2 = Linear(weights, prefix, UtoniaCheckpointKeys.MlpFc2, c, hidden, true);
        _rowsPerChunk = UtoniaRowChunks.RowsPerChunk(chunkBytes, Math.Max(channels, hidden));
    }

    /// <summary>Runs the block on <paramref name="point"/>'s features and replaces them. Requires a serialized, sparsified point.</summary>
    public void Forward(UtoniaPoint point)
    {
        Tensor x0 = point.Feat;
        Tensor cpe = CpeBranch(point, x0);
        Tensor x1 = Sum(x0, cpe);
        cpe.Dispose();

        Tensor attnIn = _preNorm ? Normalize(_norm1, x1) : x1;
        Tensor attn = _attention.Forward(attnIn, point);
        if (_preNorm)
        {
            attnIn.Dispose();
        }

        Tensor x2 = Residual(x1, attn, _gamma1);
        x1.Dispose();
        if (!_preNorm)
        {
            Tensor normed1 = Normalize(_norm1, x2);
            x2.Dispose();
            x2 = normed1;
        }

        Tensor mlpIn = _preNorm ? Normalize(_norm2, x2) : x2;
        Tensor mlp = Mlp(mlpIn);
        if (_preNorm)
        {
            mlpIn.Dispose();
        }

        Tensor x3 = Residual(x2, mlp, _gamma2);
        x2.Dispose();
        if (!_preNorm)
        {
            Tensor normed2 = Normalize(_norm2, x3);
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
        SparseTensor conv = SparseOps.SubmanifoldConv3dSparse(_backend, input, _convSlices, _convBias, _chunkBytes);
        Tensor flat = conv.Feats.Reshape(feat.Shape);
        Tensor normed = UtoniaRowChunks.Map(flat, _channels, _rowsPerChunk, (src, res) =>
        {
            using Tensor linear = new(new TensorShape(src.Shape[0], _channels), DType.F32);
            _cpeLinear.Forward(src, linear);
            _cpeNorm.Forward(linear, res);
        });
        flat.Dispose();
        conv.Feats.Dispose();
        return normed;
    }

    private Tensor Mlp(Tensor input)
    {
        return UtoniaRowChunks.Map(input, _channels, _rowsPerChunk, (src, res) =>
        {
            using Tensor hidden = new(new TensorShape(src.Shape[0], _fc1.OutChannels), DType.F32);
            _fc1.Forward(src, hidden);
            using Tensor activated = new(hidden.Shape, DType.F32);
            _backend.GeluErf(activated, hidden);
            _fc2.Forward(activated, res);
        });
    }

    private Tensor Normalize(UtoniaLayerNorm norm, Tensor input) =>
        UtoniaRowChunks.Map(input, _channels, _rowsPerChunk, (src, res) => norm.Forward(src, res));

    private Tensor Sum(Tensor a, Tensor b) =>
        UtoniaRowChunks.Map2(a, b, _channels, _rowsPerChunk, (x, y, res) => _backend.Add(res, x, y));

    /// <summary>Returns <c>shortcut + gamma * branch</c> as a new tensor; <paramref name="branch"/> is released.</summary>
    private Tensor Residual(Tensor shortcut, Tensor branch, Tensor? gamma)
    {
        Tensor sum = UtoniaRowChunks.Map2(shortcut, branch, _channels, _rowsPerChunk, (s, b, res) =>
        {
            if (gamma is null)
            {
                _backend.Add(res, s, b);
                return;
            }

            using Tensor scaled = new(b.Shape, DType.F32);
            _backend.AffineBroadcastLastDim(scaled, b, gamma, null);
            _backend.Add(res, s, scaled);
        });
        branch.Dispose();
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

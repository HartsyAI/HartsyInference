using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Utonia;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>
/// The point transformer of the skin stage: a k=5 sparse-conv stem with BatchNorm and a linear residual, then five stages of
/// blocks at one resolution (stages are joined by linear layers, nothing is pooled). Inference only. Reuses Utonia's curve codes,
/// patch plan and point structure; the attention, stem and stage layout are UniRig's own.
/// </summary>
public sealed class UniRigPtv3Encoder : IDisposable
{
    private const int StemKernel = 5;

    private readonly IBackend _backend;
    private readonly UniRigPtv3Config _config;
    private readonly UtoniaSerialOrder[] _orders;
    private readonly Tensor[] _stemSlices;
    private readonly Tensor _stemScale;
    private readonly Tensor _stemShift;
    private readonly Tensor _resWeight;
    private readonly Tensor _resBias;
    private readonly Tensor?[] _stageWeight;
    private readonly Tensor?[] _stageBias;
    private readonly UniRigPtv3Block[][] _blocks;
    private bool _disposed;

    /// <param name="weights">Checkpoint tensors by module path; they must outlive the encoder.</param>
    /// <param name="prefix">Path of the encoder module, e.g. <c>mesh_encoder</c>.</param>
    public UniRigPtv3Encoder(IBackend backend, UniRigPtv3Config config, IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        config.Validate();
        _backend = backend;
        _config = config;
        _orders = config.Order.Select(UtoniaSerialization.ParseOrder).ToArray();
        long c0 = config.Channels[0];
        long cin = config.InChannels;

        _stemSlices = SparseOps.ConvWeightSlices(UniRigCheckpoint.Get(weights, $"{prefix}.embedding.stem.conv.weight", c0, StemKernel,
            StemKernel, StemKernel, cin));
        string norm = $"{prefix}.embedding.stem.norm";
        (_stemScale, _stemShift) = FoldBatchNorm(
            UniRigCheckpoint.Get(weights, $"{norm}.weight", c0), UniRigCheckpoint.Get(weights, $"{norm}.bias", c0),
            UniRigCheckpoint.Get(weights, $"{norm}.running_mean", c0), UniRigCheckpoint.Get(weights, $"{norm}.running_var", c0), config.BatchNormEps);
        _resWeight = UniRigCheckpoint.Get(weights, $"{prefix}.embedding.res_linear.weight", c0, cin);
        _resBias = UniRigCheckpoint.Get(weights, $"{prefix}.embedding.res_linear.bias", c0);

        int stages = config.Depths.Length;
        _stageWeight = new Tensor?[stages];
        _stageBias = new Tensor?[stages];
        _blocks = new UniRigPtv3Block[stages][];
        for (int s = 0; s < stages; s++)
        {
            string stage = $"{prefix}.enc.enc{s}";
            if (s > 0)
            {
                _stageWeight[s] = UniRigCheckpoint.Get(weights, $"{stage}.0.weight", config.Channels[s], config.Channels[s - 1]);
                _stageBias[s] = UniRigCheckpoint.Get(weights, $"{stage}.0.bias", config.Channels[s]);
            }

            _blocks[s] = new UniRigPtv3Block[config.Depths[s]];
            for (int i = 0; i < config.Depths[s]; i++)
            {
                _blocks[s][i] = new UniRigPtv3Block(backend, weights, $"{stage}.block{i}", config.Channels[s], config.Heads[s],
                    config.PatchSize, i % _orders.Length);
            }
        }
    }

    /// <summary>Feature width of <see cref="Encode"/>'s output.</summary>
    public int OutputChannels => _config.Channels[^1];

    /// <summary>
    /// Features <c>[N, OutputChannels]</c> for a point cloud: inputs are <c>[xyz, normal, 0, 0, 0]</c> per point, grid cells are
    /// <c>trunc((xyz - min) / GridSize)</c>. The caller owns the result.
    /// </summary>
    /// <param name="points">Positions, length <c>3 * N</c>.</param>
    /// <param name="normals">Normals, length <c>3 * N</c>.</param>
    public Tensor Encode(ReadOnlySpan<float> points, ReadOnlySpan<float> normals)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int n = points.Length / 3;
        if (n == 0 || points.Length != 3 * n || normals.Length != 3 * n)
        {
            throw new ArgumentException("UniRig point transformer needs N points and N normals of three floats each.");
        }

        float[] min = [float.MaxValue, float.MaxValue, float.MaxValue];
        for (int i = 0; i < points.Length; i++)
        {
            min[i % 3] = Math.Min(min[i % 3], points[i]);
        }

        float[] features = new float[n * _config.InChannels];
        int[] grid = new int[3 * n];
        float[] coord = points.ToArray();
        for (int i = 0; i < n; i++)
        {
            for (int a = 0; a < 3; a++)
            {
                features[i * _config.InChannels + a] = points[3 * i + a];
                features[i * _config.InChannels + 3 + a] = normals[3 * i + a];
                grid[3 * i + a] = (int)((points[3 * i + a] - min[a]) / _config.GridSize);
            }
        }

        using UtoniaPoint point = new(UniRigOps.FromFloats(features, new TensorShape(n, _config.InChannels)), coord, grid, new int[n]);
        if (UtoniaSerialization.BitLength(grid.Max()) > UtoniaSerialization.MaxDepth)
        {
            throw new HartsyInferenceException("UniRig point cloud spans more grid cells than the serialization can encode.");
        }

        point.Serialize(_orders);
        point.Sparsify();
        Tensor x = Embed(point.Feat, grid);
        for (int s = 0; s < _blocks.Length; s++)
        {
            if (s > 0)
            {
                Tensor projected = UniRigOps.Linear(_backend, x, _stageWeight[s]!, _stageBias[s]);
                x.Dispose();
                x = projected;
            }

            foreach (UniRigPtv3Block block in _blocks[s])
            {
                x = block.Forward(point, x);
            }
        }

        return x;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (Tensor slice in _stemSlices)
        {
            slice.Dispose();
        }

        _stemScale.Dispose();
        _stemShift.Dispose();
        foreach (UniRigPtv3Block[] stage in _blocks)
        {
            foreach (UniRigPtv3Block block in stage)
            {
                block.Dispose();
            }
        }
    }

    private Tensor Embed(Tensor features, int[] grid)
    {
        using Tensor conv = UniRigSparseConv.Run(_backend, features, grid, _stemSlices, StemKernel);
        using Tensor normed = new(conv.Shape, DType.F32);
        _backend.AffineBroadcastLastDim(normed, conv, _stemScale, _stemShift);
        using Tensor activated = UniRigOps.GeluErf(_backend, normed);
        using Tensor residual = UniRigOps.Linear(_backend, features, _resWeight, _resBias);
        return UniRigOps.Add(_backend, activated, residual);
    }

    private static (Tensor Scale, Tensor Shift) FoldBatchNorm(Tensor weight, Tensor bias, Tensor mean, Tensor variance, float eps)
    {
        int c = (int)weight.Shape[0];
        Tensor scale = new(new TensorShape(c), DType.F32);
        Tensor shift = new(new TensorShape(c), DType.F32);
        Span<float> s = scale.AsSpan<float>();
        Span<float> t = shift.AsSpan<float>();
        for (int i = 0; i < c; i++)
        {
            double factor = weight.AsReadOnlySpan<float>()[i] / Math.Sqrt(variance.AsReadOnlySpan<float>()[i] + eps);
            s[i] = (float)factor;
            t[i] = (float)(bias.AsReadOnlySpan<float>()[i] - mean.AsReadOnlySpan<float>()[i] * factor);
        }

        return (scale, shift);
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>
/// Inference-only Utonia (Point Transformer V3) on the CPU reference path.
/// <see cref="ForwardLevel4"/> runs the encoder only and returns the last stage (CoLa3D's <c>forward_level4</c>).
/// </summary>
public sealed class UtoniaEncoder : IDisposable
{
    private readonly IBackend _backend;
    private readonly UtoniaConfig _config;
    private readonly UtoniaSerialOrder[] _orders;
    private readonly UtoniaLinear _embedLinear;
    private readonly UtoniaLayerNorm _embedNorm;
    private readonly UtoniaGridPooling[] _pools;
    private readonly UtoniaBlock[][] _encBlocks;
    private readonly UtoniaGridUnpooling[] _unpools;
    private readonly UtoniaBlock[][] _decBlocks;
    private readonly long _chunkBytes;
    private bool _disposed;

    /// <summary>Builds the modules from the weights and records every name it requests in <see cref="UtoniaWeights.Consumed"/>.</summary>
    public UtoniaEncoder(IBackend backend, UtoniaConfig config, UtoniaWeights weights)
        : this(backend, config, weights, UtoniaRowChunks.ChunkBytes)
    {
    }

    /// <param name="chunkBytes">Byte budget of one row chunk; a smaller budget only changes how the work is split.</param>
    internal UtoniaEncoder(IBackend backend, UtoniaConfig config, UtoniaWeights weights, long chunkBytes)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        config.Validate();
        _backend = backend;
        _config = config;
        _chunkBytes = chunkBytes;
        _orders = config.Order.Select(UtoniaSerialization.ParseOrder).ToArray();

        int stages = config.EncDepths.Length;
        long c0 = config.EncChannels[0];
        _embedLinear = new UtoniaLinear(backend,
            weights.Get(UtoniaCheckpointKeys.ParameterName(string.Empty, UtoniaCheckpointKeys.EmbeddingLinear, UtoniaCheckpointKeys.Weight), c0, config.InChannels),
            weights.Get(UtoniaCheckpointKeys.ParameterName(string.Empty, UtoniaCheckpointKeys.EmbeddingLinear, UtoniaCheckpointKeys.Bias), c0));
        _embedNorm = new UtoniaLayerNorm(backend,
            weights.Get(UtoniaCheckpointKeys.ParameterName(string.Empty, UtoniaCheckpointKeys.EmbeddingNorm, UtoniaCheckpointKeys.Weight), c0),
            weights.Get(UtoniaCheckpointKeys.ParameterName(string.Empty, UtoniaCheckpointKeys.EmbeddingNorm, UtoniaCheckpointKeys.Bias), c0));

        _pools = new UtoniaGridPooling[stages - 1];
        _encBlocks = new UtoniaBlock[stages][];
        for (int s = 0; s < stages; s++)
        {
            int c = config.EncChannels[s];
            if (s > 0)
            {
                string prefix = UtoniaCheckpointKeys.StagePrefix(false, s);
                _pools[s - 1] = new UtoniaGridPooling(backend,
                    Linear(weights, prefix, UtoniaCheckpointKeys.DownProj, c, config.EncChannels[s - 1], true),
                    Norm(weights, prefix, UtoniaCheckpointKeys.DownNorm, c),
                    config.Stride[s - 1], chunkBytes);
            }

            _encBlocks[s] = BuildStage(backend, weights, config, false, s, config.EncDepths[s], c, config.EncNumHeads[s], config.EncPatchSize[s], chunkBytes);
        }

        _unpools = Array.Empty<UtoniaGridUnpooling>();
        _decBlocks = Array.Empty<UtoniaBlock[]>();
        if (!config.EncoderOnly)
        {
            _unpools = new UtoniaGridUnpooling[stages - 1];
            _decBlocks = new UtoniaBlock[stages - 1][];
            for (int s = 0; s < stages - 1; s++)
            {
                int d = config.DecChannels[s];
                long upIn = s + 1 < stages - 1 ? config.DecChannels[s + 1] : config.EncChannels[stages - 1];
                string prefix = UtoniaCheckpointKeys.StagePrefix(true, s);
                _unpools[s] = new UtoniaGridUnpooling(backend,
                    Linear(weights, prefix, UtoniaCheckpointKeys.UpLinear, d, upIn, true),
                    Norm(weights, prefix, UtoniaCheckpointKeys.UpNorm, d),
                    Linear(weights, prefix, UtoniaCheckpointKeys.SkipLinear, d, config.EncChannels[s], true),
                    Norm(weights, prefix, UtoniaCheckpointKeys.SkipNorm, d));
                _decBlocks[s] = BuildStage(backend, weights, config, true, s, config.DecDepths[s], d, config.DecNumHeads[s], config.DecPatchSize[s], chunkBytes);
            }
        }
    }

    /// <summary>Channels of <see cref="ForwardLevel4"/>'s features: the last encoder stage width (576 for CoLa3D).</summary>
    public int BottleneckChannels => _config.EncChannels[^1];

    /// <summary>
    /// Runs the encoder on the input grid and returns the last stage: features after its final block (no norm), and that
    /// stage's grid coords (input index shifted by the total stride, so 0..63 for a 1024 grid). Rows follow the stage's
    /// (batch, x, y, z) order. Coordinates come from <see cref="UtoniaDirectGrid"/> at 0.01 per voxel, not the world span.
    /// </summary>
    /// <param name="coords">Rows of (batch, x, y, z), grouped by increasing batch.</param>
    /// <param name="batchSize">Number of samples; batch ids must lie in [0, batchSize).</param>
    /// <param name="colors">Optional <c>[N, 3]</c> colors used as-is; zeros when null.</param>
    /// <param name="normals">Optional <c>[N, 3]</c> normals; zeros when null.</param>
    public UtoniaFeatures ForwardLevel4(ReadOnlySpan<int> coords, int batchSize, float[]? colors = null, float[]? normals = null)
    {
        ThrowIfDisposed();
        if (coords.Length == 0)
        {
            return new UtoniaFeatures(new Tensor(new TensorShape(0, BottleneckChannels), DType.F32), Array.Empty<int>());
        }

        UtoniaPoint raw = UtoniaDirectGrid.Build(coords, batchSize, colors, normals);
        List<UtoniaPoint> levels = new();
        try
        {
            UtoniaPoint last = EncodeStages(raw, levels);
            Tensor feats = last.TakeFeat();
            return new UtoniaFeatures(feats, last.GridCoord);
        }
        finally
        {
            foreach (UtoniaPoint level in levels)
            {
                level.Dispose();
            }
        }
    }

    /// <summary>Runs the encoder stages on a prebuilt input point (features, coordinates and grid coordinates supplied by the caller, as Sonata's data transform
    /// produces them) and returns every stage's point, finest first. Each level after the first carries <see cref="UtoniaPoint.PoolingInverse"/>, the row of
    /// each finer point in it. The caller disposes the levels; <paramref name="raw"/> is consumed.</summary>
    public IReadOnlyList<UtoniaPoint> ForwardLevels(UtoniaPoint raw)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(raw);
        List<UtoniaPoint> levels = new();
        try
        {
            EncodeStages(raw, levels);
            return levels;
        }
        catch
        {
            foreach (UtoniaPoint level in levels)
            {
                level.Dispose();
            }

            throw;
        }
    }

    /// <summary>Runs the model as built: the encoder alone when encoder-only, otherwise the full encoder-decoder back to the input grid.</summary>
    public UtoniaFeatures Forward(ReadOnlySpan<int> coords, int batchSize, float[]? colors = null, float[]? normals = null)
    {
        ThrowIfDisposed();
        if (_config.EncoderOnly)
        {
            return ForwardLevel4(coords, batchSize, colors, normals);
        }

        if (coords.Length == 0)
        {
            return new UtoniaFeatures(new Tensor(new TensorShape(0, _config.DecChannels[0]), DType.F32), Array.Empty<int>());
        }

        UtoniaPoint raw = UtoniaDirectGrid.Build(coords, batchSize, colors, normals);
        List<UtoniaPoint> levels = new();
        try
        {
            UtoniaPoint point = EncodeStages(raw, levels);
            for (int s = _unpools.Length - 1; s >= 0; s--)
            {
                point = _unpools[s].Forward(point);
                foreach (UtoniaBlock block in _decBlocks[s])
                {
                    block.Forward(point);
                }
            }

            Tensor feats = point.TakeFeat();
            return new UtoniaFeatures(feats, point.GridCoord);
        }
        finally
        {
            foreach (UtoniaPoint level in levels)
            {
                level.Dispose();
            }
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (UtoniaBlock[] stage in _encBlocks)
        {
            DisposeBlocks(stage);
        }

        foreach (UtoniaBlock[] stage in _decBlocks)
        {
            DisposeBlocks(stage);
        }
    }

    private UtoniaPoint EncodeStages(UtoniaPoint raw, List<UtoniaPoint> levels)
    {
        UtoniaPoint point = Embed(raw);
        levels.Add(point);
        point.Serialize(_orders);
        point.Sparsify();
        for (int s = 0; s < _encBlocks.Length; s++)
        {
            if (s > 0)
            {
                point = _pools[s - 1].Forward(point, _orders);
                levels.Add(point);
            }

            foreach (UtoniaBlock block in _encBlocks[s])
            {
                block.Forward(point);
            }
        }

        return point;
    }

    private UtoniaPoint Embed(UtoniaPoint raw)
    {
        long width = _config.EncChannels[0];
        Tensor activated = UtoniaRowChunks.Map(raw.Feat, width, UtoniaRowChunks.RowsPerChunk(_chunkBytes, width), (src, res) =>
        {
            using Tensor linear = new(new TensorShape(src.Shape[0], width), DType.F32);
            _embedLinear.Forward(src, linear);
            using Tensor normed = new(linear.Shape, DType.F32);
            _embedNorm.Forward(linear, normed);
            _backend.GeluErf(res, normed);
        });
        UtoniaPoint embedded = new(activated, raw.Coord, raw.GridCoord, raw.Batch);
        raw.Dispose();
        return embedded;
    }

    private static UtoniaBlock[] BuildStage(IBackend backend, UtoniaWeights weights, UtoniaConfig config, bool decoder, int stage,
        int depth, int channels, int heads, int patchSize, long chunkBytes)
    {
        UtoniaBlock[] blocks = new UtoniaBlock[depth];
        for (int i = 0; i < depth; i++)
        {
            blocks[i] = new UtoniaBlock(backend, weights, UtoniaCheckpointKeys.BlockPrefix(decoder, stage, i), config, channels, heads,
                patchSize, i % config.Order.Length, chunkBytes);
        }

        return blocks;
    }

    private static void DisposeBlocks(UtoniaBlock[] blocks)
    {
        foreach (UtoniaBlock block in blocks)
        {
            block.Dispose();
        }
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

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(UtoniaEncoder));
        }
    }
}

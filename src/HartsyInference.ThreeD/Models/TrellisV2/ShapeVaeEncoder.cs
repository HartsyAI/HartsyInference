using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// TRELLIS.2 shape VAE encoder (<c>FlexiDualGridVaeEncoder</c>, inference path): dual-grid vertex offsets and
/// intersection flags on the finest grid → posterior mean on the grid 16x coarser. CPU reference in F32.
/// </summary>
public sealed unsafe class ShapeVaeEncoder : IDisposable
{
    private readonly int _levels, _firstChannels, _latentChannels;
    private readonly Tensor _inputW, _inputB, _latentW, _latentB;
    private readonly ShapeVaeConvNeXtBlock[][] _blocks;
    private readonly ShapeVaeSpatialToChannelBlock?[] _downs;
    private readonly List<Tensor> _ownedWeights = new();
    private bool _disposed;

    /// <summary>Validates the config and checks the weights against it; throws on any missing or mis-shaped tensor.</summary>
    public ShapeVaeEncoder(ShapeVaeEncoderConfig config, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        Dictionary<string, Tensor> w = ShapeVaeWeights.Resolve(ShapeVaeKeyMap.Encoder(config), weights, _ownedWeights);

        int[] channels = config.ModelChannels;
        _levels = channels.Length;
        _firstChannels = channels[0];
        _latentChannels = config.LatentChannels;
        _inputW = w["input_layer.weight"];
        _inputB = w["input_layer.bias"];
        _latentW = w["to_latent.weight"];
        _latentB = w["to_latent.bias"];

        _blocks = new ShapeVaeConvNeXtBlock[_levels][];
        _downs = new ShapeVaeSpatialToChannelBlock?[_levels];
        for (int i = 0; i < _levels; i++)
        {
            _blocks[i] = new ShapeVaeConvNeXtBlock[config.NumBlocks[i]];
            for (int j = 0; j < config.NumBlocks[i]; j++) _blocks[i][j] = new ShapeVaeConvNeXtBlock(w, $"blocks.{i}.{j}", channels[i]);
            if (i < _levels - 1)
                _downs[i] = new ShapeVaeSpatialToChannelBlock(w, $"blocks.{i}.{config.NumBlocks[i]}", channels[i], channels[i + 1]);
        }
    }

    /// <summary>
    /// Encodes the dual-grid inputs to the latent mean. <paramref name="vertices"/> and <paramref name="intersected"/>
    /// are <c>[N, 3]</c> on identical coords; the grid resolution must be a multiple of 16. Output is <c>[1, M, latent]</c>.
    /// </summary>
    public SparseTensor Encode(IBackend backend, SparseTensor vertices, SparseTensor intersected)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(_disposed, this);
        ShapeVaeOps.RequireInput(vertices, 3, nameof(vertices));
        ShapeVaeOps.RequireInput(intersected, 3, nameof(intersected));
        if (!vertices.Coords.AsSpan().SequenceEqual(intersected.Coords) || vertices.Resolution != intersected.Resolution)
            throw new ArgumentException("Vertices and intersection flags must share coords and resolution.", nameof(intersected));
        int divisor = 1 << (_levels - 1);
        if (vertices.Resolution % divisor != 0)
            throw new ArgumentException($"Encoder resolution {vertices.Resolution} must be a multiple of {divisor}.", nameof(vertices));

        int n = vertices.Count;
        Tensor packed = PackInput(vertices, intersected);
        Tensor h = ShapeVaeOps.Linear(backend, packed, _inputW, _inputB, n, _firstChannels);
        packed.Dispose();
        SparseTensor cur = vertices.Replace(h);

        for (int i = 0; i < _levels; i++)
        {
            foreach (ShapeVaeConvNeXtBlock block in _blocks[i]) cur = ShapeVaeOps.Advance(cur, block.Forward(backend, cur));
            if (i < _levels - 1)
            {
                SparseSpatialChannel.Plan plan = SparseSpatialChannel.PlanSpatialToChannel(cur.Coords, cur.Resolution, 2);
                cur = ShapeVaeOps.Advance(cur, _downs[i]!.Forward(backend, cur, plan));
            }
        }

        // The tail runs on the coarsest grid, whose voxel count is cur.Count rather than the input count.
        int m = cur.Count;
        Tensor normed = ShapeVaeOps.LayerNormNoAffine(backend, cur.Feats, m, cur.Channels, ShapeVaeOps.OutputNormEps);
        cur.Feats.Dispose();
        Tensor moments = ShapeVaeOps.Linear(backend, normed, _latentW, _latentB, m, 2 * _latentChannels);
        normed.Dispose();

        // Mean is the first half of the moments; inference returns it without posterior sampling.
        Tensor mean = ShapeVaeOps.Alloc(m, _latentChannels);
        backend.SliceLastDim(mean, moments, 0);
        moments.Dispose();
        return new SparseTensor(mean, cur.Coords, cur.Resolution);
    }

    /// <summary>Concatenates <c>vertices − 0.5</c> and <c>intersected − 0.5</c> per voxel, as upstream's encoder forward does.</summary>
    private static Tensor PackInput(SparseTensor vertices, SparseTensor intersected)
    {
        int n = vertices.Count;
        Tensor input = ShapeVaeOps.Alloc(n, ShapeVaeKeyMap.EncoderInputChannels);
        float* v = (float*)vertices.Feats.DataPointer, e = (float*)intersected.Feats.DataPointer, d = (float*)input.DataPointer;
        for (int i = 0; i < n; i++)
            for (int c = 0; c < 3; c++)
            {
                d[i * 6 + c] = v[i * 3 + c] - 0.5f;
                d[i * 6 + 3 + c] = e[i * 3 + c] - 0.5f;
            }
        return input;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (ShapeVaeConvNeXtBlock[] level in _blocks) foreach (ShapeVaeConvNeXtBlock block in level) block.Dispose();
        foreach (ShapeVaeSpatialToChannelBlock? down in _downs) down?.Dispose();
        foreach (Tensor t in _ownedWeights) t.Dispose();
        _ownedWeights.Clear();
    }
}

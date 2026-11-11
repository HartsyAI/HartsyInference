using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// TRELLIS.2 shape VAE decoder (<c>FlexiDualGridVaeDecoder</c>, inference path): latent on the coarse grid →
/// <see cref="ShapeVaeKeyMap.DecoderOutputChannels"/> dual-grid head values on the finest grid. CPU reference in F32.
/// The head is returned raw; the sigmoid, intersection threshold and mesh extraction are downstream steps.
/// </summary>
public sealed unsafe class ShapeVaeDecoder : IDisposable
{
    private readonly int _levels, _firstChannels, _latentChannels;
    private readonly Tensor _fromLatentW, _fromLatentB, _outputW, _outputB;
    private readonly ShapeVaeConvNeXtBlock[][] _blocks;
    private readonly ShapeVaeChannelToSpatialBlock?[] _ups;
    private readonly List<Tensor> _ownedWeights = new();
    private bool _disposed;

    /// <summary>Validates the config and checks the weights against it; throws on any missing or mis-shaped tensor.</summary>
    public ShapeVaeDecoder(ShapeVaeDecoderConfig config, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        Dictionary<string, Tensor> w = ShapeVaeWeights.Resolve(ShapeVaeKeyMap.Decoder(config), weights, _ownedWeights);

        int[] channels = config.ModelChannels;
        _levels = channels.Length;
        _firstChannels = channels[0];
        _latentChannels = config.LatentChannels;
        _fromLatentW = w["from_latent.weight"];
        _fromLatentB = w["from_latent.bias"];
        _outputW = w["output_layer.weight"];
        _outputB = w["output_layer.bias"];

        _blocks = new ShapeVaeConvNeXtBlock[_levels][];
        _ups = new ShapeVaeChannelToSpatialBlock?[_levels];
        for (int i = 0; i < _levels; i++)
        {
            _blocks[i] = new ShapeVaeConvNeXtBlock[config.NumBlocks[i]];
            for (int j = 0; j < config.NumBlocks[i]; j++) _blocks[i][j] = new ShapeVaeConvNeXtBlock(w, $"blocks.{i}.{j}", channels[i]);
            if (i < _levels - 1)
                _ups[i] = new ShapeVaeChannelToSpatialBlock(w, $"blocks.{i}.{config.NumBlocks[i]}", channels[i], channels[i + 1]);
        }
    }

    /// <summary>
    /// Decodes a latent <c>[N, latent]</c> on its coarse grid to the raw head <c>[1, M, 7]</c> on the grid
    /// <c>2^(levels-1)</c> times finer. Voxels are created where the subdivision head predicts a child.
    /// </summary>
    public SparseTensor Decode(IBackend backend, SparseTensor latent)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(_disposed, this);
        ShapeVaeOps.RequireInput(latent, _latentChannels, nameof(latent));

        SparseTensor cur = RunStages(backend, latent, _levels);
        int m = cur.Count;
        Tensor normed = ShapeVaeOps.LayerNormNoAffine(backend, cur.Feats, m, cur.Channels, ShapeVaeOps.OutputNormEps);
        cur.Feats.Dispose();
        Tensor head = ShapeVaeOps.Linear(backend, normed, _outputW, _outputB, m, ShapeVaeKeyMap.DecoderOutputChannels);
        normed.Dispose();
        return new SparseTensor(head, cur.Coords, cur.Resolution);
    }

    /// <summary>
    /// Upstream <c>upsample(x, upsample_times)</c>: runs the first <paramref name="times"/> levels, each with its subdivision step,
    /// and returns the coordinates <c>(batch, x, y, z)</c> of the voxels they create on the grid <c>2^times</c> times finer. It stops
    /// before the output head, so it costs what the decode costs up to that level and no more.
    /// </summary>
    /// <param name="times">Levels to run, from 1 to one below the level count.</param>
    public int[] Upsample(IBackend backend, SparseTensor latent, int times)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (times < 1 || times > _levels - 1)
            throw new ArgumentOutOfRangeException(nameof(times), times, $"Upsample times must be within [1, {_levels - 1}].");
        ShapeVaeOps.RequireInput(latent, _latentChannels, nameof(latent));
        SparseTensor cur = RunStages(backend, latent, times);
        cur.Feats.Dispose();
        return cur.Coords;
    }

    /// <summary>The from-latent projection and the first <paramref name="levels"/> levels with their upsample blocks (none after the last level). The caller owns the returned features.</summary>
    private SparseTensor RunStages(IBackend backend, SparseTensor latent, int levels)
    {
        int n = latent.Count;
        Tensor h = ShapeVaeOps.Linear(backend, latent.Feats, _fromLatentW, _fromLatentB, n, _firstChannels);
        SparseTensor cur = latent.Replace(h);

        try
        {
            for (int i = 0; i < levels; i++)
            {
                foreach (ShapeVaeConvNeXtBlock block in _blocks[i]) cur = ShapeVaeOps.Advance(cur, block.Forward(backend, cur));
                if (i < _levels - 1) cur = ShapeVaeOps.Advance(cur, _ups[i]!.Forward(backend, cur));
            }
        }
        catch
        {
            // cur.Feats is always a buffer this method owns (h or a block output), so it is released on failure.
            cur.Feats.Dispose();
            throw;
        }

        return cur;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (ShapeVaeConvNeXtBlock[] level in _blocks) foreach (ShapeVaeConvNeXtBlock block in level) block.Dispose();
        foreach (ShapeVaeChannelToSpatialBlock? up in _ups) up?.Dispose();
        foreach (Tensor t in _ownedWeights) t.Dispose();
        _ownedWeights.Clear();
    }
}

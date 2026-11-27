using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// TRELLIS.2 structured-latent flow (<c>SLatFlowModel</c>): predicts the rectified-flow velocity of one latent row per sparse
/// voxel. Every voxel is a token; the 3D RoPE reads the voxel's integer coordinates, so one set of weights serves any grid
/// size. Conditioned on image tokens by cross-attention. Borrows <see cref="Trellis2FlowWeights"/>, which must outlive it.
/// </summary>
public sealed class Trellis2SlatFlow : IDisposable
{
    private readonly Trellis2FlowCore _core;
    private readonly Trellis2FlowConfig _config;
    private Trellis2RopeTable? _rope;
    private int[]? _ropeCoords;
    private int _disposed;

    public Trellis2SlatFlow(Trellis2FlowWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (weights.Config.Dense)
            throw new ArgumentException("The structured-latent flow needs the sparse (SLatFlowModel) architecture.", nameof(weights));
        _config = weights.Config;
        _core = new Trellis2FlowCore(weights);
    }

    public Trellis2FlowConfig Config => _config;

    /// <summary>The tensors the flow reads, for <c>IBackend.PreloadWeights</c> and <c>FreeWeights</c>.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _core.EnumerateWeights();

    /// <summary>Velocity for <paramref name="x"/> (<c>[1, N, InChannels]</c> features over <paramref name="coords"/>) at model timestep
    /// <paramref name="tModel"/>, attending to <paramref name="cond"/>. <paramref name="coords"/> are <c>(batch, x, y, z)</c> rows; the
    /// array must keep its identity across the steps of one sample so the RoPE table is built once. The caller owns the result.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int[] coords, float tModel, Tensor cond) => Forward(backend, x, coords, tModel, cond, null);

    /// <summary>As <see cref="Forward(IBackend, Tensor, int[], float, Tensor)"/> for a projection flow, which also takes <paramref name="proj"/>
    /// <c>[1, N, ProjInChannels]</c>, one feature row per voxel in the order of <paramref name="coords"/>.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int[] coords, float tModel, Tensor cond, Tensor? proj)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(coords);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int n = coords.Length / 4;
        if (coords.Length % 4 != 0 || n == 0)
            throw new ArgumentException("Coordinates must be a non-empty list of (batch, x, y, z) rows.", nameof(coords));
        if (x.Shape.Rank != 3 || x.Shape[0] != 1 || x.Shape[1] != n || x.Shape[2] != _config.InChannels)
            throw new ArgumentException($"Features must be [1, {n}, {_config.InChannels}]; got {x.Shape}.", nameof(x));
        if (cond.Shape.Rank != 3 || cond.Shape[0] != 1 || cond.Shape[2] != _config.CondChannels || cond.Shape[1] == 0)
            throw new ArgumentException($"Conditioning must be [1, L, {_config.CondChannels}]; got {cond.Shape}.", nameof(cond));

        _core.CheckProjection(proj, n);
        if (_rope is null || !ReferenceEquals(_ropeCoords, coords))
        {
            _rope?.Dispose();
            _rope = Trellis2RopeTable.Build(_config.HeadDim, Trellis2RopeTable.VoxelPositions(coords), n);
            _ropeCoords = coords;
        }

        Tensor h = _core.Embed(backend, x);
        using Tensor mod = _core.SharedModulation(backend, tModel);
        Tensor blocks = _core.RunBlocks(backend, h, mod, cond, proj, _rope);
        return _core.Project(backend, blocks);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _rope?.Dispose();
        _core.Dispose();
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// TRELLIS.2 stage-1 flow (<c>SparseStructureFlowModel</c>): predicts the rectified-flow velocity of a dense
/// <c>[1, C, R, R, R]</c> latent, one token per cell in x-major order, conditioned on image tokens by cross-attention.
/// Borrows <see cref="Trellis2FlowWeights"/>, which must outlive it.
/// </summary>
public sealed class Trellis2SparseStructureFlow : IDisposable
{
    private readonly Trellis2FlowCore _core;
    private readonly Trellis2FlowConfig _config;
    private Trellis2RopeTable? _rope;
    private int _disposed;

    public Trellis2SparseStructureFlow(Trellis2FlowWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (!weights.Config.Dense)
            throw new ArgumentException("The sparse-structure flow needs the dense (SparseStructureFlowModel) architecture.", nameof(weights));
        _config = weights.Config;
        _core = new Trellis2FlowCore(weights);
    }

    public Trellis2FlowConfig Config => _config;

    /// <summary>The tensors the flow reads, for <c>IBackend.PreloadWeights</c> and <c>FreeWeights</c>.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _core.EnumerateWeights();

    /// <summary>Velocity for <paramref name="latent"/> <c>[1, C, R, R, R]</c> at model timestep <paramref name="tModel"/> (<c>1000 · t</c>),
    /// attending to <paramref name="cond"/> <c>[1, L, condChannels]</c>. The caller owns the result.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, float tModel, Tensor cond)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int r = _config.Resolution, tokens = _config.DenseTokens, cin = _config.InChannels;
        if (latent.Shape.Rank != 5 || latent.Shape[0] != 1 || latent.Shape[1] != cin || latent.Shape[2] != r || latent.Shape[3] != r || latent.Shape[4] != r)
            throw new ArgumentException($"Latent must be [1, {cin}, {r}, {r}, {r}]; got {latent.Shape}.", nameof(latent));
        RequireCond(cond);

        using Tensor channelsFirst = latent.Reshape(new TensorShape(1, cin, tokens));
        using Tensor rows = new(new TensorShape(1, tokens, cin), DType.F32);
        backend.Transpose2D(rows, channelsFirst, cin, tokens);
        Tensor h = _core.Embed(backend, rows);
        _rope ??= Trellis2RopeTable.Build(_config.HeadDim, Trellis2RopeTable.DenseGridPositions(r), tokens);
        using Tensor mod = _core.SharedModulation(backend, tModel);
        Tensor blocks = _core.RunBlocks(backend, h, mod, cond, _rope);
        using Tensor projected = _core.Project(backend, blocks);
        using Tensor back = new(new TensorShape(1, _config.OutChannels, tokens), DType.F32);
        backend.Transpose2D(back, projected, tokens, _config.OutChannels);
        Tensor velocity = new(new TensorShape([1L, _config.OutChannels, r, r, r]), DType.F32);
        backend.CopyInto(velocity, back.Reshape(velocity.Shape));
        return velocity;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _rope?.Dispose();
        _core.Dispose();
    }

    private void RequireCond(Tensor cond)
    {
        if (cond.Shape.Rank != 3 || cond.Shape[0] != 1 || cond.Shape[2] != _config.CondChannels || cond.Shape[1] == 0)
            throw new ArgumentException($"Conditioning must be [1, L, {_config.CondChannels}]; got {cond.Shape}.", nameof(cond));
    }
}

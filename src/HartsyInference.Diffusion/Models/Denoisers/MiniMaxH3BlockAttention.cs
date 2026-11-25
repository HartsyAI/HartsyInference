using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The attention step of one H3 block: exactly one route, chosen by a factory. A block can no longer be handed two masks or two
/// routes at once. The checks that depend on the chunked width stay with the block (<c>ForwardNamedBlock</c>), since only it knows the width.</summary>
internal readonly struct BlockAttention
{
    private readonly Tensor? _denseBias;
    private readonly Func<int, int, Tensor>? _tiledBias;
    private readonly IVideoSparseAttentionSession? _sparse;
    private readonly Func<Tensor, Tensor>? _override;

    private BlockAttention(Tensor? denseBias, Func<int, int, Tensor>? tiledBias, IVideoSparseAttentionSession? sparse, Func<Tensor, Tensor>? run)
    {
        _denseBias = denseBias;
        _tiledBias = tiledBias;
        _sparse = sparse;
        _override = run;
    }

    /// <summary>Plain attention: dense below the chunk width, chunked above it, with no mask. This is the default value.</summary>
    public static BlockAttention Plain => default;

    /// <summary>A borrowed [S, S] bias applied on the dense route. The chunked route refuses it.</summary>
    public static BlockAttention Dense(Tensor bias) => new(bias ?? throw new ArgumentNullException(nameof(bias)), null, null, null);

    /// <summary>A per-tile bias builder, called as (rowStart, rows), applied on the chunked route. The dense route refuses it.</summary>
    public static BlockAttention Tiled(Func<int, int, Tensor> tile) => new(null, tile ?? throw new ArgumentNullException(nameof(tile)), null, null);

    /// <summary>A sparse video attention session. It takes precedence over any mask route.</summary>
    public static BlockAttention Sparse(IVideoSparseAttentionSession session) => new(null, null, session ?? throw new ArgumentNullException(nameof(session)), null);

    /// <summary>The caller's whole attention step, given the block's modulated input. The route checks do not apply to it.</summary>
    public static BlockAttention Override(Func<Tensor, Tensor> run) => new(null, null, null, run ?? throw new ArgumentNullException(nameof(run)));

    internal Tensor? DenseBias => _denseBias;

    internal Func<int, int, Tensor>? TiledBias => _tiledBias;

    internal IVideoSparseAttentionSession? SparseSession => _sparse;

    internal Func<Tensor, Tensor>? Run => _override;
}

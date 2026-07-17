using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Video.Models.Cosmos;

/// <summary>Additive 3D absolute-position embedding added to token embeddings before the transformer stack — the
/// V2W-only <c>apply_abs_pos_emb</c> input signal (research-doc §6), on top of the in-attention 3D RoPE. The table
/// is <c>[T·H·W, dim]</c> in row-major <c>(t, h, w)</c> order, either a <b>learned</b> table lifted from the
/// checkpoint (<c>pos_emb.weight</c>) or a frozen sin-cos fallback seeded from
/// <see cref="DiTUtils.Sincos3DPositionEmbedding"/>. Adds host-side (the token-path embeddings are host F32).
///
/// <para><b>Integration-pending</b>: whether the shipped table is learned or factored (separate T/H/W embeddings),
/// and its exact tensor name/shape, resolve from the <c>model.pt</c> dump (research-doc Open Q §5). A learned table
/// wins when present; otherwise the sin-cos grid is a structurally-correct stand-in.</para></summary>
public sealed unsafe class CosmosAbsPos3D
{
    private readonly int _dim;
    private Tensor? _table;   // [N, dim], owned when generated; borrowed when a learned table is supplied

    /// <summary>Creates the abs-pos for hidden width <paramref name="dim"/>.</summary>
    public CosmosAbsPos3D(int dim) => _dim = dim;

    /// <summary>Number of position rows currently built, or 0.</summary>
    public int Length => _table is null ? 0 : (int)_table.Shape[0];

    /// <summary>Supplies a learned <c>[N, dim]</c> table from the checkpoint (borrowed — not disposed here).</summary>
    public void LoadLearnedTable(Tensor table)
    {
        if (table.Shape.Rank != 2 || (int)table.Shape[1] != _dim)
            throw new ArgumentException($"abs-pos table must be [N, {_dim}], got {table.Shape}.", nameof(table));
        _table = table;
    }

    /// <summary>Builds the frozen sin-cos table for the given latent grid (used only when no learned table was
    /// loaded). Idempotent for a matching grid.</summary>
    public void BuildSincos(int latentT, int latentH, int latentW)
    {
        if (_table is not null) return;
        _table = DiTUtils.Sincos3DPositionEmbedding(latentT, latentH, latentW, _dim);
    }

    /// <summary>Adds <c>table[startIndex + s]</c> into <paramref name="embeds"/> <c>[1, count, dim]</c> (host F32),
    /// row <c>s = 0..count-1</c>. Called on the prefill block and on each decode step's single input embedding.</summary>
    public void AddSlice(Tensor embeds, int startIndex, int count)
    {
        if (_table is null) throw new InvalidOperationException("abs-pos table not built (call LoadLearnedTable or BuildSincos).");
        int n = (int)_table.Shape[0];
        if (startIndex < 0 || startIndex + count > n)
            throw new ArgumentOutOfRangeException(nameof(startIndex), $"[{startIndex},{startIndex + count}) out of table range [0,{n}).");
        float* ep = (float*)embeds.DataPointer;
        float* tp = (float*)_table.DataPointer;
        for (int s = 0; s < count; s++)
        {
            long src = (long)(startIndex + s) * _dim;
            long dst = (long)s * _dim;
            for (int j = 0; j < _dim; j++) ep[dst + j] += tp[src + j];
        }
    }
}

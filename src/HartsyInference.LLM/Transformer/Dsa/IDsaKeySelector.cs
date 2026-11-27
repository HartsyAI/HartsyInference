using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer.Dsa;

/// <summary>Decides which cached latent rows each query of a <see cref="DsaAttention"/> layer attends. ERNIE-5.5 and Kimi pick a token top-k plus a
/// sliding window (<see cref="DsaIndexerSelector"/>); Hy4 Flash recalls whole pooled groups, or only the window on its sliding layers.</summary>
/// <remarks>Slots are cache positions; they equal the rotary positions of a text-only sequence.</remarks>
public interface IDsaKeySelector : IDisposable
{
    /// <summary>Width of the per-position F32 key row this selector keeps in <see cref="DsaAttentionState.IndexKeys"/>; 0 when it keeps none.</summary>
    int KeyWidth { get; }

    /// <summary>Writes the selector key row of each of <paramref name="tokens"/> new positions at <paramref name="startPos"/> into <paramref name="keyStore"/>.</summary>
    /// <param name="x">Normalized hidden states, <c>[tokens, dim]</c>.</param>
    void Append(ReadOnlySpan<float> x, int tokens, int startPos, Tensor keyStore);

    /// <summary>Picks each query rows once <paramref name="length"/> positions are cached (the new chunk included).</summary>
    /// <param name="x">Normalized hidden states, <c>[tokens, dim]</c>.</param>
    /// <param name="qLatent">Normalized query latents, <c>[tokens, qLoraRank]</c>.</param>
    /// <param name="positions">Each query slot.</param>
    /// <param name="length">Cached positions.</param>
    /// <param name="keyStore">The keys written by <see cref="Append"/>.</param>
    DsaKeySelection Select(ReadOnlySpan<float> x, ReadOnlySpan<float> qLatent, ReadOnlySpan<int> positions, int length, Tensor keyStore);
}

using HartsyInference.LLM.Transformer.Dsa;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>One sequence's state across the Hy4 Flash layers: each layer's latent and indexer-key cache and the token history the Engram n-grams read back into.</summary>
public sealed class Hy4FlashSequenceState : IDisposable
{
    /// <param name="layers">One cache per decoder layer.</param>
    public Hy4FlashSequenceState(DsaAttentionState[] layers) => Layers = layers ?? throw new ArgumentNullException(nameof(layers));

    /// <summary>Per-layer caches.</summary>
    public DsaAttentionState[] Layers { get; }

    /// <summary>Every token id fed so far.</summary>
    public List<int> Ids { get; } = [];

    /// <summary>Positions processed so far.</summary>
    public int Length { get; internal set; }

    /// <summary>The rotary position the next text token takes: one past the largest position so far (an image jumps it ahead of the slot count).</summary>
    public int NextPosition { get; internal set; }

    /// <summary>Images consumed so far, so a later chunk takes the next one.</summary>
    public int ImagesUsed { get; internal set; }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (DsaAttentionState layer in Layers) layer.Dispose();
    }
}

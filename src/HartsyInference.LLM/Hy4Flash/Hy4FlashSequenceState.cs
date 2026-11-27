using HartsyInference.LLM.Transformer.Dsa;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>One sequence's state across the Hy4 Flash layers: each layer's latent and indexer-key cache and the token history the Engram n-grams read back into.</summary>
public sealed class Hy4FlashSequenceState : IDisposable
{
    /// <param name="layers">One cache per decoder layer.</param>
    /// <param name="indexedRope">This state's rotary rows of the indexed base, one slot per position it can hold.</param>
    /// <param name="slidingRope">This state's rotary rows of the sliding base, the same size as <paramref name="indexedRope"/>.</param>
    internal Hy4FlashSequenceState(DsaAttentionState[] layers, Hy4FlashMropeRows indexedRope, Hy4FlashMropeRows slidingRope)
    {
        Layers = layers ?? throw new ArgumentNullException(nameof(layers));
        IndexedRope = indexedRope ?? throw new ArgumentNullException(nameof(indexedRope));
        SlidingRope = slidingRope ?? throw new ArgumentNullException(nameof(slidingRope));
    }

    /// <summary>Per-layer caches.</summary>
    public DsaAttentionState[] Layers { get; }

    /// <summary>This state's rotary rows of the indexed base; each slot's row is written when the slot is given its position.</summary>
    internal Hy4FlashMropeRows IndexedRope { get; }

    /// <summary>This state's rotary rows of the sliding base, written alongside <see cref="IndexedRope"/>.</summary>
    internal Hy4FlashMropeRows SlidingRope { get; }

    /// <summary>The model that created this state; the model refuses a state of another model, whose rotary rows and slots would not match.</summary>
    internal object? Owner { get; set; }

    /// <summary>Whether the last call ended inside an image placeholder run, so the next call must not open with another one.</summary>
    internal bool EndsInImageRun { get; set; }

    /// <summary>Every token id fed so far.</summary>
    public List<int> Ids { get; } = [];

    /// <summary>Positions processed so far.</summary>
    public int Length { get; internal set; }

    /// <summary>The rotary position the next text token takes: one past the largest position so far (an image jumps it ahead of the slot count).</summary>
    public int NextPosition { get; internal set; }

    /// <summary>Images consumed so far, so a later chunk takes the next one.</summary>
    public int ImagesUsed { get; internal set; }

    /// <summary>Empties every layer's cache and the token history; the next prefill starts at position 0.</summary>
    public void Reset()
    {
        Length = 0;
        NextPosition = 0;
        ImagesUsed = 0;
        EndsInImageRun = false;
        Ids.Clear();
        foreach (DsaAttentionState layer in Layers) layer.Reset();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (DsaAttentionState layer in Layers) layer.Dispose();
    }
}

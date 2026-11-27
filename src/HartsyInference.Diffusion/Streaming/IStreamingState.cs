namespace HartsyInference.Diffusion.Streaming;

/// <summary>State a streaming model carries from one chunk to the next: a recurrent state, a convolution's left context, a
/// window of attention keys and values. A chunk reads the committed state on every pass; the pass that finishes the chunk
/// stages the next state, and <see cref="Commit"/> makes every staged write current at once, so a chunk that fails or is
/// cancelled part-way leaves the committed state as it was.</summary>
public interface IStreamingState : IDisposable
{
    /// <summary>Makes the staged writes current and releases what they replace. Does nothing when nothing is staged.</summary>
    void Commit();

    /// <summary>Releases the staged writes; the committed state is unchanged.</summary>
    void Discard();

    /// <summary>Releases committed and staged state alike, returning to the state of a stream that has not started.</summary>
    void Reset();
}

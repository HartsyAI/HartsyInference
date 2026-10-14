namespace HartsyInference.Audio.Sampling;

/// <summary>The autoregressive model <see cref="BeamSearch"/> drives: one decoding state (KV cache) per beam slot.</summary>
public interface IBeamSearchModel
{
    /// <summary>Feeds <paramref name="token"/> to beam slot <paramref name="beam"/> as its generated token number
    /// <paramref name="step"/> (step 0 feeds the start token) and writes the next-token logits into
    /// <paramref name="logits"/>. Called only for beams that can still win.</summary>
    void Step(int beam, int token, int step, Span<float> logits);

    /// <summary>Rearranges the per-beam states: slot <c>b</c> continues from the state slot <c>parents[b]</c> held.
    /// Several slots may share a parent.</summary>
    void Reorder(ReadOnlySpan<int> parents);
}

namespace HartsyInference.Conditioning;

/// <summary>Where a conditioning stream is consumed inside the model — pipelines route each stream produced by an <see cref="IActionEncoder"/> to the matching injection site. See <c>docs/Research/INTERACTIVE_INFERENCE.md</c> § 2.</summary>
public enum ActionStreamRole
{
    /// <summary>Continuous stream joined to image tokens and self-attended temporally (e.g. Matrix-Game mouse).</summary>
    PerBlockSelfAttn,
    /// <summary>Discrete stream cross-attended into the hidden state (e.g. Matrix-Game keyboard).</summary>
    PerBlockCrossAttn,
    /// <summary>Camera-ray (Plücker) map added to patch embeddings (Matrix-Game 3, GameCraft).</summary>
    PluckerMap,
    /// <summary>Vector added to the per-frame timestep embedding (Oasis).</summary>
    TimestepAddon,
}

/// <summary>Static description of one conditioning stream an encoder produces per frame.</summary>
/// <param name="Name">Stream key the pipeline routes by (e.g. <c>"keyboard"</c>, <c>"mouse"</c>).</param>
/// <param name="Role">Injection site inside the model.</param>
/// <param name="FloatsPerFrame">Floats written per <see cref="IActionEncoder.Encode"/> call.</param>
public readonly record struct ActionStreamSpec(string Name, ActionStreamRole Role, int FloatsPerFrame);

/// <summary>Translates raw <see cref="ActionInput"/> payloads into the per-frame conditioning streams a specific model consumes. Encoders fill caller-owned buffers (<see cref="Span{T}"/>) — no allocations on the 25–40 ms frame path. Models without action conditioning simply have no encoder. See <c>docs/Research/INTERACTIVE_INFERENCE.md</c> § 2.</summary>
public interface IActionEncoder : IDisposable
{
    /// <summary>The streams this encoder produces; one <see cref="Encode"/> call fills one frame of one stream.</summary>
    IReadOnlyList<ActionStreamSpec> Streams { get; }

    /// <summary>Writes one frame of <paramref name="streamName"/> into <paramref name="destination"/> (length must be the stream's <see cref="ActionStreamSpec.FloatsPerFrame"/>).</summary>
    void Encode(in ActionInput action, string streamName, Span<float> destination);
}

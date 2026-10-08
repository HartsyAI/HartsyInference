namespace HartsyInference.Core.Backends;

/// <summary>Where an expert's authoritative bytes live before any cache copy. The runtime, not the model, chooses policy from this.</summary>
public enum ExpertBacking
{
    /// <summary>Ordinary host memory.</summary>
    ResidentHost,

    /// <summary>Memory-mapped checkpoint pages.</summary>
    MemoryMapped,

    /// <summary>A packed expert file written by the expert-pack tool.</summary>
    Pack,

    /// <summary>Already resident on a device.</summary>
    Device,
}

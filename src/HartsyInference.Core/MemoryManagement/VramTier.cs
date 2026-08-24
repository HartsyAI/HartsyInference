namespace HartsyInference.Core.MemoryManagement;

/// <summary>How much speed the engine may trade for VRAM headroom. Each tier is a preset over the individual levers in <see cref="VramPolicy"/>, so picking a tier and overriding one lever are the same mechanism.</summary>
public enum VramTier
{
    /// <summary>Never stream, never auto-evict; an oversized model fails loudly. The power-user escape hatch.</summary>
    Performance = 0,

    /// <summary>Seed the posture from the device's total VRAM, then decide each phase against measured free bytes.</summary>
    Auto = 1,

    /// <summary>Always release a phase's weights at its boundary; stream only when the measurement says it is needed.</summary>
    Balanced = 2,

    /// <summary>Stream even where the resident layout would fit, halve cache precision, and shrink chunk sizes.</summary>
    Aggressive = 3,

    /// <summary>Every lever on, including activation offload, quantized compute, and spilling onto a second GPU.</summary>
    Maximum = 4,
}

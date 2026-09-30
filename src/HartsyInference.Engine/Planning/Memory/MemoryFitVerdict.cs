namespace HartsyInference.Engine.Planning.Memory;

/// <summary>Whether a generation can run on one engine's device, and how.</summary>
public enum MemoryFitVerdict
{
    /// <summary>The device cannot report its memory (CPU, or a backend without a VRAM query), so nothing is claimed.</summary>
    Unknown,

    /// <summary>Every phase fits with its weights fully resident: the full-speed path.</summary>
    Resident,

    /// <summary>Only fits by streaming the denoiser's blocks through a sliding window: works, typically several times
    /// slower than resident.</summary>
    Streamed,

    /// <summary>Does not fit even with every lever the effective policy allows and this model wires.</summary>
    Infeasible,
}

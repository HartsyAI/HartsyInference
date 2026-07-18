namespace HartsyInference.Engine;

/// <summary>Real-weight validation maturity of a catalogued model, mirroring the README status legend.</summary>
public enum ModelStatus
{
    /// <summary>Verified end-to-end against a reference (README ✅).</summary>
    Verified,

    /// <summary>Built end-to-end, numerics still being validated (README 🧪).</summary>
    ValidationPending,

    /// <summary>Interfaces wired, forward pass still in progress (README 🏗️).</summary>
    Structural,
}

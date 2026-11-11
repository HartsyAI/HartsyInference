namespace HartsyInference.Engine.Worlds;

/// <summary>How a steered world takes its actions.</summary>
public enum WorldControlCadence
{
    /// <summary>Sent when the held keys change; the session keeps the last action until the next change.</summary>
    OnChange,

    /// <summary>Sent once before each step the session takes; a per-frame session consumes one action per step.</summary>
    PerStep,
}

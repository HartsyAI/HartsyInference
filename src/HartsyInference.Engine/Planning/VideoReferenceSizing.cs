namespace HartsyInference.Engine.Planning;

/// <summary>How reference visual media is sized before MiniMax-H3 VAE encoding.</summary>
public enum VideoReferenceSizing
{
    /// <summary>Use the model's native reference-canvas policy.</summary>
    Native = 0,

    /// <summary>Match the target canvas area while preserving the reference aspect ratio.</summary>
    MatchTarget = 1,
}

namespace HartsyInference.Engine.Requests;

/// <summary>The author role of a chat message.</summary>
public enum TextRole
{
    /// <summary>System / developer instruction.</summary>
    System,

    /// <summary>End-user turn.</summary>
    User,

    /// <summary>Model turn.</summary>
    Assistant,

    /// <summary>Tool / function result fed back to the model.</summary>
    Tool,
}

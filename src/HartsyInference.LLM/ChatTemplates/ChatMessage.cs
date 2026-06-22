namespace HartsyInference.LLM.ChatTemplates;

/// <summary>A single ChatML conversation turn. <paramref name="Role"/> is one of "system", "user", or "assistant".</summary>
public sealed record ChatMessage(string Role, string Content)
{
    /// <summary>Creates a "system" turn.</summary>
    public static ChatMessage System(string c) => new("system", c);

    /// <summary>Creates a "user" turn.</summary>
    public static ChatMessage User(string c) => new("user", c);

    /// <summary>Creates an "assistant" turn.</summary>
    public static ChatMessage Assistant(string c) => new("assistant", c);
}

/// <summary>What <see cref="ToolLoopOptions.OnBeforeToolCall"/> decided: run the call, or refuse it and feed
/// <see cref="DenialResult"/> back as its result.</summary>
public readonly record struct ToolCallDecision(bool Allowed, string? DenialResult)
{
    /// <summary>Run the call.</summary>
    public static ToolCallDecision Allow => new(true, null);

    /// <summary>Refuse the call; <paramref name="result"/> is the tool result the model reads.</summary>
    public static ToolCallDecision Deny(string result) => new(false, result);
}

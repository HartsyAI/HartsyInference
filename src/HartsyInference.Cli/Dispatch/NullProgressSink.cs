namespace HartsyInference.Cli.Dispatch;

/// <summary>A progress sink that discards everything, for <c>--quiet</c> / JSON output where only the final result
/// is printed.</summary>
public sealed class NullProgressSink : IProgressSink
{
    /// <inheritdoc/>
    public void Stage(string message)
    {
    }

    /// <inheritdoc/>
    public void BeginSteps(string label, int totalSteps)
    {
    }

    /// <inheritdoc/>
    public void Step(int current, string? detail = null)
    {
    }

    /// <inheritdoc/>
    public void Token(string text)
    {
    }

    /// <inheritdoc/>
    public void EndSteps(string? summary = null)
    {
    }
}

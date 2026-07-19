using HartsyInference.Engine.Dispatch;

namespace HartsyInference.Engine.Services;

/// <summary>Adapts the internal handler <see cref="IProgressSink"/> onto the typed services' progress callbacks:
/// step ticks flow to an <see cref="IProgress{StepPreview}"/>, streamed token pieces flow to an action.</summary>
internal sealed class SinkBridge : IProgressSink
{
    private readonly IProgress<StepPreview>? _steps;
    private readonly Action<string>? _onToken;
    private int _total;

    /// <summary>Creates a bridge; either callback may be null when that channel is unused.</summary>
    public SinkBridge(IProgress<StepPreview>? steps, Action<string>? onToken)
    {
        _steps = steps;
        _onToken = onToken;
    }

    /// <inheritdoc/>
    public void Stage(string message)
    {
    }

    /// <inheritdoc/>
    public void BeginSteps(string label, int totalSteps) => _total = totalSteps;

    /// <inheritdoc/>
    public void Step(int current, string? detail = null) =>
        _steps?.Report(new StepPreview { Step = current, TotalSteps = _total });

    /// <inheritdoc/>
    public void Token(string text) => _onToken?.Invoke(text);

    /// <inheritdoc/>
    public void EndSteps(string? summary = null)
    {
    }
}

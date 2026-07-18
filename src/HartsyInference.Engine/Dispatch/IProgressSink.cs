namespace HartsyInference.Engine.Dispatch;

/// <summary>How handlers report progress without binding to a concrete console/TUI, so they stay testable and the
/// CLI can swap a live display for a quiet/JSON mode.</summary>
public interface IProgressSink
{
    /// <summary>Announces a discrete loading/setup stage (e.g. "Loading UNet…").</summary>
    void Stage(string message);

    /// <summary>Begins a step counter; <paramref name="totalSteps"/> &lt;= 0 means indeterminate.</summary>
    void BeginSteps(string label, int totalSteps);

    /// <summary>Advances the current step counter.</summary>
    void Step(int current, string? detail = null);

    /// <summary>Emits a streamed text chunk (e.g. an LLM token piece) inline.</summary>
    void Token(string text);

    /// <summary>Ends the current step counter, optionally printing a summary line.</summary>
    void EndSteps(string? summary = null);
}

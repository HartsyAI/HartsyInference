using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Sees every decoded content delta of one generation before it reaches the stream and may replace it, complete a <see cref="NativeToolCall"/>, or stop generation. One instance per request, created by <see cref="EngineOptions.TextStreamFilterFactory"/>; the engine carries no format knowledge, so tool-call parsers live outside it.</summary>
/// <remarks>Both members run on the decode thread inside the model slot lock and the device gate: return promptly, never block or do I/O. <see cref="OnEnd"/> is called once after the last delta of a generation that ran to completion and is not called after a result with <see cref="TextFilterResult.Stop"/> set.</remarks>
public interface ITextStreamFilter
{
    /// <summary>Filters one decoded delta; the result says what to forward, whether a tool call completed, and whether to stop.</summary>
    TextFilterResult OnDelta(string delta);

    /// <summary>Flushes anything held back once generation ends naturally.</summary>
    TextFilterResult OnEnd();
}

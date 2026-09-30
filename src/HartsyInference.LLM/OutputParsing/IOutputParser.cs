namespace HartsyInference.LLM.OutputParsing;

/// <summary>Incrementally turns generated token ids into reasoning, content and tool-call events. Not thread-safe; one instance per generation.</summary>
public interface IOutputParser
{
    /// <summary>Feeds one generated token; events it completes are sent to <paramref name="sink"/>.</summary>
    void Push(int tokenId, Action<ParsedEvent> sink);

    /// <summary>Signals the stream ended; flushes held-back text and emits the terminal Stop.</summary>
    void Finish(Action<ParsedEvent> sink);

    /// <summary>The turn assembled so far; complete once <see cref="Finish"/> returned.</summary>
    ParsedAssistant Result { get; }
}

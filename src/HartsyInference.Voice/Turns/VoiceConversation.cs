using HartsyInference.Engine.Requests;

namespace HartsyInference.Voice.Turns;

/// <summary>The call's message history for the language model: the system message first, then user, assistant and
/// tool turns, trimmed to a token budget by dropping the oldest turns.</summary>
/// <remarks>Trimming never drops the system message or the last user message, and it never leaves the history
/// starting on an assistant or tool message: a tool result whose call was dropped, or a reply to a question that was,
/// confuses a chat template more than the missing context does. Token counts are the model's own
/// (<c>ITextService.CountTokens</c>), counted once per message. Touched only by the turn loop, one turn at a time.</remarks>
internal sealed class VoiceConversation
{
    /// <summary>Tokens a chat template spends on one message's role markers, on top of its content.</summary>
    public const int MessageOverheadTokens = 4;

    private readonly TextMessage _system;
    private readonly List<Entry> _entries = [];
    private int _systemTokens = -1;

    public VoiceConversation(string systemPrompt)
    {
        ArgumentNullException.ThrowIfNull(systemPrompt);
        _system = new TextMessage { Role = TextRole.System, Content = systemPrompt };
    }

    /// <summary>The history after the system message, oldest first.</summary>
    public IReadOnlyList<TextMessage> History => [.. _entries.Select(e => e.Message)];

    /// <summary>Appends what the caller said.</summary>
    public void AddUser(string text) => Add(new TextMessage { Role = TextRole.User, Content = text });

    /// <summary>Appends a reply; <paramref name="toolCalls"/> are the calls it made, each answered by a following
    /// <see cref="AddToolResult"/>.</summary>
    public void AddAssistant(string text, IReadOnlyList<NativeToolCall>? toolCalls = null) =>
        Add(new TextMessage { Role = TextRole.Assistant, Content = text, ToolCalls = toolCalls is { Count: > 0 } ? toolCalls : null });

    /// <summary>Appends the result of <paramref name="call"/>.</summary>
    public void AddToolResult(NativeToolCall call, string result)
    {
        ArgumentNullException.ThrowIfNull(call);
        Add(new TextMessage { Role = TextRole.Tool, Content = result, ToolCallId = call.Id, Name = call.Name });
    }

    /// <summary>Drops the oldest turns until the history fits <paramref name="maxTokens"/>, then returns the request
    /// messages, system message first.</summary>
    public IReadOnlyList<TextMessage> ToRequest(Func<string, int> countTokens, int maxTokens)
    {
        ArgumentNullException.ThrowIfNull(countTokens);
        if (_systemTokens < 0)
        {
            _systemTokens = countTokens(_system.Content) + MessageOverheadTokens;
        }
        long total = _systemTokens;
        int lastUser = -1;
        for (int i = 0; i < _entries.Count; i++)
        {
            total += TokensOf(i, countTokens);
            if (_entries[i].Message.Role == TextRole.User)
            {
                lastUser = i;
            }
        }
        int drop = 0;
        if (total > maxTokens && lastUser > 0)
        {
            while (drop < lastUser && total > maxTokens)
            {
                total -= _entries[drop].Tokens;
                drop++;
            }
            while (drop < lastUser && _entries[drop].Message.Role != TextRole.User)
            {
                drop++;
            }
            _entries.RemoveRange(0, drop);
        }
        TextMessage[] messages = new TextMessage[_entries.Count + 1];
        messages[0] = _system;
        for (int i = 0; i < _entries.Count; i++)
        {
            messages[i + 1] = _entries[i].Message;
        }
        return messages;
    }

    private void Add(TextMessage message) => _entries.Add(new Entry(message, -1));

    private int TokensOf(int index, Func<string, int> countTokens)
    {
        Entry entry = _entries[index];
        if (entry.Tokens >= 0)
        {
            return entry.Tokens;
        }
        int tokens = countTokens(entry.Message.Content) + MessageOverheadTokens;
        if (entry.Message.ToolCalls is { } calls)
        {
            foreach (NativeToolCall call in calls)
            {
                tokens += countTokens(call.Name) + countTokens(call.Arguments);
            }
        }
        _entries[index] = entry with { Tokens = tokens };
        return tokens;
    }

    private readonly record struct Entry(TextMessage Message, int Tokens);
}

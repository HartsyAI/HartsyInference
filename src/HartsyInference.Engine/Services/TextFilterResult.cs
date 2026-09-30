using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>What an <see cref="ITextStreamFilter"/> decided for one delta: <paramref name="ForwardText"/> is streamed as content (empty forwards nothing), <paramref name="ToolCall"/> is emitted as a completed <see cref="TextChunkKind.NativeToolCall"/>, and <paramref name="Stop"/> ends generation on this token: as <see cref="StopReason.ToolCall"/> when a tool call was completed, otherwise as a plain <see cref="StopReason.Stop"/>.</summary>
public readonly record struct TextFilterResult(string ForwardText, NativeToolCall? ToolCall = null, bool Stop = false)
{
    /// <summary>Forwards nothing and keeps generating.</summary>
    public static TextFilterResult Empty => new("");

    /// <summary>Forwards <paramref name="text"/> unchanged.</summary>
    public static TextFilterResult Forward(string text) => new(text);
}

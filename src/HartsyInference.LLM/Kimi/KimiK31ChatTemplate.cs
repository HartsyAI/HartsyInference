using System.Text;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Kimi;

/// <summary>The Kimi K3.1 XTML chat format for the text path, ported from the checkpoint's <c>encoding_k3.py</c> (<c>build_chat_segments</c>). The
/// checkpoint ships no <c>chat_template.jinja</c>, so this is the template the Text path renders with. Control markers are emitted by id; text is
/// encoded as ordinary BPE, one segment at a time, exactly as the reference does, so pre-token boundaries match.</summary>
/// <remarks>Covers user, system and assistant turns with text content, reasoning and names, plus the thinking-effort turn and the generation prompt.
/// Tool schemas, tool calls and results, images, response formats and prefilled turns are refused by name rather than rendered wrong. The
/// engine's numeric reasoning effort is not used: the reference fixes the effort to <c>max</c>.</remarks>
public sealed class KimiK31ChatTemplate : IChatTemplate
{
    private const string ThinkingEffortBody = "`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), "
        + "supported values include `low`, `medium`, `high`, and `max`.\nNow the system is invoked with `thinking_effort=max`.";

    /// <inheritdoc />
    public string Name => "kimi_k31";

    /// <summary>Renders the prompt as text, with each control marker spelled out. Matches the reference's joined segment text.</summary>
    public static string RenderText(IReadOnlyList<ChatMessage> messages, bool thinking = true, bool addGenerationPrompt = true)
    {
        ArgumentNullException.ThrowIfNull(messages);
        List<Piece> pieces = Pieces(messages, thinking, addGenerationPrompt);
        StringBuilder text = new();
        foreach (Piece piece in pieces)
            text.Append(piece.Text);
        return text.ToString();
    }

    /// <inheritdoc />
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking = null)
        => Encode(tokenizer, messages, addGenerationPrompt, enableThinking, tools: null);

    /// <inheritdoc />
    /// <exception cref="NotSupportedException">The conversation offers tools, which this text path does not render.</exception>
    public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking,
        IReadOnlyList<ToolSpec>? tools)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(messages);
        if (tools is { Count: > 0 })
            throw new NotSupportedException("Kimi K3.1 tool schemas are not rendered by the Text path yet.");

        List<int> ids = [];
        foreach (Piece piece in Pieces(messages, enableThinking ?? true, addGenerationPrompt))
        {
            if (piece.Control)
                ids.Add(tokenizer.SpecialId(piece.Text) ?? throw new InvalidOperationException($"The tokenizer has no '{piece.Text}' control token."));
            else
                ids.AddRange(tokenizer.EncodeOrdinary(piece.Text));
        }
        return [.. ids];
    }

    private static List<Piece> Pieces(IReadOnlyList<ChatMessage> messages, bool thinking, bool addGenerationPrompt)
    {
        List<Piece> pieces = [];
        if (thinking)
        {
            OpenTag(pieces, "message", ("role", "system"), ("type", "thinking-effort"));
            Text(pieces, ThinkingEffortBody);
            CloseTag(pieces, "message");
            EndOfMessage(pieces);
        }

        foreach (ChatMessage message in messages)
        {
            RefuseUnsupported(message);
            switch (message.Role)
            {
                case "user":
                case "system":
                    OpenTag(pieces, "message", ("role", message.Role), NameAttribute(message));
                    Content(pieces, message);
                    CloseTag(pieces, "message");
                    EndOfMessage(pieces);
                    break;
                case "assistant":
                    OpenTag(pieces, "message", ("role", "assistant"), NameAttribute(message));
                    AssistantBody(pieces, message, thinking);
                    CloseTag(pieces, "message");
                    EndOfMessage(pieces);
                    break;
                default:
                    throw new NotSupportedException($"Kimi K3.1 chat turns with role '{message.Role}' are not rendered by the Text path yet.");
            }
        }

        if (addGenerationPrompt)
        {
            OpenTag(pieces, "message", ("role", "assistant"));
            OpenTag(pieces, thinking ? "think" : "response");
        }
        return pieces;
    }

    private static void AssistantBody(List<Piece> pieces, ChatMessage message, bool thinking)
    {
        if (thinking)
        {
            OpenTag(pieces, "think");
            if (!string.IsNullOrEmpty(message.ReasoningContent))
                Text(pieces, message.ReasoningContent);
            CloseTag(pieces, "think");
        }
        OpenTag(pieces, "response");
        Content(pieces, message);
        CloseTag(pieces, "response");
    }

    // A message's structured text blocks replace its content when present, as the reference's list-of-parts content does.
    private static void Content(List<Piece> pieces, ChatMessage message)
    {
        if (message.Blocks is { Count: > 0 } blocks)
        {
            foreach (ContentBlock block in blocks)
                Text(pieces, ((TextBlock)block).Text);
            return;
        }
        Text(pieces, message.Content);
    }

    private static (string Key, string Value)? NameAttribute(ChatMessage message) =>
        string.IsNullOrEmpty(message.Name) ? null : ("name", message.Name);

    private static void RefuseUnsupported(ChatMessage message)
    {
        if (message.Tools is { Count: > 0 })
            throw new NotSupportedException("Kimi K3.1 tool schemas are not rendered by the Text path yet.");
        if (message.ToolCalls is { Count: > 0 } || message.ToolCallId is not null)
            throw new NotSupportedException("Kimi K3.1 tool calls and tool results are not rendered by the Text path yet.");
        if (message.ResponseFormatJson is not null)
            throw new NotSupportedException("Kimi K3.1 response formats are not rendered by the Text path yet.");
        if (message.WithoutEos)
            throw new NotSupportedException("Kimi K3.1 prefilled assistant turns are not rendered by the Text path yet.");
        if (message.Task is not null)
            throw new NotSupportedException("Kimi K3.1 internal classification tasks are not rendered by the Text path yet.");
        if (message.Blocks is { Count: > 0 } blocks && blocks.Any(block => block is not TextBlock))
            throw new NotSupportedException("Kimi K3.1 image input is not supported: the Text path renders text blocks only.");
    }

    // Mirrors _open_tag: the tag and each attribute are separate text pieces, then the separator, so BPE boundaries match the reference.
    private static void OpenTag(List<Piece> pieces, string tag, params (string Key, string Value)?[] attributes)
    {
        pieces.Add(Piece.Marker("<|open|>"));
        Text(pieces, tag);
        foreach ((string Key, string Value)? attribute in attributes)
        {
            if (attribute is not { } attr)
                continue;
            Text(pieces, " " + attr.Key);
            Text(pieces, "=\"");
            Text(pieces, EscapeAttribute(attr.Value));
            Text(pieces, "\"");
        }
        pieces.Add(Piece.Marker("<|sep|>"));
    }

    private static void CloseTag(List<Piece> pieces, string tag)
    {
        pieces.Add(Piece.Marker("<|close|>"));
        Text(pieces, tag);
        pieces.Add(Piece.Marker("<|sep|>"));
    }

    private static void EndOfMessage(List<Piece> pieces) => pieces.Add(Piece.Marker("<|end_of_msg|>"));

    // Empty text is dropped, as the reference's _text does.
    private static void Text(List<Piece> pieces, string? text)
    {
        if (!string.IsNullOrEmpty(text))
            pieces.Add(new Piece(text, Control: false));
    }

    private static string EscapeAttribute(string value) =>
        value.Replace("&", "&amp;", StringComparison.Ordinal).Replace("\"", "&quot;", StringComparison.Ordinal);

    private readonly record struct Piece(string Text, bool Control)
    {
        public static Piece Marker(string literal) => new(literal, Control: true);
    }
}

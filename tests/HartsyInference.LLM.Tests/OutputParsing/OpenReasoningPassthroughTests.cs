using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>A Jinja template that prefills <c>&lt;think&gt;</c> (Dory, Nemotron, Qwen3-Thinking) starts the completion inside the think block: the text up
/// to <c>&lt;/think&gt;</c> is reasoning, not the answer, and the close token itself is consumed.</summary>
public sealed class OpenReasoningPassthroughTests
{
    /// <summary>Dory's generation prompt, reduced to its thinking switch.</summary>
    private const string PrefillTemplate =
        "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}"
        + "{%- for m in messages %}{{- '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{%- endfor %}"
        + "{%- if add_generation_prompt %}{%- if enable_thinking %}{{- '<|im_start|>assistant\\n<think>\\n' }}"
        + "{%- else %}{{- '<|im_start|>assistant\\n<think></think>' }}{%- endif %}{%- endif %}";

    /// <summary>Qwen3's generation prompt: the model opens its own think block, or an empty one is prefilled when thinking is off.</summary>
    private const string SelfOpeningTemplate =
        "{%- for m in messages %}{{- '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{%- endfor %}"
        + "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
        + "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}";

    private static readonly IReadOnlyList<ChatMessage> Conversation = [ChatMessage.System("You are Swarmie."), ChatMessage.User("Why is the sky blue?")];

    [Fact]
    public void ReasoningBeforeTheCloseTokenIsReportedApartFromTheAnswer()
    {
        PieceTokenizer tok = new();
        int close = tok.SpecialIdOf(PieceTokenizer.ThinkEnd);
        int[] ids = tok.RandomWithSpecials("The user asks about the sky.</think>\n\nRayleigh scattering.", new Random(3), 3);
        // TextService also asks the passthrough parser to show the reasoning markers as text; the close must still end the block.
        PassthroughOutputParser parser = new(tok, new HashSet<int> { tok.SpecialIdOf(PieceTokenizer.Think), close }, close);
        List<ParsedEvent> events = [];
        foreach (int id in ids) parser.Push(id, events.Add);
        parser.Finish(events.Add);

        Assert.Equal("The user asks about the sky.", parser.Result.Reasoning);
        Assert.Equal("Rayleigh scattering.", parser.Result.Content);
        Assert.Equal("The user asks about the sky.", Joined(events, ParsedEventKind.ReasoningDelta));
        Assert.Equal("Rayleigh scattering.", Joined(events, ParsedEventKind.ContentDelta));
        int lastReasoning = events.FindLastIndex(e => e.Kind == ParsedEventKind.ReasoningDelta);
        Assert.True(lastReasoning < events.FindIndex(e => e.Kind == ParsedEventKind.ContentDelta));
        Assert.Equal("Rayleigh scattering.", TextService.FinalText(parser, callCount: 0, tok.Decode(ids)));
    }

    [Fact]
    public void ReasoningThatNeverClosesKeepsThePartialReasoning()
    {
        PieceTokenizer tok = new();
        int close = tok.SpecialIdOf(PieceTokenizer.ThinkEnd);
        PassthroughOutputParser parser = new(tok, null, close);
        List<ParsedEvent> events = [];
        foreach (int id in tok.SplitBytes("Still thinking about it", 4, 9)) parser.Push(id, events.Add);
        parser.Finish(events.Add);

        Assert.Equal("Still thinking about it", parser.Result.Reasoning);
        Assert.Equal("", parser.Result.Content);
        Assert.True(parser.Result.Completed);
        Assert.DoesNotContain(events, e => e.Kind == ParsedEventKind.ContentDelta);
    }

    [Fact]
    public async Task BudgetRunningOutMidReasoningStreamsTheReasoningThenALengthStop()
    {
        // Thinking on, as TextService wires it: the parser starts in reasoning, and a decode that stops on the token budget ends as Length.
        PieceTokenizer tok = new();
        int close = tok.SpecialIdOf(PieceTokenizer.ThinkEnd);
        int[] ids = tok.SplitBytes("The user asks why the sky is blue. Rayleigh", 9, 20);
        List<(TextChunkKind Kind, string? Text, StopReason? Stop)> got = await Stream(tok, ids, close, StopReason.Length);

        Assert.Equal("The user asks why the sky is blue. Rayleigh", string.Concat(got.Where(c => c.Kind == TextChunkKind.Reasoning).Select(c => c.Text)));
        Assert.DoesNotContain(got, c => c.Kind == TextChunkKind.Chunk);
        Assert.Equal([(TextChunkKind.Result, "", null), (TextChunkKind.StopReason, null, StopReason.Length)], got.TakeLast(2));
    }

    [Fact]
    public async Task ThinkingOffAnswersDirectly()
    {
        // Thinking off: the template prefills an empty think block, TextService passes no close token, and every token is the answer.
        LiteralControlTokenizer literal = new();
        GenerationRequest genRequest = new() { Messages = Conversation };
        TextRequest notThinking = new() { Messages = [], EnableThinking = false };
        int? close = TextService.OpenReasoningCloseId(new JinjaChatTemplate(PrefillTemplate), literal, genRequest, notThinking, rawCompletion: false);
        Assert.Null(close);

        PieceTokenizer tok = new();
        int[] ids = tok.SplitBytes("Rayleigh scattering.", 4, 11);
        List<(TextChunkKind Kind, string? Text, StopReason? Stop)> got = await Stream(tok, ids, close, StopReason.Stop);

        Assert.DoesNotContain(got, c => c.Kind == TextChunkKind.Reasoning);
        Assert.Equal("Rayleigh scattering.", string.Concat(got.Where(c => c.Kind == TextChunkKind.Chunk).Select(c => c.Text)));
        Assert.Equal([(TextChunkKind.Result, "Rayleigh scattering.", null), (TextChunkKind.StopReason, null, StopReason.Stop)], got.TakeLast(2));
    }

    [Fact]
    public void TemplateThatPrefillsThinkOpensReasoningUnlessThinkingIsOff()
    {
        LiteralControlTokenizer tok = new();
        JinjaChatTemplate template = new(PrefillTemplate);
        Assert.True(template.ResolveParserState(tok, Conversation, enableThinking: null, tools: null).ReasoningOpen);
        Assert.True(template.ResolveParserState(tok, Conversation, enableThinking: true, tools: null).ReasoningOpen);
        Assert.False(template.ResolveParserState(tok, Conversation, enableThinking: false, tools: null).ReasoningOpen);
    }

    [Fact]
    public void TemplateWhoseModelOpensItsOwnThinkBlockStartsInContent()
    {
        LiteralControlTokenizer tok = new();
        JinjaChatTemplate template = new(SelfOpeningTemplate);
        // Message content keeps reasoning markers literal, so an unmatched one in the user turn must not count.
        IReadOnlyList<ChatMessage> quoting = [ChatMessage.User("What does <think> do?")];
        Assert.False(template.ResolveParserState(tok, quoting, enableThinking: null, tools: null).ReasoningOpen);
        Assert.False(template.ResolveParserState(tok, quoting, enableThinking: false, tools: null).ReasoningOpen);
    }

    [Fact]
    public void TokenizerWithoutThinkControlTokensStartsInContent()
    {
        JinjaChatTemplate template = new(PrefillTemplate);
        Assert.False(template.ResolveParserState(new NoThinkTokenizer(), Conversation, enableThinking: null, tools: null).ReasoningOpen);
    }

    [Fact]
    public void TextServiceHandsThePassthroughParserTheCloseTokenOnlyForAnOpenPrompt()
    {
        LiteralControlTokenizer tok = new();
        JinjaChatTemplate template = new(PrefillTemplate);
        GenerationRequest genRequest = new() { Messages = Conversation };
        TextRequest thinking = new() { Messages = [] };
        TextRequest notThinking = new() { Messages = [], EnableThinking = false };
        int close = LiteralControlTokenizer.IdOf(LiteralControlTokenizer.ThinkClose);

        Assert.Equal(close, TextService.OpenReasoningCloseId(template, tok, genRequest, thinking, rawCompletion: false));
        Assert.Null(TextService.OpenReasoningCloseId(template, tok, genRequest, notThinking, rawCompletion: false));
        Assert.Null(TextService.OpenReasoningCloseId(template, tok, genRequest, thinking, rawCompletion: true));
        Assert.Null(TextService.OpenReasoningCloseId(new ChatMlTemplate(), tok, genRequest, thinking, rawCompletion: false));
        Assert.Null(TextService.OpenReasoningCloseId(template, tok, new GenerationRequest { Prompt = "hi" }, thinking, rawCompletion: false));
    }

    /// <summary>Runs <paramref name="ids"/> through the wiring <c>TextService.StreamAsync</c> uses (parser, translator, pump), ending with the
    /// Result chunk (the parser's content) and <paramref name="stop"/>.</summary>
    private static async Task<List<(TextChunkKind Kind, string? Text, StopReason? Stop)>> Stream(PieceTokenizer tok, int[] ids, int? close, StopReason stop)
    {
        Task<IReadOnlyList<TextChunk>> Produce(Action<TextChunk> sink, CancellationToken ct)
        {
            PassthroughOutputParser parser = new(tok, null, close);
            ParsedEventTranslator translator = new(sink, 1);
            foreach (int id in ids) parser.Push(id, translator.Handle);
            parser.Finish(translator.Handle);
            return Task.FromResult<IReadOnlyList<TextChunk>>(
            [
                new TextChunk { Kind = TextChunkKind.Result, Text = TextService.FinalText(parser, callCount: 0, tok.Decode(ids)) },
                new TextChunk { Kind = TextChunkKind.StopReason, Stop = stop },
            ]);
        }
        List<(TextChunkKind Kind, string? Text, StopReason? Stop)> got = [];
        await foreach (TextChunk c in TextStreamPump.Run(Produce)) got.Add((c.Kind, c.Text, c.Stop));
        return got;
    }

    private static string Joined(List<ParsedEvent> events, ParsedEventKind kind) => string.Concat(events.Where(e => e.Kind == kind).Select(e => e.Text));

    /// <summary>The literal tokenizer's vocabulary without the reasoning markers as control tokens.</summary>
    private sealed class NoThinkTokenizer : HartsyInference.ModelAssets.Tokenizers.ILlmTokenizer
    {
        private readonly LiteralControlTokenizer _inner = new();

        public IReadOnlyList<string> SpecialLiterals => [.. _inner.SpecialLiterals.Where(l => !SpecialLiteralEscaper.ReasoningMarkers.Contains(l))];

        public int[] Encode(string text, bool addSpecial) => _inner.Encode(text, addSpecial);

        public int[] EncodeOrdinary(string text) => _inner.EncodeOrdinary(text);

        public string Decode(IReadOnlyList<int> ids) => _inner.Decode(ids);

        public int? SpecialId(string token) => SpecialLiteralEscaper.ReasoningMarkers.Contains(token) ? null : _inner.SpecialId(token);

        public int? BosId => _inner.BosId;

        public int? EosId => _inner.EosId;

        public IReadOnlyList<int> StopIds => _inner.StopIds;

        public string? BosToken => _inner.BosToken;

        public string? EosToken => _inner.EosToken;
    }
}

using System.Diagnostics;
using System.Text;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Models.MossVl;
using HartsyInference.LLM.Multimodal.Streaming;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Tests;

/// <summary>The live session loop with a scripted decoder and encoder (no weights): what it splices into the stream, when it idles,
/// and what it reports.</summary>
public sealed class RealtimeVlmSessionTests
{
    private const int Silence = 1;
    private static readonly MossVlRealtimeProtocol Protocol = new(imagePlaceholderId: 8, silenceTokenId: Silence);

    [Fact]
    public async Task PromptAndFramesAreSplicedAfterTheLastTokenInTheReferenceOrder()
    {
        FakeTokenizer tokenizer = new();
        ScriptedDecoder decoder = new([Silence, tokenizer.Char('O'), tokenizer.Char('K'), Silence]);
        await using RealtimeVlmSession session = new(decoder, new FakeEncoder(), Protocol, tokenizer);
        // Queued before the loop starts, so the first cycle drains them together.
        session.PushFrame(Frame(), 0.25);
        session.PushFrame(Frame(), 1.0);
        session.PushPrompt("hi");
        session.Start(systemPrompt: "sys", initialPrompt: "watch");

        List<RealtimeVlmEvent> events = await ReadUntil(session, e => e.Count(x => x.Kind == RealtimeVlmEventKind.Silence) == 3);

        Assert.Equal(
            [RealtimeVlmEventKind.Silence, RealtimeVlmEventKind.RoundStart, RealtimeVlmEventKind.Silence,
             RealtimeVlmEventKind.Text, RealtimeVlmEventKind.Text, RealtimeVlmEventKind.Silence],
            events.Select(e => e.Kind).ToArray());
        Assert.Equal("hi", events[1].Text);
        Assert.Equal("<|silence|>", events[0].Text);
        Assert.Equal("OK", events[3].Text + events[4].Text);

        Assert.Equal(tokenizer.Encode(Protocol.InitialPrompt("sys", "watch"), addSpecial: true), decoder.Calls[0].Ids);
        // The cycle appends the token just generated, the prompt, the spliced silence, then the frames in time order.
        string segment = Protocol.PromptSegment("hi") + "<|silence|>" + Protocol.FrameSegment(0.25) + Protocol.FrameSegment(1.0);
        Assert.Equal([Silence, .. tokenizer.Encode(segment, addSpecial: true)], decoder.Calls[1].Ids);
        Assert.Equal([0.25, 1.0], decoder.Calls[1].Timestamps);
        // Decode steps carry only the previous token.
        Assert.Equal([tokenizer.Char('O')], decoder.Calls[2].Ids);
        Assert.Empty(decoder.Calls[2].Timestamps);
        Assert.Equal(2, session.Stats.FramesProcessed);
    }

    [Fact]
    public async Task SilenceIdlesUntilInputArrives()
    {
        FakeTokenizer tokenizer = new();
        ScriptedDecoder decoder = new([Silence, tokenizer.Char('A'), Silence]);
        await using RealtimeVlmSession session = new(decoder, new FakeEncoder(), Protocol, tokenizer);
        session.Start(null, "");
        await ReadUntil(session, e => e.Count == 1);
        await Task.Delay(150);
        Assert.Single(decoder.Calls);   // idle: no forward without input

        session.PushPrompt("now?");
        List<RealtimeVlmEvent> events = await ReadUntil(session, e => e.Any(x => x.Kind == RealtimeVlmEventKind.Text));
        Assert.Contains(events, e => e.Kind == RealtimeVlmEventKind.RoundStart && e.Text == "now?");
        // A prompt with no frames splices no silence.
        Assert.Equal([Silence, .. tokenizer.Encode(Protocol.PromptSegment("now?"), addSpecial: true)], decoder.Calls[1].Ids);
    }

    [Fact]
    public async Task ASplitCharacterIsEmittedOnceWhole()
    {
        FakeTokenizer tokenizer = new();
        byte[] e = Encoding.UTF8.GetBytes("é");
        ScriptedDecoder decoder = new([FakeTokenizer.Byte(e[0]), FakeTokenizer.Byte(e[1]), Silence]);
        await using RealtimeVlmSession session = new(decoder, new FakeEncoder(), Protocol, tokenizer);
        session.Start(null, "");
        List<RealtimeVlmEvent> events = await ReadUntil(session, x => x.Any(v => v.Kind == RealtimeVlmEventKind.Silence));
        RealtimeVlmEvent text = Assert.Single(events, v => v.Kind == RealtimeVlmEventKind.Text);
        Assert.Equal("é", text.Text);
    }

    [Fact]
    public async Task ControlTokensAreReportedApartFromTheReplyText()
    {
        FakeTokenizer tokenizer = new();
        int response = tokenizer.SpecialId("<|response|>")!.Value;
        ScriptedDecoder decoder = new([response, tokenizer.Char('A'), Silence]);
        await using RealtimeVlmSession session = new(decoder, new FakeEncoder(), Protocol, tokenizer);
        session.Start(null, "");
        List<RealtimeVlmEvent> events = await ReadUntil(session, x => x.Any(v => v.Kind == RealtimeVlmEventKind.Silence));
        Assert.Equal([RealtimeVlmEventKind.Control, RealtimeVlmEventKind.Text, RealtimeVlmEventKind.Silence], events.Select(e => e.Kind).ToArray());
        Assert.Equal(["<|response|>", "A", "<|silence|>"], events.Select(e => e.Text).ToArray());
    }

    [Fact]
    public async Task AFullFrameQueueDropsTheOldestAndReportsIt()
    {
        FakeTokenizer tokenizer = new();
        ScriptedDecoder decoder = new([Silence, Silence]);
        await using RealtimeVlmSession session = new(decoder, new FakeEncoder(), Protocol, tokenizer,
            new RealtimeVlmOptions { FrameQueueSize = 2 });
        Assert.False(session.PushFrame(Frame(), 0));
        Assert.False(session.PushFrame(Frame(), 1));
        Assert.True(session.PushFrame(Frame(), 2));
        session.Start(null, "");

        List<RealtimeVlmEvent> events = await ReadUntil(session, x => x.Count(v => v.Kind == RealtimeVlmEventKind.Silence) == 2);
        RealtimeVlmEvent dropped = Assert.Single(events, v => v.Kind == RealtimeVlmEventKind.FramesDropped);
        Assert.Equal(1, dropped.Count);
        Assert.Equal([1.0, 2.0], decoder.Calls[1].Timestamps);
        Assert.Equal(1, session.Stats.FramesDropped);
    }

    [Fact]
    public async Task FrameTimestampsMustNotGoBackwards()
    {
        await using RealtimeVlmSession session = new(new ScriptedDecoder([]), new FakeEncoder(), Protocol, new FakeTokenizer());
        session.PushFrame(Frame(), 2.0);
        Assert.Throws<ArgumentException>(() => session.PushFrame(Frame(), 1.0));
        session.PushFrame(Frame(), 2.0);   // equal is fine
    }

    [Fact]
    public async Task ADecoderFailureEndsTheStreamWithAFault()
    {
        FakeTokenizer tokenizer = new();
        ScriptedDecoder decoder = new([tokenizer.Char('x')]) { FailAtCall = 1 };
        await using RealtimeVlmSession session = new(decoder, new FakeEncoder(), Protocol, tokenizer);
        session.Start(null, "");
        List<RealtimeVlmEvent> events = [];
        await foreach (RealtimeVlmEvent e in session.ReadAsync().WithTimeout(TimeSpan.FromSeconds(10))) events.Add(e);
        Assert.Equal(RealtimeVlmEventKind.Faulted, events[^1].Kind);
        Assert.Contains("scripted failure", events[^1].Text);
    }

    [Fact]
    public async Task InputAfterAFaultIsRefusedInsteadOfAcceptedSilently()
    {
        FakeTokenizer tokenizer = new();
        // The second forward throws, as MossVlDecodeSession does when a vision or text cap is reached.
        ScriptedDecoder decoder = new([Silence, tokenizer.Char('x')]) { FailAtCall = 1 };
        await using RealtimeVlmSession session = new(decoder, new FakeEncoder(), Protocol, tokenizer);
        session.Start(null, "");
        await ReadUntil(session, x => x.Count == 1);
        session.PushPrompt("go on");   // wakes the idle loop, whose forward then fails
        List<RealtimeVlmEvent> rest = [];
        await foreach (RealtimeVlmEvent e in session.ReadAsync().WithTimeout(TimeSpan.FromSeconds(10))) rest.Add(e);
        Assert.Equal(RealtimeVlmEventKind.Faulted, rest[^1].Kind);

        Assert.Throws<ObjectDisposedException>(() => session.PushPrompt("late"));
        Assert.Throws<ObjectDisposedException>(() => session.PushFrame(Frame(), 5.0));
    }

    [Fact]
    public async Task DisposingASessionThatNeverStartedEndsItsStream()
    {
        RealtimeVlmSession session = new(new ScriptedDecoder([]), new FakeEncoder(), Protocol, new FakeTokenizer());
        Task<List<RealtimeVlmEvent>> reader = Task.Run(async () =>
        {
            List<RealtimeVlmEvent> events = [];
            await foreach (RealtimeVlmEvent e in session.ReadAsync().WithTimeout(TimeSpan.FromSeconds(10))) events.Add(e);
            return events;
        });
        await session.DisposeAsync();
        Assert.Empty(await reader);
        Assert.Throws<ObjectDisposedException>(() => session.Start(null, ""));
    }

    [Fact]
    public async Task PacingCapsTheTokenRate()
    {
        FakeTokenizer tokenizer = new();
        int a = tokenizer.Char('a');
        ScriptedDecoder decoder = new([a, a, a, a, a, a, Silence]);
        await using RealtimeVlmSession session = new(decoder, new FakeEncoder(), Protocol, tokenizer,
            new RealtimeVlmOptions { MaxTokensPerSecond = 20 });
        Stopwatch clock = Stopwatch.StartNew();
        session.Start(null, "");
        await ReadUntil(session, x => x.Any(v => v.Kind == RealtimeVlmEventKind.Silence));
        // Six paced tokens at 20/s take at least ~250 ms after the first.
        Assert.True(clock.ElapsedMilliseconds >= 230, $"took {clock.ElapsedMilliseconds} ms");
    }

    [Fact]
    public async Task DisposeStopsTheLoopAndReleasesTheDecoder()
    {
        ScriptedDecoder decoder = new([Silence]);
        RealtimeVlmSession session = new(decoder, new FakeEncoder(), Protocol, new FakeTokenizer());
        session.Start(null, "");
        await ReadUntil(session, x => x.Count == 1);
        await session.DisposeAsync();
        Assert.True(decoder.Disposed);
        Assert.Throws<ObjectDisposedException>(() => session.PushPrompt("late"));
    }

    private static RawRgbFrame Frame() => new(new byte[2 * 2 * 3], 2, 2);

    private static async Task<List<RealtimeVlmEvent>> ReadUntil(RealtimeVlmSession session, Func<List<RealtimeVlmEvent>, bool> done)
    {
        List<RealtimeVlmEvent> events = [];
        using CancellationTokenSource timeout = new(TimeSpan.FromSeconds(10));
        await foreach (RealtimeVlmEvent e in session.ReadAsync(timeout.Token))
        {
            events.Add(e);
            if (done(events)) return events;
        }
        throw new Xunit.Sdk.XunitException($"The stream ended before the expected events: {string.Join(", ", events.Select(e => e.Kind))}");
    }

    /// <summary>Returns scripted tokens, one per forward, then silence; records each call.</summary>
    private sealed class ScriptedDecoder(IReadOnlyList<int> script) : IStreamingVlmDecoder
    {
        private int _next;

        public List<(int[] Ids, double[] Timestamps)> Calls { get; } = [];

        public int FailAtCall { get; init; } = -1;

        public bool Disposed { get; private set; }

        public int TextTokens { get; private set; }

        public int VisionTokens { get; private set; }

        public float[] ForwardLogits(ReadOnlySpan<int> ids, IReadOnlyList<VisionFrameFeatures> frames, CancellationToken cancel) =>
            throw new NotSupportedException();

        public int ForwardSample(ReadOnlySpan<int> ids, IReadOnlyList<VisionFrameFeatures> frames, SamplerChain sampler, List<int> history,
            CancellationToken cancel)
        {
            lock (Calls)
            {
                if (Calls.Count == FailAtCall) throw new InvalidOperationException("scripted failure");
                Calls.Add((ids.ToArray(), frames.Select(f => f.TimestampSeconds).ToArray()));
                TextTokens += ids.Length;
                VisionTokens += frames.Sum(f => f.MergedHeight * f.MergedWidth + 1);
                return _next < script.Count ? script[_next++] : Silence;
            }
        }

        public void Dispose() => Disposed = true;
    }

    /// <summary>One 1x1 feature per frame, carrying the frame's time.</summary>
    private sealed class FakeEncoder : IStreamingFrameEncoder
    {
        public IReadOnlyList<VisionFrameFeatures> Encode(IReadOnlyList<TimedRgbFrame> frames, CancellationToken cancel) =>
            [.. frames.Select(f => new VisionFrameFeatures(new Tensor(new TensorShape(1, 4), DType.F32), 1, 1, f.TimestampSeconds))];
    }

    /// <summary>Control literals map to small ids, raw bytes to 500 + byte, and every other character to 1000 + its code.</summary>
    private sealed class FakeTokenizer : ILlmTokenizer
    {
        private static readonly string[] Specials =
            ["<|silence|>", "<|im_end|>", "<|im_start|>", "<|vision_start|>", "<|vision_end|>", "<|time_start|>", "<|time_end|>", "<|image_pad|>",
             "<|response|>"];

        public int Char(char c) => 1000 + c;

        public static int Byte(byte b) => 500 + b;

        public int[] Encode(string text, bool addSpecial)
        {
            List<int> ids = [];
            for (int i = 0; i < text.Length;)
            {
                int special = addSpecial ? Array.FindIndex(Specials, s => string.CompareOrdinal(text, i, s, 0, s.Length) == 0) : -1;
                if (special >= 0)
                {
                    ids.Add(special + 1);
                    i += Specials[special].Length;
                }
                else
                {
                    ids.Add(Char(text[i++]));
                }
            }
            return [.. ids];
        }

        public int[] EncodeOrdinary(string text) => Encode(text, addSpecial: false);

        public string Decode(IReadOnlyList<int> ids) => Encoding.UTF8.GetString(ids.SelectMany(id => TokenBytes(id, false)!).ToArray());

        public byte[]? TokenBytes(int id, bool includeSpecial) => id switch
        {
            >= 1 and <= 9 => includeSpecial ? Encoding.UTF8.GetBytes(Specials[id - 1]) : [],
            >= 500 and < 756 => [(byte)(id - 500)],
            >= 1000 => Encoding.UTF8.GetBytes(((char)(id - 1000)).ToString()),
            _ => [],
        };

        public int? SpecialId(string token) => Array.IndexOf(Specials, token) is int i and >= 0 ? i + 1 : null;

        public int? BosId => null;

        public int? EosId => 2;

        public IReadOnlyList<int> StopIds => [2];

        public string? BosToken => null;

        public string? EosToken => "<|im_end|>";
    }
}

internal static class AsyncEnumerableTimeout
{
    /// <summary>Ends the enumeration with a timeout failure instead of hanging a test.</summary>
    public static async IAsyncEnumerable<T> WithTimeout<T>(this IAsyncEnumerable<T> source, TimeSpan timeout)
    {
        using CancellationTokenSource cts = new(timeout);
        await foreach (T item in source.WithCancellation(cts.Token)) yield return item;
    }
}

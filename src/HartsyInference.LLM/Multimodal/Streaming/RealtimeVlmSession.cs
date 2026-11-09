using System.Diagnostics;
using System.Threading.Channels;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Logging;
using HartsyInference.LLM.OutputParsing;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Multimodal.Streaming;

/// <summary>What a live vision-language session reports.</summary>
public enum RealtimeVlmEventKind
{
    /// <summary>A user prompt was taken into the stream; the model's answer to it follows.</summary>
    RoundStart,

    /// <summary>Decoded reply text, without control tokens.</summary>
    Text,

    /// <summary>The model chose to stay silent (its silence token), or the protocol spliced one in before a cycle's frames.</summary>
    Silence,

    /// <summary>The model emitted another control token (MOSS-VL: <c>&lt;|response|&gt;</c>); <see cref="RealtimeVlmEvent.Text"/> is its
    /// literal. Applications decide what it means; it is not part of the reply text.</summary>
    Control,

    /// <summary>Frames were dropped because the frame queue was full; <see cref="RealtimeVlmEvent.Count"/> says how many.</summary>
    FramesDropped,

    /// <summary>The loop failed; <see cref="RealtimeVlmEvent.Text"/> holds the reason. No event follows.</summary>
    Faulted,
}

/// <summary>One event from a <see cref="RealtimeVlmSession"/>.</summary>
/// <param name="Text">The reply text, a control token's literal (silence included), the prompt for
/// <see cref="RealtimeVlmEventKind.RoundStart"/>, or the error for <see cref="RealtimeVlmEventKind.Faulted"/>.</param>
/// <param name="Count">How many frames were dropped, for <see cref="RealtimeVlmEventKind.FramesDropped"/>.</param>
public readonly record struct RealtimeVlmEvent(RealtimeVlmEventKind Kind, string Text = "", int Count = 0);

/// <summary>Settings of a <see cref="RealtimeVlmSession"/>.</summary>
public sealed record RealtimeVlmOptions
{
    /// <summary>Frames waiting for the next cycle; when full, the oldest is dropped (the reference session's default is 256).</summary>
    public int FrameQueueSize { get; init; } = 256;

    /// <summary>The most tokens decoded per second. The reference's default (86,400) is effectively unpaced.</summary>
    public double MaxTokensPerSecond { get; init; } = 86_400;

    /// <summary>How often an idle session (its last token was silence) checks for new input.</summary>
    public TimeSpan IdlePoll { get; init; } = TimeSpan.FromMilliseconds(20);

    /// <summary>The most frames one cycle takes from the queue; 0 takes them all. The rest wait for the next cycle.</summary>
    public int MaxFramesPerCycle { get; init; }

    /// <summary>Token selection; null is greedy, as the reference loop runs with the checkpoint's generation config (no sampling).</summary>
    public SamplingOptions? Sampling { get; init; }
}

/// <summary>Counters of a running <see cref="RealtimeVlmSession"/>.</summary>
public readonly record struct RealtimeVlmStats(int TextTokens, int VisionTokens, long GeneratedTokens, long FramesProcessed, long FramesDropped);

/// <summary>The decoder side of a live session: a model state that segments of text and frames are appended to, one forward each.</summary>
public interface IStreamingVlmDecoder : IDisposable
{
    /// <summary>Text tokens the state holds.</summary>
    int TextTokens { get; }

    /// <summary>Vision tokens the state holds.</summary>
    int VisionTokens { get; }

    /// <summary>Appends <paramref name="ids"/> with the <paramref name="frames"/> its placeholders reference, in order, and returns the
    /// logits after the last token. A cancelled call appends nothing.</summary>
    float[] ForwardLogits(ReadOnlySpan<int> ids, IReadOnlyList<VisionFrameFeatures> frames, CancellationToken cancel);

    /// <summary>As <see cref="ForwardLogits"/>, selecting the next token with <paramref name="sampler"/> without copying the logits.</summary>
    int ForwardSample(ReadOnlySpan<int> ids, IReadOnlyList<VisionFrameFeatures> frames, SamplerChain sampler, List<int> history,
        CancellationToken cancel);
}

/// <summary>Turns the frames of one cycle into vision features. The frames are one batch (a model may share a pixel budget across
/// them); the caller owns the returned tensors.</summary>
public interface IStreamingFrameEncoder
{
    /// <summary>Encodes <paramref name="frames"/>, in the order given.</summary>
    IReadOnlyList<VisionFrameFeatures> Encode(IReadOnlyList<TimedRgbFrame> frames, CancellationToken cancel);
}

/// <summary>A live vision-language stream: frames and prompts are pushed from any thread, one loop thread splices them into the model's
/// running context between tokens, and the reply streams out as events. The protocol supplies the model's text layouts, so any
/// streaming VLM plugs in through <see cref="IRealtimeVlmProtocol"/>, <see cref="IStreamingVlmDecoder"/> and <see cref="IStreamingFrameEncoder"/>.</summary>
/// <remarks>
/// <para>The loop follows MOSS-VL's <c>_real_time_sample</c>: after each token it drains the queues; when the token was the silence
/// token and nothing is pending it idles, polling every <see cref="RealtimeVlmOptions.IdlePoll"/>. A cycle appends the token just
/// generated, then each prompt's segment, the protocol's text after prompts when frames also arrived, then each frame's segment
/// (frames sorted by time), and samples the next token after them. A prompt never truncates the context: the stream only grows.</para>
/// <para>The session owns <paramref name="decoder"/> and disposes it. Dispose the session to stop the loop.</para>
/// </remarks>
public sealed class RealtimeVlmSession : IAsyncDisposable
{
    private readonly IStreamingVlmDecoder _decoder;
    private readonly IStreamingFrameEncoder _encoder;
    private readonly IRealtimeVlmProtocol _protocol;
    private readonly ILlmTokenizer _tokenizer;
    private readonly RealtimeVlmOptions _options;
    private readonly Channel<TimedRgbFrame> _frames;
    private readonly Channel<string> _prompts = Channel.CreateUnbounded<string>(new UnboundedChannelOptions { SingleReader = true });
    private readonly Channel<RealtimeVlmEvent> _events = Channel.CreateUnbounded<RealtimeVlmEvent>(new UnboundedChannelOptions { SingleReader = false, SingleWriter = true });
    private readonly CancellationTokenSource _stop = new();
    private readonly object _pushLock = new();
    private Thread? _thread;
    private double _lastTimestamp = double.NegativeInfinity;
    private long _generated, _framesProcessed, _framesDropped;
    private int _disposed;

    public RealtimeVlmSession(IStreamingVlmDecoder decoder, IStreamingFrameEncoder encoder, IRealtimeVlmProtocol protocol,
        ILlmTokenizer tokenizer, RealtimeVlmOptions? options = null)
    {
        _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _protocol = protocol ?? throw new ArgumentNullException(nameof(protocol));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _options = options ?? new RealtimeVlmOptions();
        if (_options.FrameQueueSize < 1) throw new ArgumentOutOfRangeException(nameof(options), "FrameQueueSize must be at least 1.");
        if (_options.MaxTokensPerSecond <= 0) throw new ArgumentOutOfRangeException(nameof(options), "MaxTokensPerSecond must be positive.");
        _frames = Channel.CreateBounded<TimedRgbFrame>(new BoundedChannelOptions(_options.FrameQueueSize)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
        }, _ => Interlocked.Increment(ref _framesDropped));
    }

    /// <summary>The session's counters.</summary>
    public RealtimeVlmStats Stats => new(_decoder.TextTokens, _decoder.VisionTokens, Interlocked.Read(ref _generated),
        Interlocked.Read(ref _framesProcessed), Interlocked.Read(ref _framesDropped));

    /// <summary>Starts the loop: the chat opens with <paramref name="systemPrompt"/> (the protocol's default when null) and
    /// <paramref name="initialPrompt"/>.</summary>
    /// <exception cref="InvalidOperationException">The session already started.</exception>
    public void Start(string? systemPrompt, string initialPrompt)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        string opening = _protocol.InitialPrompt(systemPrompt ?? _protocol.DefaultSystemPrompt, initialPrompt ?? string.Empty);
        Thread thread = new(() => Run(opening)) { IsBackground = true, Name = "realtime-vlm" };
        if (Interlocked.CompareExchange(ref _thread, thread, null) is not null)
            throw new InvalidOperationException("The session is already running.");
        thread.Start();
    }

    /// <summary>Queues a frame captured at <paramref name="timestampSeconds"/> in the stream. Returns true when the queue was full and
    /// the oldest waiting frame was dropped to make room.</summary>
    /// <exception cref="ArgumentException">The timestamp is earlier than the last pushed frame's.</exception>
    public bool PushFrame(RawRgbFrame frame, double timestampSeconds)
    {
        if (frame.Rgb.Length != frame.Width * frame.Height * 3)
            throw new ArgumentException($"A {frame.Width}x{frame.Height} RGB frame needs {frame.Width * frame.Height * 3} bytes, got {frame.Rgb.Length}.", nameof(frame));
        if (!double.IsFinite(timestampSeconds)) throw new ArgumentOutOfRangeException(nameof(timestampSeconds), "A frame timestamp must be finite.");
        lock (_pushLock)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            if (timestampSeconds < _lastTimestamp)
                throw new ArgumentException($"Frame timestamps must not go backwards: {timestampSeconds} after {_lastTimestamp}.", nameof(timestampSeconds));
            _lastTimestamp = timestampSeconds;
            long before = Interlocked.Read(ref _framesDropped);
            _frames.Writer.TryWrite(new TimedRgbFrame(frame, timestampSeconds));
            return Interlocked.Read(ref _framesDropped) != before;
        }
    }

    /// <summary>Queues a user prompt; it is spliced in after the next token.</summary>
    public void PushPrompt(string prompt)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        _prompts.Writer.TryWrite(prompt);
    }

    /// <summary>The session's events, in order, until it stops or faults.</summary>
    public IAsyncEnumerable<RealtimeVlmEvent> ReadAsync(CancellationToken cancel = default) => _events.Reader.ReadAllAsync(cancel);

    /// <summary>Stops the loop, waits for it to leave the model, and releases the decoder.</summary>
    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _stop.Cancel();
        _frames.Writer.TryComplete();
        _prompts.Writer.TryComplete();
        if (Volatile.Read(ref _thread) is { } thread)
            await Task.Run(thread.Join).ConfigureAwait(false);
        _events.Writer.TryComplete();
        _decoder.Dispose();
        _stop.Dispose();
    }

    private void Run(string opening)
    {
        CancellationToken stop = _stop.Token;
        try
        {
            SamplerChain sampler = SamplerChain.FromOptions(_options.Sampling ?? new SamplingOptions { Greedy = true }, _tokenizer);
            List<int> history = [];
            IncrementalDetokenizer text = new(_tokenizer, includeSpecial: false);
            int token = Sample(_tokenizer.Encode(opening, addSpecial: true), [], sampler, history, stop);
            Stopwatch clock = Stopwatch.StartNew();
            List<TimedRgbFrame> frames = [];
            List<string> prompts = [];
            long droppedReported = 0;
            while (!stop.IsCancellationRequested)
            {
                Interlocked.Increment(ref _generated);
                history.Add(token);
                Emit(text, token);

                // Pace to at most MaxTokensPerSecond, as the reference does after each token.
                TimeSpan budget = TimeSpan.FromSeconds(1.0 / _options.MaxTokensPerSecond) - clock.Elapsed;
                if (budget > TimeSpan.Zero) stop.WaitHandle.WaitOne(budget);
                clock.Restart();

                bool idle = token == _protocol.SilenceTokenId;
                Drain(frames, prompts);
                while (idle && frames.Count == 0 && prompts.Count == 0 && !stop.IsCancellationRequested)
                {
                    stop.WaitHandle.WaitOne(_options.IdlePoll);
                    Drain(frames, prompts);
                }
                if (stop.IsCancellationRequested) break;
                // Frames can be dropped at any push; report what was dropped since the last report.
                long droppedTotal = Interlocked.Read(ref _framesDropped);
                if (droppedTotal > droppedReported)
                {
                    _events.Writer.TryWrite(new RealtimeVlmEvent(RealtimeVlmEventKind.FramesDropped, Count: (int)(droppedTotal - droppedReported)));
                    droppedReported = droppedTotal;
                }

                string segment = BuildSegment(frames, prompts, ref text);
                int[] ids = [token, .. segment.Length > 0 ? _tokenizer.Encode(segment, addSpecial: true) : []];
                IReadOnlyList<VisionFrameFeatures> features = frames.Count > 0 ? _encoder.Encode(frames, stop) : [];
                try
                {
                    token = Sample(ids, features, sampler, history, stop);
                }
                finally
                {
                    foreach (VisionFrameFeatures f in features) f.MergedTokens.Dispose();
                }
                Interlocked.Add(ref _framesProcessed, frames.Count);
                frames.Clear();
                prompts.Clear();
            }
        }
        catch (OperationCanceledException) when (stop.IsCancellationRequested)
        {
            // Disposed: the loop ends quietly.
        }
        catch (Exception ex)
        {
            Logs.Warning($"[RealtimeVlm] The session stopped: {ex.Message}");
            _events.Writer.TryWrite(new RealtimeVlmEvent(RealtimeVlmEventKind.Faulted, ex.Message));
        }
        finally
        {
            _events.Writer.TryComplete();
        }
    }

    /// <summary>Selects the token after <paramref name="ids"/>, dropping the protocol's suppressed tokens first when it has any.</summary>
    private int Sample(int[] ids, IReadOnlyList<VisionFrameFeatures> frames, SamplerChain sampler, List<int> history, CancellationToken stop)
    {
        IReadOnlyList<int> suppressed = _protocol.SuppressedTokenIds;
        if (suppressed.Count == 0)
            return _decoder.ForwardSample(ids, frames, sampler, history, stop);
        float[] logits = _decoder.ForwardLogits(ids, frames, stop);
        foreach (int id in suppressed)
            if ((uint)id < (uint)logits.Length) logits[id] = float.NegativeInfinity;
        return sampler.Next(logits, history);
    }

    /// <summary>Streams <paramref name="token"/>: reply text once its characters are whole, the silence token as a silence, and any other
    /// control token as a control event with its literal.</summary>
    private void Emit(IncrementalDetokenizer text, int token)
    {
        // Control tokens decode to nothing here, so the reply text stays clean; their literals come from the tokenizer.
        string delta = text.Push(token);
        if (delta.Length > 0)
            _events.Writer.TryWrite(new RealtimeVlmEvent(RealtimeVlmEventKind.Text, delta));
        if (token == _protocol.SilenceTokenId)
            _events.Writer.TryWrite(new RealtimeVlmEvent(RealtimeVlmEventKind.Silence, ControlLiteral(token) ?? string.Empty));
        else if (delta.Length == 0 && ControlLiteral(token) is { } literal)
            _events.Writer.TryWrite(new RealtimeVlmEvent(RealtimeVlmEventKind.Control, literal));
    }

    /// <summary>The literal of a control token (one that decodes to no text but has a written form), or null for ordinary tokens and
    /// tokenizers that cannot report bytes.</summary>
    private string? ControlLiteral(int token)
    {
        if (_tokenizer.TokenBytes(token, includeSpecial: false) is not { Length: 0 }) return null;
        return _tokenizer.TokenBytes(token, includeSpecial: true) is { Length: > 0 } bytes ? System.Text.Encoding.UTF8.GetString(bytes) : null;
    }

    private void Drain(List<TimedRgbFrame> frames, List<string> prompts)
    {
        int cap = _options.MaxFramesPerCycle > 0 ? _options.MaxFramesPerCycle : int.MaxValue;
        while (frames.Count < cap && _frames.Reader.TryRead(out TimedRgbFrame frame)) frames.Add(frame);
        while (_prompts.Reader.TryRead(out string? prompt)) prompts.Add(prompt);
    }

    /// <summary>The text one cycle appends: each prompt's segment (announced as a round start; partial text from before it is dropped),
    /// the protocol's text after prompts when frames came too, then each frame's segment in time order.</summary>
    private string BuildSegment(List<TimedRgbFrame> frames, List<string> prompts, ref IncrementalDetokenizer text)
    {
        System.Text.StringBuilder segment = new();
        foreach (string prompt in prompts)
        {
            segment.Append(_protocol.PromptSegment(prompt));
            _events.Writer.TryWrite(new RealtimeVlmEvent(RealtimeVlmEventKind.RoundStart, prompt));
        }
        // The reference clears its pending output bytes when a prompt arrives.
        if (prompts.Count > 0) text = new IncrementalDetokenizer(_tokenizer, includeSpecial: false);
        if (prompts.Count > 0 && frames.Count > 0 && _protocol.AfterPromptsWithFrames is { } after)
        {
            segment.Append(after);
            _events.Writer.TryWrite(new RealtimeVlmEvent(RealtimeVlmEventKind.Silence, after));
        }
        frames.Sort(static (a, b) => a.TimestampSeconds.CompareTo(b.TimestampSeconds));
        foreach (TimedRgbFrame frame in frames) segment.Append(_protocol.FrameSegment(frame.TimestampSeconds));
        return segment.ToString();
    }
}

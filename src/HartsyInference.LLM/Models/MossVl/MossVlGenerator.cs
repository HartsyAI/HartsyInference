using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Multimodal;
using HartsyInference.LLM.Multimodal.Streaming;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>One image of a MOSS-VL request: its merged vision tokens from the vision tower and its merged grid.</summary>
/// <param name="MergedTokens">The image's merged vision tokens <c>[h·w, hidden]</c> F32.</param>
/// <param name="MergedHeight">The merged grid height (patch height divided by the merge size).</param>
/// <param name="MergedWidth">The merged grid width.</param>
public readonly record struct MossVlImageFeatures(Tensor MergedTokens, int MergedHeight, int MergedWidth);

//// <summary>Size limits of a <see cref="MossVlDecodeSession"/>.</summary>
/// <param name="InitialTextCapacity">Text positions the self-attention cache starts with; it doubles as a session grows.</param>
/// <param name="MaxTextTokens">The most text tokens a session holds before it refuses to grow.</param>
/// <param name="MaxVisionTokens">The most vision tokens (frames and separators) the cross-attention cache holds. The model has no
/// visual-KV window, so a session that reaches it is refused rather than silently dropping vision.</param>
public sealed record MossVlSessionLimits(int InitialTextCapacity = 1024, int MaxTextTokens = 32768, int MaxVisionTokens = 65536);

/// <summary>A MOSS-VL token stream: segments of text and images are appended one forward at a time, and the model's state (the
/// self-attention cache, the vision keys and values of every image so far, and the next position) carries across them. A plain
/// request is one segment (the prompt with its images) followed by one-token segments (decode steps); a video or a live session
/// appends frames and prompts as they come.</summary>
/// <remarks>
/// <para>Positions follow the realtime model's <c>_compute_realtime_mrope_for_segment</c>: each segment starts where the last ended, and a
/// row sees every image before it in the stream (<see cref="MossVlPositions.PlanSegment"/>). Each image's vision keys and values are
/// projected once, when its segment runs, and appended to the cross-attention cache.</para>
/// <para>A forward cancelled through its token commits nothing and the session stays usable. A forward that throws anything else
/// faults the session, which then refuses further calls, because its caches may no longer match the tokens.</para>
/// </remarks>
public sealed class MossVlDecodeSession : IDisposable
{
    private readonly MossVlTextDecoder _decoder;
    private readonly MossVlConfig _config;
    private readonly IBackend _backend;
    private readonly MossVlSessionLimits _limits;
    private readonly CrossKvCache _crossKv;
    private FixedKvCache _cache;
    private int _cacheLength;
    private int _nextPosition;
    private bool _faulted;
    private int _disposed;
    private (int[] Ids, MossVlImageFeatures[] Images)? _pendingPrompt;

    /// <summary>Creates an empty session; append segments with <see cref="Forward{T}"/>.</summary>
    public MossVlDecodeSession(MossVlTextDecoder decoder, MossVlConfig config, IBackend backend, MossVlSessionLimits? limits = null)
    {
        _decoder = decoder;
        _config = config;
        _backend = backend;
        _limits = limits ?? new MossVlSessionLimits();
        TransformerConfig text = decoder.TextConfig;
        _cache = new FixedKvCache(text.NumLayers, 1, text.NumKvHeads, text.HeadDim, Math.Max(1, _limits.InitialTextCapacity));
        _crossKv = new CrossKvCache(text.NumLayers);
    }

    /// <summary>A one-request session: <see cref="Prefill()"/> runs <paramref name="promptIds"/> with <paramref name="images"/>, then
    /// <see cref="Step(int)"/> decodes. The caches are sized for the prompt and <paramref name="maxNewTokens"/>.</summary>
    /// <param name="images">The request's images, in the order their placeholders appear in <paramref name="promptIds"/>.</param>
    /// <exception cref="ArgumentException">No image, or a merged tensor that does not match its grid.</exception>
    public MossVlDecodeSession(MossVlTextDecoder decoder, MossVlConfig config, IBackend backend, int[] promptIds,
        IReadOnlyList<MossVlImageFeatures> images, int maxNewTokens)
        : this(decoder, config, backend, new MossVlSessionLimits(InitialTextCapacity: promptIds.Length + maxNewTokens + 1))
    {
        if (images.Count == 0)
        {
            Dispose();
            throw new ArgumentException("A MOSS-VL session needs at least one image.", nameof(images));
        }
        try
        {
            ValidateImages(images);
        }
        catch
        {
            Dispose();
            throw;
        }
        _pendingPrompt = (promptIds, [.. images]);
    }

    /// <summary>The position the next token takes.</summary>
    public int Position => _nextPosition;

    /// <summary>Text tokens the session holds.</summary>
    public int TextTokens => _cacheLength;

    /// <summary>Vision tokens (merged tokens and separators) the session holds.</summary>
    public int VisionTokens => _crossKv.Length;

    /// <summary>Appends one segment, <paramref name="ids"/> with the <paramref name="images"/> its placeholders reference (in order), and runs
    /// <paramref name="read"/> over the final-normed hidden state of its <c>ids.Length</c> rows.</summary>
    /// <exception cref="ArgumentException">An empty segment, or a merged tensor that does not match its grid.</exception>
    /// <exception cref="NotSupportedException">The segment would pass <see cref="MossVlSessionLimits"/>.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancel"/> fired; nothing was appended and the session is still usable.</exception>
    public T Forward<T>(ReadOnlySpan<int> ids, IReadOnlyList<MossVlImageFeatures> images, Func<Tensor, int, T> read,
        CancellationToken cancel = default)
    {
        ThrowIfUnusable();
        if (ids.IsEmpty) throw new ArgumentException("A segment needs at least one token.", nameof(ids));
        ValidateImages(images);
        int t = ids.Length;
        MossVlImage[] grids = images.Select(i => new MossVlImage(i.MergedHeight, i.MergedWidth)).ToArray();
        MossVlPositionPlan plan = MossVlPositions.PlanSegment(ids, _config.ImageTokenId, grids, _nextPosition, _crossKv.Length);
        if (_crossKv.Length + plan.VisionLength > _limits.MaxVisionTokens)
            throw new NotSupportedException($"The session would hold {_crossKv.Length + plan.VisionLength} vision tokens, past its limit of "
                + $"{_limits.MaxVisionTokens}. MOSS-VL has no visual-KV window to drop old frames; start a new session.");
        EnsureTextCapacity(_cacheLength + t);

        int[] tokens = ids.ToArray();
        Tensor? vision = images.Count > 0 ? _decoder.AppendSeparators(images.Select(i => i.MergedTokens).ToArray()) : null;
        try
        {
            using MossVlRope rope = MossVlPositions.BuildSegmentRope(plan, _config);
            using Tensor embeds = new(new TensorShape(1, t, _decoder.TextConfig.HiddenSize), DType.F32);
            _decoder.Model.EmbedLookup(embeds, tokens);
            using Tensor hidden = _decoder.Model.ForwardEmbeds(_backend, embeds, t, _cacheLength, _cache, applyFinalNorm: true,
                crossStates: vision, crossLen: plan.VisionLength, cancel: cancel, rope: rope.Rope, crossKv: _crossKv);
            _cacheLength += t;
            _nextPosition = plan.NextPosition;
            // Read before the tables and vision tensor are released, so an asynchronous backend never frees them under a queued kernel.
            return read(hidden, t);
        }
        catch (OperationCanceledException) when (cancel.IsCancellationRequested)
        {
            throw;   // the forward committed nothing: both caches still match the tokens
        }
        catch
        {
            _faulted = true;   // the caches may hold rows the session never counted
            throw;
        }
        finally
        {
            vision?.Dispose();
        }
    }

    /// <summary>Appends a segment and returns the logits after its last token.</summary>
    public float[] ForwardLogits(ReadOnlySpan<int> ids, IReadOnlyList<MossVlImageFeatures> images, CancellationToken cancel = default) =>
        Forward(ids, images, (hidden, t) => VlmDecodeOps.LastLogits(_backend, _decoder.Model, hidden, t), cancel);

    /// <summary>Appends a segment and samples the token after it with <paramref name="sampler"/>; the logits are read on the device.</summary>
    public int ForwardSample(ReadOnlySpan<int> ids, IReadOnlyList<MossVlImageFeatures> images, SamplerChain sampler, List<int> history,
        CancellationToken cancel = default) =>
        Forward(ids, images, (hidden, t) => VlmDecodeOps.SampleLast(_backend, _decoder.Model, hidden, t, sampler, history), cancel);

    /// <summary>Runs the prompt given at construction and returns the logits after its last token.</summary>
    public float[] Prefill()
    {
        (int[] ids, MossVlImageFeatures[] images) = TakePrompt();
        return ForwardLogits(ids, images);
    }

    /// <summary>Runs the prompt given at construction and selects the first generated token with <paramref name="sampler"/>.</summary>
    public int Prefill(SamplerChain sampler, List<int> history)
    {
        (int[] ids, MossVlImageFeatures[] images) = TakePrompt();
        return ForwardSample(ids, images, sampler, history);
    }

    /// <summary>Decodes one token after the last one and returns the logits after it.</summary>
    public float[] Step(int token)
    {
        RequirePrefilled();
        return ForwardLogits([token], []);
    }

    /// <summary>Decodes one token after the last one and selects the next with <paramref name="sampler"/>.</summary>
    public int Step(int token, SamplerChain sampler, List<int> history)
    {
        RequirePrefilled();
        return ForwardSample([token], [], sampler, history);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _crossKv.Dispose();
        _cache.Dispose();
    }

    private (int[] Ids, MossVlImageFeatures[] Images) TakePrompt()
    {
        ThrowIfUnusable();
        if (_pendingPrompt is not { } prompt)
            throw new InvalidOperationException(_cacheLength > 0 ? "The prompt is already prefilled." : "This session was created without a prompt; use Forward.");
        _pendingPrompt = null;
        return prompt;
    }

    private void RequirePrefilled()
    {
        if (_cacheLength == 0) throw new InvalidOperationException("Prefill must run before a decode step.");
    }

    /// <summary>Grows the self-attention cache (doubling, copied on the device) so it holds <paramref name="needed"/> tokens.</summary>
    private void EnsureTextCapacity(int needed)
    {
        if (needed <= _cache.Capacity) return;
        if (needed > _limits.MaxTextTokens)
            throw new NotSupportedException($"The session would hold {needed} text tokens, past its limit of {_limits.MaxTextTokens}; start a new session.");
        int capacity = Math.Min(_limits.MaxTextTokens, Math.Max(needed, _cache.Capacity * 2));
        FixedKvCache grown = _cache.CopyWithCapacity(capacity);
        _backend.Sync();   // the copy is queued on the device; the old buffers must outlive it
        _cache.Dispose();
        _cache = grown;
    }

    private static void ValidateImages(IReadOnlyList<MossVlImageFeatures> images)
    {
        for (int k = 0; k < images.Count; k++)
        {
            // Each image's own token count must match its grid: a total that only matches in sum would misplace the separators.
            long expected = (long)images[k].MergedHeight * images[k].MergedWidth;
            if (images[k].MergedTokens.Shape[0] != expected)
                throw new ArgumentException($"Image {k} has {images[k].MergedTokens.Shape[0]} merged tokens for a {images[k].MergedHeight}x{images[k].MergedWidth} grid ({expected}).", nameof(images));
        }
    }

    private void ThrowIfUnusable()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(MossVlDecodeSession));
        if (_faulted) throw new InvalidOperationException("An earlier forward failed part-way, so the session's caches no longer match its tokens; start a new session.");
    }
}

/// <summary>Greedy or sampled MOSS-VL generation for still images and a video with one question, in the layout the checkpoint's
/// processor and chat template produce (<see cref="MossVlRealtimeProtocol.OfflinePrompt"/>).</summary>
public sealed class MossVlGenerator(MossVlTextDecoder decoder, MossVlConfig config, ILlmTokenizer tokenizer, IBackend backend)
{
    private readonly MossVlRealtimeProtocol _protocol = new(config, tokenizer);

    /// <summary>Generates a reply to <paramref name="question"/> about one image. Sampling defaults match the other VLM generators
    /// (<see cref="MllamaGenerator"/>).</summary>
    public string Generate(string question, Tensor mergedVision, int mergedHeight, int mergedWidth, int maxTokens = 64,
        SamplingOptions? sampling = null) =>
        Generate(question, [new MossVlImageFeatures(mergedVision, mergedHeight, mergedWidth)], maxTokens, sampling);

    /// <summary>Generates a reply to <paramref name="question"/> about <paramref name="images"/>, shown in that order before the question.
    /// At least one image is required: a text-only prompt has no placeholder for the vision sequence.</summary>
    /// <exception cref="ArgumentException">No image is given.</exception>
    public string Generate(string question, IReadOnlyList<MossVlImageFeatures> images, int maxTokens = 64, SamplingOptions? sampling = null) =>
        Generate(question, images, maxTokens, sampling, onDelta: null, CancellationToken.None).Text;

    /// <summary>As <see cref="Generate(string, IReadOnlyList{MossVlImageFeatures}, int, SamplingOptions?)"/>, streaming the reply:
    /// <paramref name="onDelta"/> receives each new piece of decoded text as soon as it ends on a whole character, and
    /// <paramref name="cancel"/> is checked before every step.</summary>
    /// <exception cref="ArgumentException">No image is given.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancel"/> fired; the text so far was already streamed.</exception>
    public MossVlGeneration Generate(string question, IReadOnlyList<MossVlImageFeatures> images, int maxTokens, SamplingOptions? sampling,
        Action<string>? onDelta, CancellationToken cancel)
    {
        if (images.Count == 0) throw new ArgumentException("MOSS-VL generation needs at least one image.", nameof(images));
        return Run(question, images, [], maxTokens, sampling, onDelta, cancel);
    }

    /// <summary>Generates a reply to <paramref name="question"/> about a video: <paramref name="frames"/> are its sampled frames in time
    /// order, each with its timestamp, after any still <paramref name="images"/> (the processor's order).</summary>
    /// <exception cref="ArgumentException">Neither a frame nor an image is given.</exception>
    public MossVlGeneration GenerateVideo(string question, IReadOnlyList<VisionFrameFeatures> frames, int maxTokens, SamplingOptions? sampling,
        Action<string>? onDelta, CancellationToken cancel, IReadOnlyList<MossVlImageFeatures>? images = null)
    {
        images ??= [];
        if (frames.Count == 0 && images.Count == 0)
            throw new ArgumentException("MOSS-VL video generation needs at least one frame.", nameof(frames));
        return Run(question, images, frames, maxTokens, sampling, onDelta, cancel);
    }

    private MossVlGeneration Run(string question, IReadOnlyList<MossVlImageFeatures> images, IReadOnlyList<VisionFrameFeatures> frames,
        int maxTokens, SamplingOptions? sampling, Action<string>? onDelta, CancellationToken cancel)
    {
        int[] ids = tokenizer.Encode(_protocol.OfflinePrompt(images.Count, frames.Select(f => f.TimestampSeconds).ToArray(), question),
            addSpecial: true);
        if (maxTokens <= 0) return new MossVlGeneration(string.Empty, ids.Length, 0, HitTokenLimit: true);
        SamplerChain sampler = SamplerChain.FromOptions(sampling ?? new SamplingOptions
        {
            Temperature = 0.4f, TopP = 0.9f, RepetitionPenalty = 1.1f, Seed = 1,
        }, tokenizer, config.Text.VocabSize);
        HashSet<int> stops = new(tokenizer.StopIds);
        // The turn ends with <|im_end|> whatever the tokenizer source lists as its EOS (a GGUF's own vocabulary may name another).
        if (tokenizer.SpecialId("<|im_end|>") is int imEnd) stops.Add(imEnd);
        List<int> history = [];
        // Placeholders appear images first, then the video's frames, which is the order the session reads the features in.
        MossVlImageFeatures[] vision = [.. images, .. frames.Select(f => new MossVlImageFeatures(f.MergedTokens, f.MergedHeight, f.MergedWidth))];

        using MossVlDecodeSession session = new(decoder, config, backend,
            new MossVlSessionLimits(InitialTextCapacity: ids.Length + maxTokens + 1, MaxTextTokens: Math.Max(32768, ids.Length + maxTokens + 1)));
        cancel.ThrowIfCancellationRequested();
        int first = session.ForwardSample(ids, vision, sampler, history, cancel);
        (string text, int tokens, bool hitLimit) = VlmDecodeOps.DecodeLoop(tokenizer, first,
            token => session.ForwardSample([token], [], sampler, history, cancel), stops, maxTokens, history, onDelta, cancel);
        return new MossVlGeneration(text, ids.Length, tokens, hitLimit);
    }
}

/// <summary>One finished MOSS-VL reply.</summary>
/// <param name="HitTokenLimit">True when generation stopped at the token limit rather than at the end of the turn.</param>
public readonly record struct MossVlGeneration(string Text, int PromptTokens, int CompletionTokens, bool HitTokenLimit);

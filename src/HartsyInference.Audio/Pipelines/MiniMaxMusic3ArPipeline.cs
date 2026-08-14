using System.Diagnostics;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Pipelines;

/// <summary>MiniMax Music 3's autoregressive stage: frame by frame the global language model samples a semantic code
/// under classifier-free guidance and the depth decoder samples the seven residual codes, and the per-frame hidden
/// states of both are concatenated into the conditioning the flow-matching stage consumes.
///
/// <para>Classifier-free guidance runs as two batch-1 branches with their own KV caches rather than one batch-2
/// pass, matching how the engine's other guided autoregressive audio models decode. The two branches differ only in
/// their prompt: the unconditional one replaces every token except the first and the last two with the audio-CFG
/// token.</para></summary>
public sealed unsafe class MiniMaxMusic3ArPipeline : IDisposable
{
    /// <summary>Autoregressive frames per second.</summary>
    public const double FrameRate = 25.0;

    /// <summary>Longest generation the checkpoint supports.</summary>
    public const int MaxAudioFrames = 9000;

    /// <summary>Width of one frame's conditioning: the language model's hidden state plus the seven depth states.</summary>
    public const int FrameHiddenWidth = MiniMaxMusic3DepthDecoder.NumCodebooks * MiniMaxMusic3GlobalLm.HiddenSize;

    // Both scales and the top-k cutoff are fixed by the reference inference recipe, not user knobs.
    private const float CfgScale = 1.5f;
    private const int TopK = 50;

    /// <summary>Classifier-free rows carried through the depth decoder: conditional, then unconditional.</summary>
    private const int CfgRows = 2;

    private readonly MiniMaxMusic3GlobalLm _languageModel;
    private readonly MiniMaxMusic3DepthDecoder _depthDecoder;
    private int _disposed;

    public MiniMaxMusic3ArPipeline(MiniMaxMusic3GlobalLm languageModel, MiniMaxMusic3DepthDecoder depthDecoder)
    {
        ArgumentNullException.ThrowIfNull(languageModel);
        ArgumentNullException.ThrowIfNull(depthDecoder);
        _languageModel = languageModel;
        _depthDecoder = depthDecoder;
    }

    /// <summary>Generates up to <paramref name="maxFrames"/> frames and returns their concatenated hidden states,
    /// flat <c>[frames · 8 · 4096]</c>, plus the frame count. Generation stops early on the end-of-audio token.
    ///
    /// <para><paramref name="forcedCodes"/> replaces sampling with the supplied per-frame codes, which is how the
    /// parity tests take the sampler's RNG out of the comparison; pass null for real generation.</para></summary>
    public (float[] FrameHiddens, int Frames) Generate(
        IBackend backend,
        ReadOnlySpan<int> conditionalIds,
        ReadOnlySpan<int> unconditionalIds,
        int maxFrames,
        int seed,
        Action<int, int>? onFrame = null,
        CancellationToken cancel = default,
        IReadOnlyList<int[]>? forcedCodes = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (conditionalIds.Length != unconditionalIds.Length)
        {
            throw new ArgumentException("the conditional and unconditional prompts must be the same length.", nameof(unconditionalIds));
        }
        int frameLimit = Math.Clamp(maxFrames, 1, MaxAudioFrames);

        // The context never revisits a position, so the cache is sized to what this run can reach — not to the
        // config's max_position_embeddings, which the reference also exceeds for long songs.
        int cacheLength = conditionalIds.Length + frameLimit + 2;
        using IKvCache conditionalCache = _languageModel.CreateCache(cacheLength);
        using IKvCache unconditionalCache = _languageModel.CreateCache(cacheLength);
        using MiniMaxMusic3DepthCache depthCache = _depthDecoder.CreateCache(CfgRows);

        uint rng = DeterministicRng.Seed(seed);
        // Phase attribution for the perf grind. CUDA launches are async, so each phase is billed at the next host
        // read that forces a sync -- good enough to rank the phases, not to trust to the millisecond.
        long lmTicks = 0, depthTicks = 0, sampleTicks = 0, feedbackTicks = 0;
        int codebooks = MiniMaxMusic3DepthDecoder.NumCodebooks;
        int hidden = MiniMaxMusic3GlobalLm.HiddenSize;
        List<float[]> frames = new List<float[]>(Math.Min(frameLimit, 1024));
        int[] frameCodes = new int[codebooks];

        Tensor conditionalHidden;
        Tensor unconditionalHidden;
        using (Tensor conditionalEmbeds = _languageModel.Embed(conditionalIds))
        {
            conditionalHidden = _languageModel.Forward(backend, conditionalEmbeds, conditionalCache);
        }
        using (Tensor unconditionalEmbeds = _languageModel.Embed(unconditionalIds))
        {
            unconditionalHidden = _languageModel.Forward(backend, unconditionalEmbeds, unconditionalCache);
        }

        try
        {
            // The first decode only advances the state past <|audio_start|>; its frame is not emitted.
            for (int frameIndex = 0; frameIndex <= frameLimit; frameIndex++)
            {
                cancel.ThrowIfCancellationRequested();
                long phase = Stopwatch.GetTimestamp();
                int semanticCode = forcedCodes is not null
                    ? (frameIndex < forcedCodes.Count ? forcedCodes[frameIndex][0] : MiniMaxMusic3GlobalLm.AudioEndLogitIndex)
                    : SampleSemantic(backend, conditionalHidden, unconditionalHidden, ref rng);
                sampleTicks += Stopwatch.GetTimestamp() - phase;
                if (semanticCode == MiniMaxMusic3GlobalLm.AudioEndLogitIndex)
                {
                    break;
                }

                frameCodes[0] = semanticCode;
                phase = Stopwatch.GetTimestamp();
                float[] depthHidden = DecodeDepth(backend, conditionalHidden, unconditionalHidden, frameCodes,
                    ref rng, forcedCodes is not null ? forcedCodes[frameIndex] : null, depthCache);
                depthTicks += Stopwatch.GetTimestamp() - phase;

                if (frameIndex > 0)
                {
                    float[] frame = new float[FrameHiddenWidth];
                    conditionalHidden.AsReadOnlySpan<float>().CopyTo(frame);
                    depthHidden.CopyTo(frame, hidden);
                    frames.Add(frame);
                    onFrame?.Invoke(frames.Count, frameLimit);
                    if (frames.Count >= frameLimit)
                    {
                        break;
                    }
                }

                phase = Stopwatch.GetTimestamp();
                using Tensor feedback = BuildFeedback(frameCodes);
                feedbackTicks += Stopwatch.GetTimestamp() - phase;
                phase = Stopwatch.GetTimestamp();
                Tensor nextConditional = _languageModel.Forward(backend, feedback, conditionalCache);
                Tensor nextUnconditional = _languageModel.Forward(backend, feedback, unconditionalCache);
                lmTicks += Stopwatch.GetTimestamp() - phase;
                conditionalHidden.Dispose();
                unconditionalHidden.Dispose();
                conditionalHidden = nextConditional;
                unconditionalHidden = nextUnconditional;
            }
        }
        finally
        {
            conditionalHidden.Dispose();
            unconditionalHidden.Dispose();
        }

        double ms = 1000.0 / Stopwatch.Frequency;
        Logs.Info($"[Audio][MiniMaxMusic3] AR phases over {frames.Count} frames — language model {lmTicks * ms / 1000:0.0}s, "
            + $"depth decoder {depthTicks * ms / 1000:0.0}s, sampling {sampleTicks * ms / 1000:0.0}s, "
            + $"feedback embed {feedbackTicks * ms / 1000:0.0}s.");
        if (frames.Count == 0)
        {
            throw new InvalidOperationException(
                "MiniMax Music 3 generated zero audio frames — the prompt ended generation immediately.");
        }
        float[] frameHiddens = new float[(long)frames.Count * FrameHiddenWidth];
        for (int i = 0; i < frames.Count; i++)
        {
            frames[i].CopyTo(frameHiddens, (long)i * FrameHiddenWidth);
        }
        return (frameHiddens, frames.Count);
    }

    public void Dispose() => Interlocked.Exchange(ref _disposed, 1);

    /// <summary>Guided semantic draw: the candidate set is the conditional branch's top-K, but the distribution
    /// sampled is the guided one. Guidance over two masked-out logits would be NaN, so the mask is applied after.</summary>
    private int SampleSemantic(IBackend backend, Tensor conditionalHidden, Tensor unconditionalHidden, ref uint rng)
    {
        using Tensor conditionalLogits = _languageModel.SemanticLogits(backend, conditionalHidden);
        using Tensor unconditionalLogits = _languageModel.SemanticLogits(backend, unconditionalHidden);
        int count = MiniMaxMusic3GlobalLm.SemanticVocabSize + 1;
        ReadOnlySpan<float> conditional = conditionalLogits.AsReadOnlySpan<float>()[..count];
        ReadOnlySpan<float> unconditional = unconditionalLogits.AsReadOnlySpan<float>()[..count];
        float[] guided = new float[count];
        float threshold = KthLargest(conditional, TopK);
        for (int i = 0; i < count; i++)
        {
            guided[i] = conditional[i] < threshold
                ? float.NegativeInfinity
                : unconditional[i] + ((conditional[i] - unconditional[i]) * CfgScale);
        }
        return NucleusSampler.Draw(guided, count, temperature: 1f, topK: TopK, topP: 1f, ref rng);
    }

    /// <summary>Samples the seven residual codes for one frame and returns their concatenated conditional hidden
    /// states, <c>[7 · 4096]</c>. Both classifier-free rows share every sampled code, so the depth sequence differs
    /// between them only in its first element.</summary>
    private float[] DecodeDepth(IBackend backend, Tensor conditionalHidden, Tensor unconditionalHidden,
        int[] frameCodes, ref uint rng, int[]? forced, MiniMaxMusic3DepthCache cache)
    {
        int hidden = MiniMaxMusic3GlobalLm.HiddenSize;
        int codebooks = MiniMaxMusic3DepthDecoder.NumCodebooks;
        float[] depthHidden = new float[(codebooks - 1) * hidden];
        cache.Reset();

        Tensor states;
        // The global hidden state and the frame's semantic code are both known before any residual is sampled,
        // so they enter as one two-step block; every later step is a single token against the cache.
        using (Tensor prefix = new Tensor(new TensorShape(CfgRows, 2, hidden), DType.F32))
        {
            float* prefixData = (float*)prefix.DataPointer;
            conditionalHidden.AsReadOnlySpan<float>().CopyTo(new Span<float>(prefixData, hidden));
            unconditionalHidden.AsReadOnlySpan<float>().CopyTo(new Span<float>(prefixData + (2 * hidden), hidden));
            _languageModel.ReadEmbeddingRow(MiniMaxMusic3GlobalLm.AudioCodeOffset + frameCodes[0],
                new Span<float>(prefixData + hidden, hidden));
            new ReadOnlySpan<float>(prefixData + hidden, hidden).CopyTo(new Span<float>(prefixData + (3 * hidden), hidden));
            using Tensor projected = _depthDecoder.Project(backend, prefix);
            using Tensor block = _depthDecoder.Forward(backend, projected, cache);
            states = LastStep(block, hidden);
        }
        try
        {
            for (int index = 1; index < codebooks; index++)
            {
                int code;
                if (forced is not null)
                {
                    code = forced[index];
                }
                else
                {
                    // Ahead of the host read below, which would otherwise strand the head's input on the host
                    // and pay to upload it again.
                    using Tensor logits = _depthDecoder.Head(backend, index, states);
                    ReadOnlySpan<float> values = logits.AsReadOnlySpan<float>();
                    int vocab = MiniMaxMusic3DepthDecoder.AudioVocabSize;
                    float[] guided = new float[vocab];
                    for (int i = 0; i < vocab; i++)
                    {
                        guided[i] = values[vocab + i] + ((values[i] - values[vocab + i]) * CfgScale);
                    }
                    code = NucleusSampler.Draw(guided, vocab, temperature: 1f, topK: TopK, topP: 1f, ref rng);
                }
                states.AsReadOnlySpan<float>()[..hidden].CopyTo(depthHidden.AsSpan((index - 1) * hidden, hidden));
                frameCodes[index] = code;

                if (index < codebooks - 1)
                {
                    using Tensor embedded = new Tensor(new TensorShape(CfgRows, 1, hidden), DType.F32);
                    float* embeddedData = (float*)embedded.DataPointer;
                    using (Tensor row = _depthDecoder.EmbedResidual(index, code))
                    {
                        row.AsReadOnlySpan<float>().CopyTo(new Span<float>(embeddedData, hidden));
                    }
                    new ReadOnlySpan<float>(embeddedData, hidden).CopyTo(new Span<float>(embeddedData + hidden, hidden));
                    using Tensor projected = _depthDecoder.Project(backend, embedded);
                    Tensor next = _depthDecoder.Forward(backend, projected, cache);
                    states.Dispose();
                    states = next;
                }
            }
        }
        finally
        {
            states.Dispose();
        }
        return depthHidden;
    }

    /// <summary>The last step of each row of <paramref name="block"/> <c>[rows, steps, hidden]</c>, as
    /// <c>[rows, 1, hidden]</c> — the only step the caller samples from.</summary>
    private static Tensor LastStep(Tensor block, int hidden)
    {
        int steps = (int)block.Shape[1];
        Tensor last = new Tensor(new TensorShape(CfgRows, 1, hidden), DType.F32);
        float* data = (float*)last.DataPointer;
        ReadOnlySpan<float> values = block.AsReadOnlySpan<float>();
        for (int row = 0; row < CfgRows; row++)
        {
            values.Slice(((row * steps) + steps - 1) * hidden, hidden)
                .CopyTo(new Span<float>(data + ((long)row * hidden), hidden));
        }
        return last;
    }

    /// <summary>The global model's next input: the semantic code's token embedding plus every residual code's
    /// embedding, scaled by <c>numCodebooks^-0.5</c>. Both branches consume the same frame, so one row suffices.</summary>
    private Tensor BuildFeedback(ReadOnlySpan<int> frameCodes)
    {
        int hidden = MiniMaxMusic3GlobalLm.HiddenSize;
        Tensor feedback = new Tensor(new TensorShape(1, 1, hidden), DType.F32);
        Span<float> values = new Span<float>((float*)feedback.DataPointer, hidden);
        _languageModel.ReadEmbeddingRow(MiniMaxMusic3GlobalLm.AudioCodeOffset + frameCodes[0], values);
        _depthDecoder.AccumulateFrameResiduals(frameCodes, values);
        float scale = 1f / MathF.Sqrt(MiniMaxMusic3DepthDecoder.NumCodebooks);
        for (int i = 0; i < hidden; i++)
        {
            values[i] *= scale;
        }
        return feedback;
    }

    /// <summary>The <paramref name="k"/>-th largest value, the cutoff <c>torch.topk(...).values[..., -1]</c> returns.</summary>
    private static float KthLargest(ReadOnlySpan<float> values, int k)
    {
        int count = Math.Min(k, values.Length);
        Span<float> top = stackalloc float[count];
        top.Fill(float.NegativeInfinity);
        for (int i = 0; i < values.Length; i++)
        {
            float value = values[i];
            if (value <= top[count - 1])
            {
                continue;
            }
            int position = count - 1;
            while (position > 0 && top[position - 1] < value)
            {
                top[position] = top[position - 1];
                position--;
            }
            top[position] = value;
        }
        return top[count - 1];
    }
}

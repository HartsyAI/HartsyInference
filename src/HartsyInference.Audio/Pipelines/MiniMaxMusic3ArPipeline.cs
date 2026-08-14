using System.Diagnostics;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Pipelines;

/// <summary>MiniMax Music 3's autoregressive stage: frame by frame the global language model samples a semantic code
/// under classifier-free guidance and the depth decoder samples the seven residual codes, and the per-frame hidden
/// states of both are concatenated into the conditioning the flow-matching stage consumes.
///
/// <para>Classifier-free guidance keeps one KV cache per branch, but the steady-state decode runs both branches
/// through a single batch-2 forward: at one token per step the language model is bound by streaming its weights,
/// so two batch-1 forwards read all of them twice per frame. The branches differ only in their prompt — the
/// unconditional one replaces every token except the first and the last two with the audio-CFG token — so they stay
/// position-aligned and can share a step. <c>HARTSY_MM3_CFG_BATCH=0</c> restores the two-forward decode.</para></summary>
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
    private readonly bool _batchCfg = EnvSwitch.IsEnabled("HARTSY_MM3_CFG_BATCH", defaultOn: true);
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

        Tensor conditionalPrefill;
        using (Tensor conditionalEmbeds = _languageModel.Embed(conditionalIds))
        {
            conditionalPrefill = _languageModel.Forward(backend, conditionalEmbeds, conditionalCache);
        }
        Tensor unconditionalPrefill;
        using (Tensor unconditionalEmbeds = _languageModel.Embed(unconditionalIds))
        {
            unconditionalPrefill = _languageModel.Forward(backend, unconditionalEmbeds, unconditionalCache);
        }
        // Prefill stays one forward per branch — it runs twice per song, not twice per frame, and the two prompts
        // are separate sequences of many tokens rather than one aligned step.
        Tensor hiddens = StackRows(conditionalPrefill, unconditionalPrefill, hidden);
        conditionalPrefill.Dispose();
        unconditionalPrefill.Dispose();

        try
        {
            // The first decode only advances the state past <|audio_start|>; its frame is not emitted.
            for (int frameIndex = 0; frameIndex <= frameLimit; frameIndex++)
            {
                cancel.ThrowIfCancellationRequested();
                long phase = Stopwatch.GetTimestamp();
                int semanticCode = forcedCodes is not null
                    ? (frameIndex < forcedCodes.Count ? forcedCodes[frameIndex][0] : MiniMaxMusic3GlobalLm.AudioEndLogitIndex)
                    : SampleSemantic(backend, hiddens, ref rng);
                sampleTicks += Stopwatch.GetTimestamp() - phase;
                if (semanticCode == MiniMaxMusic3GlobalLm.AudioEndLogitIndex)
                {
                    break;
                }

                frameCodes[0] = semanticCode;
                phase = Stopwatch.GetTimestamp();
                float[] depthHidden = DecodeDepth(backend, hiddens, frameCodes,
                    ref rng, forcedCodes is not null ? forcedCodes[frameIndex] : null, depthCache);
                depthTicks += Stopwatch.GetTimestamp() - phase;

                if (frameIndex > 0)
                {
                    float[] frame = new float[FrameHiddenWidth];
                    hiddens.AsReadOnlySpan<float>()[..hidden].CopyTo(frame);
                    depthHidden.CopyTo(frame, hidden);
                    frames.Add(frame);
                    onFrame?.Invoke(frames.Count, frameLimit);
                    if (frames.Count >= frameLimit)
                    {
                        break;
                    }
                }

                phase = Stopwatch.GetTimestamp();
                using Tensor feedback = BuildFeedback(frameCodes, _batchCfg ? CfgRows : 1);
                feedbackTicks += Stopwatch.GetTimestamp() - phase;
                phase = Stopwatch.GetTimestamp();
                Tensor next = _batchCfg
                    ? _languageModel.ForwardCfgStep(backend, feedback, conditionalCache, unconditionalCache)
                    : ForwardBranches(backend, feedback, conditionalCache, unconditionalCache, hidden);
                lmTicks += Stopwatch.GetTimestamp() - phase;
                hiddens.Dispose();
                hiddens = next;
            }
        }
        finally
        {
            hiddens.Dispose();
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

    /// <summary>Stacks the two branches' last hidden states into <c>[2, 4096]</c>, row 0 conditional.</summary>
    private static Tensor StackRows(Tensor conditional, Tensor unconditional, int hidden)
    {
        Tensor rows = new Tensor(new TensorShape(CfgRows, hidden), DType.F32);
        float* data = (float*)rows.DataPointer;
        conditional.AsReadOnlySpan<float>()[..hidden].CopyTo(new Span<float>(data, hidden));
        unconditional.AsReadOnlySpan<float>()[..hidden].CopyTo(new Span<float>(data + hidden, hidden));
        return rows;
    }

    /// <summary>The pre-batching decode: one batch-1 forward per branch, stacked into the same <c>[2, 4096]</c> the
    /// batched step returns. Reachable only through <c>HARTSY_MM3_CFG_BATCH=0</c>.</summary>
    private Tensor ForwardBranches(IBackend backend, Tensor feedback, IKvCache conditionalCache,
        IKvCache unconditionalCache, int hidden)
    {
        using Tensor conditional = _languageModel.Forward(backend, feedback, conditionalCache);
        using Tensor unconditional = _languageModel.Forward(backend, feedback, unconditionalCache);
        return StackRows(conditional, unconditional, hidden);
    }

    /// <summary>Row <paramref name="row"/> of a <c>[rows, hidden]</c> tensor as its own <c>[1, hidden]</c>.</summary>
    private static Tensor Row(Tensor rows, int row, int hidden)
    {
        Tensor single = new Tensor(new TensorShape(1, hidden), DType.F32);
        rows.AsReadOnlySpan<float>()
            .Slice(row * hidden, hidden)
            .CopyTo(new Span<float>((float*)single.DataPointer, hidden));
        return single;
    }

    /// <summary>Guided semantic draw: the candidate set is the conditional branch's top-K, but the distribution
    /// sampled is the guided one. Guidance over two masked-out logits would be NaN, so the mask is applied after.
    ///
    /// <para>The head runs per branch even though both rows are on hand. Batching it saves ~0.2 s per 15 s of audio,
    /// but cuBLAS picks a different algorithm at two rows than at one, and the last-bit difference in the logits is
    /// enough to make the sampler pick a different song for the same seed — measured, not theoretical. Batching the
    /// DECODE step is bit-exact and free of that; this is not, and 0.2 s does not buy a reproducibility break.</para></summary>
    private int SampleSemantic(IBackend backend, Tensor hiddens, ref uint rng)
    {
        int count = MiniMaxMusic3GlobalLm.SemanticVocabSize + 1;
        int hidden = MiniMaxMusic3GlobalLm.HiddenSize;
        float[] guided = new float[count];
        using (Tensor conditionalRow = Row(hiddens, 0, hidden))
        using (Tensor unconditionalRow = Row(hiddens, 1, hidden))
        using (Tensor conditionalLogits = _languageModel.SemanticLogits(backend, conditionalRow))
        using (Tensor unconditionalLogits = _languageModel.SemanticLogits(backend, unconditionalRow))
        {
            BlendGuided(guided, conditionalLogits.AsReadOnlySpan<float>()[..count],
                unconditionalLogits.AsReadOnlySpan<float>()[..count], count);
        }
        return NucleusSampler.Draw(guided, count, temperature: 1f, topK: TopK, topP: 1f, ref rng);
    }

    /// <summary>Writes the guided distribution over the conditional branch's top-K into <paramref name="guided"/>,
    /// masking the rest to <c>-inf</c> AFTER the blend — guidance over two already-masked logits is NaN.</summary>
    private static void BlendGuided(float[] guided, ReadOnlySpan<float> conditional,
        ReadOnlySpan<float> unconditional, int count)
    {
        float threshold = KthLargest(conditional, TopK);
        for (int i = 0; i < count; i++)
        {
            guided[i] = conditional[i] < threshold
                ? float.NegativeInfinity
                : unconditional[i] + ((conditional[i] - unconditional[i]) * CfgScale);
        }
    }

    /// <summary>Samples the seven residual codes for one frame and returns their concatenated conditional hidden
    /// states, <c>[7 · 4096]</c>. Both classifier-free rows share every sampled code, so the depth sequence differs
    /// between them only in its first element.</summary>
    private float[] DecodeDepth(IBackend backend, Tensor hiddens,
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
            ReadOnlySpan<float> rows = hiddens.AsReadOnlySpan<float>();
            rows[..hidden].CopyTo(new Span<float>(prefixData, hidden));
            rows.Slice(hidden, hidden).CopyTo(new Span<float>(prefixData + (2 * hidden), hidden));
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
    /// embedding, scaled by <c>numCodebooks^-0.5</c>. Both branches consume the same frame, so the single computed
    /// row is simply repeated <paramref name="rows"/> times for the batched step.</summary>
    private Tensor BuildFeedback(ReadOnlySpan<int> frameCodes, int rows)
    {
        int hidden = MiniMaxMusic3GlobalLm.HiddenSize;
        Tensor feedback = new Tensor(new TensorShape(1, rows, hidden), DType.F32);
        float* data = (float*)feedback.DataPointer;
        Span<float> values = new Span<float>(data, hidden);
        _languageModel.ReadEmbeddingRow(MiniMaxMusic3GlobalLm.AudioCodeOffset + frameCodes[0], values);
        _depthDecoder.AccumulateFrameResiduals(frameCodes, values);
        float scale = 1f / MathF.Sqrt(MiniMaxMusic3DepthDecoder.NumCodebooks);
        for (int i = 0; i < hidden; i++)
        {
            values[i] *= scale;
        }
        for (int row = 1; row < rows; row++)
        {
            new ReadOnlySpan<float>(data, hidden).CopyTo(new Span<float>(data + ((long)row * hidden), hidden));
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

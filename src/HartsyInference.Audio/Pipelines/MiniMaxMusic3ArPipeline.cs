using System.Diagnostics;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
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
/// position-aligned and can share a step. <c>numerics.mm3CfgBatch=false</c> restores the two-forward decode.</para>
///</summary>
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
        // Read once here, not per use: the two call sites below must agree — the feedback is built with two rows
        // exactly when the batched step consumes two — so a live read at each would let them tear if a concurrent
        // request changed the setting between them. Once per generation is also what the knob's Runtime scope
        // means; this was a readonly field bound in the constructor, and the pipeline is built during model load
        // and then cached and reused, so every later request inherited whatever the first load happened to see.
        bool batchCfg = EngineKnobs.Mm3CfgBatch.Value;
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

        if (forcedCodes is null && batchCfg && EngineKnobs.Mm3DeviceAr.Value && backend.DeviceSamplingSupported
            && TopK <= backend.DeviceSamplingMaxTopK)
        {
            return GenerateOnDevice(backend, hiddens, conditionalCache, unconditionalCache, depthCache, frameLimit, seed, onFrame, cancel);
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
                using Tensor feedback = BuildFeedback(frameCodes, batchCfg ? CfgRows : 1);
                feedbackTicks += Stopwatch.GetTimestamp() - phase;
                phase = Stopwatch.GetTimestamp();
                Tensor next = batchCfg
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

    /// <summary>Frames per end-of-audio batch in the device loop. Frame vectors and codes live on the device until a batch
    /// is read back, so a batch bounds the device memory and the readbacks.</summary>
    private const int DeviceFrameRing = 256;

    /// <summary>The first end-of-audio batch in the device loop. Each later batch doubles up to <see cref="DeviceFrameRing"/>,
    /// so a short clip stops after a few extra frames instead of a whole ring.</summary>
    private const int FirstDeviceBatchFrames = 16;

    /// <summary>The device-resident AR loop (opt-in: <c>numerics.mm3DeviceAr</c>). The semantic draw, the six residual draws, the
    /// depth embeddings, the feedback and the frame hiddens stay on the device. The end-of-audio check reads the codes once per
    /// <see cref="DeviceFrameRing"/> frames; frames at or after end-of-audio are dropped, so the output matches a per-frame check.
    /// The draws use the device RNG, so same-seed output differs from the host loop.</summary>
    private (float[] FrameHiddens, int Frames) GenerateOnDevice(
        IBackend backend, Tensor hiddens, IKvCache conditionalCache, IKvCache unconditionalCache,
        MiniMaxMusic3DepthCache depthCache, int frameLimit, int seed, Action<int, int>? onFrame, CancellationToken cancel)
    {
        const int codebooks = MiniMaxMusic3DepthDecoder.NumCodebooks;
        int hidden = MiniMaxMusic3GlobalLm.HiddenSize;
        int frameWidth = FrameHiddenWidth;
        Tensor semanticRows = _languageModel.SemanticEmbedRows;
        ulong rngState = backend.AllocDeviceRng(((ulong)(uint)seed << 32) | 0x9E3779B9ul, TopK);
        ulong[] tokens = new ulong[DeviceFrameRing * codebooks];
        List<(int FrameIndex, Tensor Vector)> pending = new List<(int, Tensor)>(DeviceFrameRing);
        List<float[]> frames = new List<float[]>(Math.Min(frameLimit, 1024));
        bool ended = false;
        try
        {
            for (int i = 0; i < tokens.Length; i++)
            {
                tokens[i] = backend.AllocDeviceTokenId();
            }
            backend.PreloadWeights([semanticRows]);
            for (int index = 1; index < codebooks; index++)
            {
                backend.PreloadWeights([_depthDecoder.ResidualRows(index)]);
            }
            float feedbackScale = 1f / MathF.Sqrt(codebooks);
            int cycleStart = 0;
            int batchFrames = FirstDeviceBatchFrames;
            for (int frameIndex = 0; frameIndex <= frameLimit && !ended; frameIndex++)
            {
                cancel.ThrowIfCancellationRequested();
                int slot = (frameIndex % DeviceFrameRing) * codebooks;

                // Semantic draw from the guided top-K of the conditional row, then its embedding.
                DrawSemanticDevice(backend, hiddens, tokens[slot], rngState);
                Tensor semEmbed = new Tensor(new TensorShape(1, 1, hidden), DType.F32);
                backend.EmbedGatherDecodeStep(semEmbed, semanticRows, tokens[slot]);

                // The depth decoder over [hidden, semantic embedding] per CFG row, then the residual draws.
                depthCache.Reset();
                Tensor frameRow0 = new Tensor(new TensorShape(1, hidden), DType.F32);
                backend.SliceRows(frameRow0, hiddens, 0);
                Tensor state = DepthPrefixStates(backend, hiddens, semEmbed, depthCache);
                Tensor[] parts = new Tensor[codebooks];
                parts[0] = frameRow0;
                for (int index = 1; index < codebooks; index++)
                {
                    parts[index] = new Tensor(new TensorShape(1, hidden), DType.F32);
                    backend.SliceRows(parts[index], state, 0);
                    DrawResidualDevice(backend, state, index, tokens[slot + index], rngState);
                    if (index < codebooks - 1)
                    {
                        state = DepthStepStates(backend, index, tokens[slot + index], depthCache, state);
                    }
                }
                state.Dispose();

                // Frame vector for this frame: the hidden row, then the seven residual states of the conditional row.
                if (frameIndex > 0)
                {
                    Tensor vector = new Tensor(new TensorShape(codebooks, hidden), DType.F32);
                    backend.Concat(vector, parts, 0);
                    pending.Add((frameIndex, vector));
                }
                foreach (Tensor part in parts)
                {
                    part.Dispose();   // Concat copied the parts (or none was needed on frame 0)
                }

                // Feedback: the semantic embedding plus the residual embeddings, scaled, duplicated for the two CFG rows.
                Tensor feedbackSum = new Tensor(new TensorShape(1, 1, hidden), DType.F32);
                backend.Scale(feedbackSum, semEmbed, 1f);
                semEmbed.Dispose();
                for (int index = 1; index < codebooks; index++)
                {
                    Tensor residual = new Tensor(new TensorShape(1, 1, hidden), DType.F32);
                    backend.EmbedGatherDecodeStep(residual, _depthDecoder.ResidualRows(index), tokens[slot + index]);
                    backend.Add(feedbackSum, feedbackSum, residual);
                    residual.Dispose();
                }
                backend.Scale(feedbackSum, feedbackSum, feedbackScale);
                Tensor feedback = new Tensor(new TensorShape(1, 2, hidden), DType.F32);
                backend.Concat(feedback, [feedbackSum, feedbackSum], 1);
                feedbackSum.Dispose();
                Tensor next = _languageModel.ForwardCfgStep(backend, feedback, conditionalCache, unconditionalCache);
                feedback.Dispose();
                hiddens.Dispose();
                hiddens = next;

                // Batches double from FirstDeviceBatchFrames up to DeviceFrameRing. A batch never exceeds the ring, so its
                // slots are distinct and each is read before the next batch overwrites it.
                bool batchEnd = frameIndex - cycleStart + 1 == batchFrames || frameIndex == frameLimit;
                if (batchEnd)
                {
                    ended = FlushDeviceBatch(backend, tokens, cycleStart, frameIndex, pending, frames, frameLimit, onFrame, frameWidth);
                    cycleStart = frameIndex + 1;
                    batchFrames = Math.Min(DeviceFrameRing, batchFrames * 2);
                }
            }
        }
        finally
        {
            foreach ((int _, Tensor vector) in pending)
            {
                vector.Dispose();
            }
            foreach (ulong token in tokens)
            {
                if (token != 0) backend.FreeDeviceTokenId(token);
            }
            backend.FreeDeviceRng(rngState);
            backend.FreeWeights([semanticRows]);
            for (int index = 1; index < codebooks; index++)
            {
                backend.FreeWeights([_depthDecoder.ResidualRows(index)]);
            }
            hiddens.Dispose();
        }

        if (frames.Count == 0)
        {
            throw new InvalidOperationException(
                "MiniMax Music 3 generated zero audio frames — the prompt ended generation immediately.");
        }
        float[] frameHiddens = new float[(long)frames.Count * frameWidth];
        for (int i = 0; i < frames.Count; i++)
        {
            frames[i].CopyTo(frameHiddens, (long)i * frameWidth);
        }
        return (frameHiddens, frames.Count);
    }

    /// <summary>Guided semantic draw on the device: the conditional row's top-K candidates get the CFG blend, the rest -inf, and
    /// the device sampler writes the code into <paramref name="token"/>.</summary>
    private void DrawSemanticDevice(IBackend backend, Tensor hiddens, ulong token, ulong rngState)
    {
        int hidden = MiniMaxMusic3GlobalLm.HiddenSize;
        Tensor condRow = new Tensor(new TensorShape(1, hidden), DType.F32);
        Tensor uncondRow = new Tensor(new TensorShape(1, hidden), DType.F32);
        backend.SliceRows(condRow, hiddens, 0);
        backend.SliceRows(uncondRow, hiddens, 1);
        Tensor condLogits = _languageModel.SemanticLogits(backend, condRow);
        Tensor uncondLogits = _languageModel.SemanticLogits(backend, uncondRow);
        Tensor guided = new Tensor(new TensorShape(1, MiniMaxMusic3GlobalLm.SemanticVocabSize + 1), DType.F32);
        backend.CfgTopKGuided(guided, condLogits, uncondLogits, TopK, CfgScale);
        backend.SampleTopKInto(token, guided, TopK, 1f, 1f, 0f, rngState);
        condRow.Dispose();
        uncondRow.Dispose();
        condLogits.Dispose();
        uncondLogits.Dispose();
        guided.Dispose();
    }

    /// <summary>Residual draw of codebook <paramref name="index"/> from the conditional and unconditional rows of the depth head,
    /// blended as <c>g·c + (1−g)·u</c> by the device (the host forms <c>u + (c−u)·g</c>; the last bit differs).</summary>
    private void DrawResidualDevice(IBackend backend, Tensor states, int index, ulong token, ulong rngState)
    {
        int vocab = MiniMaxMusic3DepthDecoder.AudioVocabSize;
        Tensor logits = _depthDecoder.Head(backend, index, states);
        Tensor cond = new Tensor(new TensorShape(1, vocab), DType.F32);
        Tensor uncond = new Tensor(new TensorShape(1, vocab), DType.F32);
        backend.SliceRows(cond, logits, 0);
        backend.SliceRows(uncond, logits, 1);
        Tensor guided = new Tensor(new TensorShape(1, vocab), DType.F32);
        backend.Fill(guided, 0f);
        backend.CfgEulerStep(guided, cond, uncond, CfgScale, 1f);
        backend.SampleTopKInto(token, guided, TopK, 1f, 1f, 0f, rngState);
        logits.Dispose();
        cond.Dispose();
        uncond.Dispose();
        guided.Dispose();
    }

    /// <summary>The depth decoder's prefix for one frame: rows of <c>[hidden, semantic embedding]</c>, projected, forwarded, and the
    /// last step of each row kept as the states <c>[2, 1, H]</c>.</summary>
    private Tensor DepthPrefixStates(IBackend backend, Tensor hiddens, Tensor semEmbed, MiniMaxMusic3DepthCache cache)
    {
        int hidden = MiniMaxMusic3GlobalLm.HiddenSize;
        Tensor[] pairs = new Tensor[CfgRows];
        for (int row = 0; row < CfgRows; row++)
        {
            Tensor hiddenRow = new Tensor(new TensorShape(1, 1, hidden), DType.F32);
            backend.SliceRows(hiddenRow, hiddens, row);
            pairs[row] = new Tensor(new TensorShape(1, 2, hidden), DType.F32);
            backend.Concat(pairs[row], [hiddenRow, semEmbed], 1);
            hiddenRow.Dispose();
        }
        Tensor prefix = new Tensor(new TensorShape(CfgRows, 2, hidden), DType.F32);
        backend.Concat(prefix, pairs, 0);
        foreach (Tensor pair in pairs)
        {
            pair.Dispose();
        }
        using Tensor projected = _depthDecoder.Project(backend, prefix);
        prefix.Dispose();
        using Tensor block = _depthDecoder.Forward(backend, projected, cache);
        return LastStepsDevice(backend, block, 2);
    }

    /// <summary>One residual step: the gathered embedding of <paramref name="token"/> (codebook <paramref name="index"/>) for both CFG
    /// rows, projected and forwarded; returns the new states. Consumes <paramref name="previous"/>.</summary>
    private Tensor DepthStepStates(IBackend backend, int index, ulong token, MiniMaxMusic3DepthCache cache, Tensor previous)
    {
        int hidden = MiniMaxMusic3GlobalLm.HiddenSize;
        Tensor embedRow = new Tensor(new TensorShape(1, 1, hidden), DType.F32);
        backend.EmbedGatherDecodeStep(embedRow, _depthDecoder.ResidualRows(index), token);
        Tensor embedded = new Tensor(new TensorShape(CfgRows, 1, hidden), DType.F32);
        backend.Concat(embedded, [embedRow, embedRow], 0);
        embedRow.Dispose();
        previous.Dispose();
        using Tensor projected = _depthDecoder.Project(backend, embedded);
        embedded.Dispose();
        return _depthDecoder.Forward(backend, projected, cache);
    }

    /// <summary>The last step of each CFG row of <paramref name="block"/> <c>[rows, steps, H]</c>, as <c>[rows, 1, H]</c>, on the device.</summary>
    private static Tensor LastStepsDevice(IBackend backend, Tensor block, int steps)
    {
        int hidden = MiniMaxMusic3GlobalLm.HiddenSize;
        Tensor[] lasts = new Tensor[CfgRows];
        for (int row = 0; row < CfgRows; row++)
        {
            lasts[row] = new Tensor(new TensorShape(1, 1, hidden), DType.F32);
            backend.SliceRows(lasts[row], block, row * steps + steps - 1);
        }
        Tensor states = new Tensor(new TensorShape(CfgRows, 1, hidden), DType.F32);
        backend.Concat(states, lasts, 0);
        foreach (Tensor last in lasts)
        {
            last.Dispose();
        }
        block.Dispose();
        return states;
    }

    /// <summary>Reads the codes of one end-of-audio batch (frames <paramref name="from"/>..<paramref name="to"/>), keeps the frames
    /// before the first end-of-audio, and releases the batch's device vectors. Returns true once the generation is complete.</summary>
    private bool FlushDeviceBatch(IBackend backend, ulong[] tokens, int from, int to, List<(int FrameIndex, Tensor Vector)> pending,
        List<float[]> frames, int frameLimit, Action<int, int>? onFrame, int frameWidth)
    {
        int endAt = int.MaxValue;
        for (int frame = from; frame <= to; frame++)
        {
            int code = backend.ReadDeviceTokenId(tokens[(frame % DeviceFrameRing) * MiniMaxMusic3DepthDecoder.NumCodebooks]);
            if (code == MiniMaxMusic3GlobalLm.AudioEndLogitIndex)
            {
                endAt = frame;
                break;
            }
        }
        bool complete = endAt != int.MaxValue;
        foreach ((int frameIndex, Tensor vector) in pending)
        {
            if (frameIndex >= endAt) continue;
            if (frames.Count >= frameLimit) { complete = true; break; }
            float[] host = new float[frameWidth];
            vector.AsReadOnlySpan<float>().Slice(0, frameWidth).CopyTo(host);
            frames.Add(host);
            onFrame?.Invoke(frames.Count, frameLimit);
            if (frames.Count >= frameLimit) { complete = true; break; }
        }
        foreach ((int _, Tensor vector) in pending)
        {
            vector.Dispose();
        }
        pending.Clear();
        return complete;
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
    /// batched step returns. Reachable only through <c>numerics.mm3CfgBatch=false</c>.</summary>
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
            guided[i] = conditional[i] < threshold ? float.NegativeInfinity
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

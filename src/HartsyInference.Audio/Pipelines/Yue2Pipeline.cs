using HartsyInference.Audio.Models.Codecs.Oobleck;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Audio.Dsp;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Pipelines;

/// <summary>One YuE2 song request: what to write and how hard to think about it.</summary>
public sealed record Yue2Request
{
    /// <summary>Style tags — genre, instrumentation, mood, tempo.</summary>
    public string Style { get; init; } = "";

    /// <summary>Lyrics, with the usual <c>[verse]</c> / <c>[chorus]</c> section markers.</summary>
    public string Lyrics { get; init; } = "";

    /// <summary>How much symbolic planning to do before writing audio.</summary>
    public Yue2Cot Cot { get; init; } = Yue2Cot.Full;

    /// <summary>An externally supplied or edited ABC score. When set, the planning pass is skipped and this score
    /// is used verbatim. Ignored by <see cref="Yue2Cot.Off"/>.</summary>
    public string Abc { get; init; } = "";

    public long Seed { get; init; } = 831_001;

    /// <summary>Upper bound on song length. The model may stop earlier; it is a budget, not a target.</summary>
    public double MaxDurationSeconds { get; init; } = Yue2Protocol.MaxDurationSeconds;

    public Yue2Sampling AbcSampling { get; init; } = Yue2Sampling.Abc;

    public Yue2Sampling SemanticSampling { get; init; } = Yue2Sampling.Semantic;

    /// <summary>Guidance for the semantic pass. Null takes the release default for the chosen mode — 1.01 for
    /// <see cref="Yue2Cot.Off"/> (the only guided mode) and 1.0 otherwise.</summary>
    public float? CfgScale { get; init; }

    public int OdeSteps { get; init; } = 32;
}

/// <summary>What a YuE2 generation produced, beyond the audio itself.</summary>
public sealed record Yue2Result(float[] Left, float[] Right, int SampleRate, string? Abc, bool AbcTruncated, bool SemanticTruncated)
{
    public double DurationSeconds => Left.Length / (double)SampleRate;
}

/// <summary>Drives a full YuE2 song: plan a score, write semantic codec tokens under guidance, flow-match those
/// into acoustic latents, and decode 48 kHz stereo.</summary>
/// <remarks><para>The acoustic stage re-prefills the AR stack once per chunk over
/// <c>prefix ‖ codec ‖ MUSIC_END</c>. That end token is appended <b>unconditionally</b> — the sampler stops
/// <i>before</i> emitting it, and a run that hits its token budget never saw it at all, but the acoustic model was
/// trained with it present either way.</para>
/// <para>Sampling draws from the engine's own RNG, so a seed reproduces our output but not the reference
/// implementation's. Parity tests inject the reference's noise and run at temperature zero rather than trying to
/// match torch's generator.</para></remarks>
public sealed class Yue2Pipeline : IDisposable
{
    private readonly Yue2Config _config;
    private readonly Yue2Tokenizer _tokenizer;
    private readonly Yue2ArLm _ar;
    private readonly Yue2AcousticTransformer _acoustic;
    private readonly OobleckVae _vae;
    private int _disposed;

    public Yue2Pipeline(Yue2Config config, Yue2Tokenizer tokenizer, Yue2ArLm ar, Yue2AcousticTransformer acoustic, OobleckVae vae)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(ar);
        ArgumentNullException.ThrowIfNull(acoustic);
        ArgumentNullException.ThrowIfNull(vae);
        _config = config; _tokenizer = tokenizer; _ar = ar; _acoustic = acoustic; _vae = vae;
    }

    public int SampleRate => _config.SampleRate;

    /// <summary>Plans the score only, returning the ABC text. Useful on its own: the score can be edited and fed
    /// back through <see cref="Yue2Request.Abc"/>.</summary>
    public (string Abc, int[] Ids, bool Truncated) PlanScore(IBackend backend, Yue2Request request,
        Action<int, int>? onProgress = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Cot == Yue2Cot.Off) return ("", [], false);
        int[] promptIds = _tokenizer.Encode(Yue2Protocol.PromptText(request.Cot, request.Style, request.Lyrics));
        int[] prefix = Yue2Protocol.TokenPrefix(request.Cot, promptIds, null);
        request.AbcSampling.Validate();
        (List<int> ids, bool truncated) = Sample(backend, prefix, null, 1f, request.AbcSampling,
            Yue2Phase.Abc, legacyOff: false, request.Seed, onProgress, cancel);
        return (_tokenizer.Decode(ids), [.. ids], truncated);
    }

    /// <summary>Generates a complete song.</summary>
    public Yue2Result Generate(IBackend backend, Yue2Request request,
        Action<string, int, int>? onProgress = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(request);
        ThrowIfDisposed();
        request.AbcSampling.Validate();
        request.SemanticSampling.Validate();

        Yue2Cot cot = request.Cot;
        int[] promptIds = _tokenizer.Encode(Yue2Protocol.PromptText(cot, request.Style, request.Lyrics));
        int[] instructionIds = _tokenizer.Encode(Yue2Protocol.Instruction(cot));

        // ── 1. The score ──
        string? abcText = null;
        int[] abcIds = [];
        bool abcTruncated = false;
        if (cot != Yue2Cot.Off)
        {
            if (request.Abc.Trim().Length > 0)
            {
                abcText = request.Abc;
                abcIds = _tokenizer.Encode(request.Abc);
            }
            else
            {
                int[] planningPrefix = Yue2Protocol.TokenPrefix(cot, promptIds, null);
                (List<int> planned, bool truncated) = Sample(backend, planningPrefix, null, 1f, request.AbcSampling,
                    Yue2Phase.Abc, legacyOff: false, request.Seed,
                    onProgress is null ? null : (done, total) => onProgress("plan", done, total), cancel);
                abcIds = [.. planned];
                abcText = _tokenizer.Decode(planned);
                abcTruncated = truncated;
            }
        }

        // ── 2. Semantic codec tokens ──
        int[] prefix = Yue2Protocol.TokenPrefix(cot, promptIds, cot == Yue2Cot.Off ? null : abcIds);
        float cfgScale = request.CfgScale ?? Yue2Protocol.DefaultCfgScale(cot);
        int[]? negative = cfgScale == 1f ? null
            : Yue2Protocol.NegativePrefix(cot, instructionIds, cot == Yue2Cot.Off ? null : abcIds);

        Yue2Sampling semanticSampling = request.SemanticSampling with
        {
            MaxTokens = Math.Min(request.SemanticSampling.MaxTokens, Yue2Protocol.TokensForSeconds(request.MaxDurationSeconds)),
        };
        semanticSampling = semanticSampling with { MinTokens = Math.Min(semanticSampling.MinTokens, semanticSampling.MaxTokens) };
        (List<int> semantic, bool semanticTruncated) = Sample(backend, prefix, negative, cfgScale, semanticSampling,
            Yue2Phase.Semantic, cot == Yue2Cot.Off, request.Seed,
            onProgress is null ? null : (done, total) => onProgress("semantic", done, total), cancel);
        if (semantic.Count == 0)
            throw new InvalidOperationException("YuE2 produced no codec tokens; try a longer duration or a different seed.");

        int[] codec = new int[semantic.Count];
        for (int i = 0; i < codec.Length; i++) codec[i] = semantic[i] - Yue2Protocol.CodecOffset;

        // ── 3. Acoustic flow ──
        float[] latents = Synthesize(backend, prefix, codec, request, onProgress, cancel);

        // ── 4. Waveform ──
        (float[] left, float[] right) = Decode(backend, latents, codec.Length);
        return new Yue2Result(left, right, SampleRate, abcText, abcTruncated, semanticTruncated);
    }

    /// <summary>Runs one autoregressive pass, optionally under classifier-free guidance.</summary>
    /// <remarks>Guidance keeps two independent caches rather than a padded batch: the positive and negative
    /// prefixes have different lengths, and the released implementation prefills and decodes each separately.</remarks>
    private (List<int> Ids, bool Truncated) Sample(IBackend backend, int[] prefix, int[]? negative, float cfgScale,
        Yue2Sampling sampling, Yue2Phase phase, bool legacyOff, long seed,
        Action<int, int>? onProgress, CancellationToken cancel)
    {
        int longest = Math.Max(prefix.Length, negative?.Length ?? 0);
        if (longest + sampling.MaxTokens > Yue2Protocol.Context)
        {
            throw new InvalidOperationException(
                $"YuE2's prompt ({longest} tokens) plus its {sampling.MaxTokens}-token budget exceeds the "
                + $"{Yue2Protocol.Context}-token context. Shorten the lyrics or the duration.");
        }

        using IKvCache positiveCache = _ar.CreateCache(prefix.Length + sampling.MaxTokens);
        using IKvCache? negativeCache = negative is null ? null : _ar.CreateCache(negative.Length + sampling.MaxTokens);
        float[] conditional = Yue2ArLm.AllocateLogits();
        float[]? unconditional = negative is null ? null : Yue2ArLm.AllocateLogits();
        float[] scores = Yue2ArLm.AllocateLogits();

        _ar.Forward(backend, prefix, posStart: 0, positiveCache, conditional);
        if (negative is not null) _ar.Forward(backend, negative, posStart: 0, negativeCache!, unconditional!);

        uint rng = DeterministicRng.Seed(unchecked((int)seed));
        List<int> history = [];
        int end = phase == Yue2Phase.Abc ? Yue2Protocol.AbcEnd : Yue2Protocol.MusicEnd;
        bool truncated = true;

        for (int step = 0; step < sampling.MaxTokens; step++)
        {
            cancel.ThrowIfCancellationRequested();
            if (unconditional is null)
            {
                conditional.AsSpan().CopyTo(scores);
            }
            else
            {
                for (int i = 0; i < scores.Length; i++)
                    scores[i] = unconditional[i] + cfgScale * (conditional[i] - unconditional[i]);
            }

            Yue2LogitProcessor.Apply(scores, sampling, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(history),
                step, phase, legacyOff);
            int token = sampling.Temperature == 0 ? ArgMax(scores) : Draw(scores, ref rng);
            onProgress?.Invoke(step + 1, sampling.MaxTokens);
            if (token == end) { truncated = false; break; }
            history.Add(token);

            if (step + 1 >= sampling.MaxTokens) break;
            _ar.Forward(backend, [token], prefix.Length + step, positiveCache, conditional);
            if (negative is not null) _ar.Forward(backend, [token], negative!.Length + step, negativeCache!, unconditional!);
        }
        return (history, truncated);
    }

    /// <summary>Flow-matches every acoustic chunk, returning frame-major <c>[frames, 64]</c> latents.</summary>
    private float[] Synthesize(IBackend backend, int[] prefix, int[] codec, Yue2Request request,
        Action<string, int, int>? onProgress, CancellationToken cancel)
    {
        int latentDim = _config.LatentDim;
        // One draw for the whole song, sliced per chunk — drawing per chunk would change every result.
        float[] noise = new float[codec.Length * latentDim];
        // A separate stream from the sampler's, so changing the token budget does not reshuffle the noise.
        uint noiseRng = DeterministicRng.Seed(unchecked((int)request.Seed) ^ 0x5F37_2B19);
        for (int i = 0; i < noise.Length; i++) noise[i] = DeterministicRng.NextGaussian(ref noiseRng);

        (int Start, int End)[] ranges = Yue2Protocol.ChunkRanges(codec.Length, prefix.Length);
        float[] latents = new float[noise.Length];
        int stepsDone = 0, stepsTotal = ranges.Length * request.OdeSteps;

        foreach ((int start, int end) in ranges)
        {
            cancel.ThrowIfCancellationRequested();

            // prefix + this chunk's codec + MUSIC_END, which the acoustic model always expects to be present.
            int[] arTokens = new int[prefix.Length + (end - start) + 1];
            prefix.CopyTo(arTokens, 0);
            for (int i = start; i < end; i++) arTokens[prefix.Length + (i - start)] = codec[i] + Yue2Protocol.CodecOffset;
            arTokens[^1] = Yue2Protocol.MusicEnd;

            // Sized to EXACTLY the prefix length: the cache hands back its whole capacity buffer, and the acoustic
            // stack concatenates it onto its own keys, so any unpopulated tail would be attended over.
            using IKvCache cache = _ar.CreateCache(arTokens.Length);
            float[] logits = Yue2ArLm.AllocateLogits();
            _ar.Forward(backend, arTokens, posStart: 0, cache, logits);
            (Tensor Key, Tensor Value)[] arPrefix = _ar.ExportPrefix(cache);

            int offset = start * latentDim, length = (end - start) * latentDim;
            int baseStep = stepsDone;
            float[] chunk = Yue2FlowSolver.Solve(backend, _acoustic, noise.AsSpan(offset, length),
                arPrefix, arTokens.Length, request.OdeSteps,
                onProgress is null ? null : (done, _) => onProgress("acoustic", baseStep + done, stepsTotal), cancel);
            chunk.CopyTo(latents, offset);
            stepsDone += request.OdeSteps;
        }
        return latents;
    }

    /// <summary>Decodes frame-major latents to stereo channels.</summary>
    private (float[] Left, float[] Right) Decode(IBackend backend, float[] latents, int frames)
    {
        int latentDim = _config.LatentDim;
        using Tensor input = new(new TensorShape(1, latentDim, frames), DType.F32);
        Span<float> destination = input.AsSpan<float>();
        for (int f = 0; f < frames; f++)
        {
            for (int c = 0; c < latentDim; c++) destination[c * frames + f] = latents[f * latentDim + c];
        }

        using Tensor audio = _vae.Decode(backend, input);
        int samples = (int)audio.Shape[audio.Shape.Rank - 1];
        ReadOnlySpan<float> decoded = audio.AsReadOnlySpan<float>();
        float[] left = new float[samples], right = new float[samples];
        for (int i = 0; i < samples; i++)
        {
            left[i] = Math.Clamp(decoded[i], -1f, 1f);
            right[i] = Math.Clamp(decoded[samples + i], -1f, 1f);
        }
        return (left, right);
    }

    private static int ArgMax(ReadOnlySpan<float> scores)
    {
        int best = 0;
        float bestValue = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++)
        {
            if (scores[i] > bestValue) { bestValue = scores[i]; best = i; }
        }
        return best;
    }

    /// <summary>Multinomial draw over the already-filtered distribution.</summary>
    private static int Draw(ReadOnlySpan<float> scores, ref uint rng)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++)
        {
            if (scores[i] > max) max = scores[i];
        }
        double total = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            if (float.IsFinite(scores[i])) total += Math.Exp(scores[i] - max);
        }
        double target = DeterministicRng.NextUniform(ref rng) * total, running = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            if (!float.IsFinite(scores[i])) continue;
            running += Math.Exp(scores[i] - max);
            if (running >= target) return i;
        }
        return ArgMax(scores);
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _ar.Dispose();
        _acoustic.Dispose();
    }
}

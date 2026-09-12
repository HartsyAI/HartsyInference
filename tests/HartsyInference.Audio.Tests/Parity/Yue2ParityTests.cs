using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using HartsyInference.Audio.Models.Codecs.Oobleck;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for YuE2 against the official <c>yue2_infer</c> wheel running the original
/// <c>m-a-p/YuE2-3B</c> weights. The ladder is ordered so each gate can only fail for its own reason:
/// <list type="bullet">
///   <item><b>A1/A2</b> — the embedded tokenizer's ids and the exact prefix each <c>cot</c> mode builds.</item>
///   <item><b>A3</b> — prefill plus eight greedy decode steps, temperature 0, so no RNG is in the comparison.</item>
///   <item><b>A4</b> — <see cref="Yue2LogitProcessor"/> on canned logits, including both <c>legacy_off</c> branches.</item>
///   <item><b>A5</b> — the per-layer post-RoPE K/V the acoustic stack attends over.</item>
/// </list>
/// Gated on <c>YUE2_CHECKPOINT</c> (the Comfy-Org single file) and <c>YUE2_REF_DIR</c>
/// (<c>dump_yue2_reference.py</c> output). See <c>docs/Research/YUE2_ARCHITECTURE.md</c>.</summary>
public sealed class Yue2ParityTests
{
    private readonly ITestOutputHelper _out;
    public Yue2ParityTests(ITestOutputHelper o) => _out = o;

    private static string? Checkpoint => Environment.GetEnvironmentVariable("YUE2_CHECKPOINT");
    private static string? ReferenceDir => Environment.GetEnvironmentVariable("YUE2_REF_DIR");

    private static bool Gated(out string checkpoint, out string referenceDir)
    {
        checkpoint = Checkpoint ?? "";
        referenceDir = ReferenceDir ?? "";
        return checkpoint.Length == 0 || !File.Exists(checkpoint)
            || referenceDir.Length == 0 || !Directory.Exists(referenceDir);
    }

    /// <summary>CUDA when a device and PTX are present. The CUDA path is different code — fused QKV GEMV, graph
    /// decode, a BF16 head GEMM — so a CPU-only parity run proves nothing about the backend that ships.
    /// <c>YUE2_FORCE_CPU=1</c> pins the host path.</summary>
    private static IBackend CreateBackend(out string name)
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Environment.GetEnvironmentVariable("YUE2_FORCE_CPU") != "1" && Directory.Exists(ptxDir))
        {
            try
            {
                name = "CUDA";
                return new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
            }
            catch (Exception)
            {
                // fall through to the host path
            }
        }
        name = "CPU";
        return new CpuBackend();   // tier-lint: guarded
    }

    private static JsonElement Protocol(string referenceDir)
        => JsonDocument.Parse(File.ReadAllText(Path.Combine(referenceDir, "protocol.json"))).RootElement;

    private static int[] Ints(JsonElement e) => [.. e.EnumerateArray().Select(x => x.GetInt32())];

    [Fact]
    [Trait("Category", "Integration")]
    public void Tokenizer_And_Prefixes_MatchReference()
    {
        if (Gated(out string checkpoint, out string referenceDir)) return;

        using SafeTensorsLoader loader = new();
        loader.Load(checkpoint);
        Assert.True(Yue2CheckpointConverter.IsComfyCheckpoint(loader.Descriptors));
        using Tensor tokenizerBlob = loader.GetTensor(Yue2CheckpointConverter.TokenizerKey);
        Yue2Tokenizer tokenizer = new(tokenizerBlob.AsReadOnlySpan<byte>());

        JsonElement protocol = Protocol(referenceDir);
        string style = protocol.GetProperty("style").GetString()!;
        string lyrics = protocol.GetProperty("lyrics").GetString()!;

        foreach ((string name, Yue2Cot cot) in new[] { ("off", Yue2Cot.Off), ("melody", Yue2Cot.Melody), ("full", Yue2Cot.Full) })
        {
            JsonElement mode = protocol.GetProperty("modes").GetProperty(name);

            // A1: the prompt body and its ids.
            Assert.Equal(mode.GetProperty("prompt_text").GetString(), Yue2Protocol.PromptText(cot, style, lyrics));
            int[] promptIds = tokenizer.Encode(Yue2Protocol.PromptText(cot, style, lyrics));
            Assert.Equal(Ints(mode.GetProperty("prompt_ids")), promptIds);
            int[] instructionIds = tokenizer.Encode(Yue2Protocol.Instruction(cot));
            Assert.Equal(Ints(mode.GetProperty("instruction_ids")), instructionIds);
            Assert.Equal(mode.GetProperty("guidance").GetSingle(), Yue2Protocol.DefaultCfgScale(cot), 6);

            // A2: the prefixes, which are what the model actually reads.
            Assert.Equal(Ints(mode.GetProperty("prefix_planning")), Yue2Protocol.TokenPrefix(cot, promptIds, null));
            if (cot == Yue2Cot.Off)
            {
                Assert.Equal(Ints(mode.GetProperty("negative")), Yue2Protocol.NegativePrefix(cot, instructionIds, null));
            }
            else
            {
                int[] abcIds = tokenizer.Encode(mode.GetProperty("abc_text").GetString()!);
                Assert.Equal(Ints(mode.GetProperty("abc_ids")), abcIds);
                Assert.Equal(Ints(mode.GetProperty("prefix_with_abc")), Yue2Protocol.TokenPrefix(cot, promptIds, abcIds));
                Assert.Equal(Ints(mode.GetProperty("negative_with_abc")), Yue2Protocol.NegativePrefix(cot, instructionIds, abcIds));
            }
            _out.WriteLine($"{name}: prompt={promptIds.Length} ids, prefix={Ints(mode.GetProperty("prefix_planning")).Length}");
        }

        // The chunk split the acoustic stage runs under.
        (int Start, int End)[] ranges = Yue2Protocol.ChunkRanges(1000, 300);
        JsonElement expected = protocol.GetProperty("chunk_ranges_1000f_300p");
        Assert.Equal(expected.GetArrayLength(), ranges.Length);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ArLm_GreedyLogits_MatchReference()
    {
        if (Gated(out string checkpoint, out string referenceDir)) return;
        string fixtures = Path.Combine(referenceDir, "yue2_fixtures.safetensors");
        if (!File.Exists(fixtures)) return;

        using SafeTensorsLoader loader = new();
        loader.Load(checkpoint);
        using Yue2Weights weights = Yue2CheckpointConverter.Convert(loader.GetAllTensors());
        _out.WriteLine($"converted: ar={weights.Ar.Count} nar={weights.Nar.Count} vae={weights.Vae.Count}");

        using SafeTensorsLoader reference = new();
        reference.Load(fixtures);
        using Tensor prefixTensor = reference.GetTensor("ar_greedy.prefix");
        using Tensor expectedLogits = reference.GetTensor("ar_greedy.logits");
        using Tensor expectedTokens = reference.GetTensor("ar_greedy.tokens");
        int[] prefix = [.. prefixTensor.AsReadOnlySpan<int>()];
        int steps = expectedTokens.AsReadOnlySpan<int>().Length;

        using Yue2ArLm lm = new(Yue2Config.V1);
        lm.LoadWeights(weights.Ar);
        using IBackend backend = CreateBackend(out string backendName);
        _out.WriteLine($"backend: {backendName}");
        using IKvCache cache = lm.CreateCache(prefix.Length + steps + 1);

        // The semantic pass projects only its own head window, so the fixture's full-vocabulary rows are compared
        // over the matching slice — which is every logit this phase's sampler can reach.
        (int baseId, int width) = Yue2Protocol.Window(Yue2Phase.Semantic);
        float[] logits = Yue2ArLm.AllocateLogits(Yue2Phase.Semantic);
        lm.Forward(backend, prefix, posStart: 0, cache, logits, Yue2Phase.Semantic);
        ReadOnlySpan<float> expected = expectedLogits.AsReadOnlySpan<float>();
        AssertLogitsAgree(expected.Slice(baseId, width), logits, "prefill");

        List<int> produced = [];
        for (int step = 0; step < steps; step++)
        {
            // Greedy over the codec span, mirroring the reference dump's temperature-0 mask.
            int best = ArgMaxCodec(logits);
            produced.Add(best);
            lm.Forward(backend, [best], posStart: prefix.Length + step, cache, logits, Yue2Phase.Semantic);
            AssertLogitsAgree(expected.Slice((step + 1) * Yue2Protocol.VocabSize + baseId, width), logits, $"step {step}");
        }
        Assert.Equal(expectedTokens.AsReadOnlySpan<int>().ToArray(), produced);
        _out.WriteLine($"greedy tokens matched: [{string.Join(", ", produced.Take(4))}, …]");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ArKvPrefix_MatchesReference()
    {
        if (Gated(out string checkpoint, out string referenceDir)) return;
        string fixtures = Path.Combine(referenceDir, "yue2_fixtures.safetensors");
        if (!File.Exists(fixtures)) return;

        using SafeTensorsLoader loader = new();
        loader.Load(checkpoint);
        using Yue2Weights weights = Yue2CheckpointConverter.Convert(loader.GetAllTensors());

        using SafeTensorsLoader reference = new();
        reference.Load(fixtures);
        using Tensor tokensTensor = reference.GetTensor("ar_kv.ar_tokens");
        using Tensor expectedKeys = reference.GetTensor("ar_kv.keys");
        using Tensor expectedValues = reference.GetTensor("ar_kv.values");
        int[] arTokens = [.. tokensTensor.AsReadOnlySpan<int>()];

        // The reference's chunk is prefix + codec + [MUSIC_END]. The decode loop breaks on the end token BEFORE
        // feeding it, so the acoustic stage must run one extra forward over MUSIC_END — unconditionally, including
        // on a run that hit its token budget instead of ending naturally.
        Assert.Equal(Yue2Protocol.MusicEnd, arTokens[^1]);

        using Yue2ArLm lm = new(Yue2Config.V1);
        lm.LoadWeights(weights.Ar);
        using IBackend backend = CreateBackend(out string backendName);
        _out.WriteLine($"backend: {backendName}, prefix={arTokens.Length} tokens");
        using IKvCache cache = lm.CreateCache(arTokens.Length + 1);
        float[] logits = Yue2ArLm.AllocateLogits(Yue2Phase.Semantic);
        lm.Forward(backend, arTokens, posStart: 0, cache, logits, Yue2Phase.Semantic);

        (Tensor Key, Tensor Value)[] prefix = lm.ExportPrefix(cache);
        Assert.Equal(lm.NumLayers, prefix.Length);

        int tokens = arTokens.Length, heads = lm.KvHeads, dim = lm.HeadDim;
        ReadOnlySpan<float> keys = expectedKeys.AsReadOnlySpan<float>();
        ReadOnlySpan<float> values = expectedValues.AsReadOnlySpan<float>();
        double worst = 1.0;
        for (int layer = 0; layer < prefix.Length; layer++)
        {
            // Ours is [1, heads, capacity, dim]; the fixture is [layers, tokens, heads, dim].
            worst = Math.Min(worst, CompareTransposed(prefix[layer].Key, keys, layer, tokens, heads, dim, $"K L{layer}"));
            worst = Math.Min(worst, CompareTransposed(prefix[layer].Value, values, layer, tokens, heads, dim, $"V L{layer}"));
        }
        _out.WriteLine($"lowest per-layer KV correlation across {prefix.Length} layers: {worst:F6}");
    }

    /// <summary>Compares a <c>[1, heads, tokens, dim]</c> cache slab against the reference's
    /// <c>[layers, tokens, heads, dim]</c> dump, returning the largest absolute deviation.</summary>
    /// <summary>Compares a <c>[1, heads, capacity, dim]</c> cache slab against the reference's
    /// <c>[layers, tokens, heads, dim]</c> dump, returning the correlation.</summary>
    /// <remarks>The bar is correlation plus normalised RMS, not worst-element deviation. The reference runs the
    /// whole forward in BF16 and caches BF16; we run F32 kernels off the same BF16 weights, so the two hidden
    /// states drift apart layer by layer — by L25 the worst single element differs by ~2% of the tensor scale,
    /// while the logits those keys produce still agree to correlation 0.99998 with an identical argmax. A layout,
    /// head-order or RoPE fault destroys correlation outright, which is what this is here to catch.</remarks>
    private double CompareTransposed(Tensor actual, ReadOnlySpan<float> expected, int layer,
        int tokens, int heads, int dim, string label)
    {
        ReadOnlySpan<float> ours = actual.AsReadOnlySpan<float>();
        // The cache hands back its WHOLE capacity buffer, not a length-trimmed view, so the token stride is the
        // capacity rather than the populated length.
        int capacity = (int)actual.Shape[2];
        long layerBase = (long)layer * tokens * heads * dim;
        int count = heads * tokens * dim;
        double sumA = 0, sumB = 0;
        for (int h = 0; h < heads; h++)
        {
            for (int t = 0; t < tokens; t++)
            {
                for (int d = 0; d < dim; d++)
                {
                    sumA += ours[(h * capacity + t) * dim + d];
                    sumB += expected[(int)(layerBase + ((long)t * heads + h) * dim + d)];
                }
            }
        }
        double meanA = sumA / count, meanB = sumB / count;
        double cov = 0, varA = 0, varB = 0, sumSquaredError = 0;
        for (int h = 0; h < heads; h++)
        {
            for (int t = 0; t < tokens; t++)
            {
                for (int d = 0; d < dim; d++)
                {
                    double a = ours[(h * capacity + t) * dim + d];
                    double b = expected[(int)(layerBase + ((long)t * heads + h) * dim + d)];
                    cov += (a - meanA) * (b - meanB);
                    varA += (a - meanA) * (a - meanA);
                    varB += (b - meanB) * (b - meanB);
                    sumSquaredError += (a - b) * (a - b);
                }
            }
        }
        double correlation = cov / Math.Sqrt(varA * varB);
        double normalisedRms = Math.Sqrt(sumSquaredError / count) / Math.Sqrt(varB / count);
        Assert.True(correlation > 0.999, $"{label}: correlation {correlation:F6} below 0.999");
        Assert.True(normalisedRms < 0.05, $"{label}: normalised RMS {normalisedRms:P3} exceeds 5%");
        return correlation;
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void AcousticVelocityAndOde_MatchReference()
    {
        if (Gated(out string checkpoint, out string referenceDir)) return;
        string fixtures = Path.Combine(referenceDir, "yue2_fixtures.safetensors");
        if (!File.Exists(fixtures)) return;

        using SafeTensorsLoader loader = new();
        loader.Load(checkpoint);
        using Yue2Weights weights = Yue2CheckpointConverter.Convert(loader.GetAllTensors());

        using SafeTensorsLoader reference = new();
        reference.Load(fixtures);
        using Tensor arTokensTensor = reference.GetTensor("ar_kv.ar_tokens");
        using Tensor noiseTensor = reference.GetTensor("nar.noise");
        using Tensor rawTimestepTensor = reference.GetTensor("nar.raw_t");
        using Tensor expectedVelocity = reference.GetTensor("nar.velocity");
        using Tensor expectedLatents = reference.GetTensor("nar.latents");
        using Tensor odeStepsTensor = reference.GetTensor("nar.ode_steps");
        int[] arTokens = [.. arTokensTensor.AsReadOnlySpan<int>()];
        ReadOnlySpan<float> noise = noiseTensor.AsReadOnlySpan<float>();
        float rawTimestep = rawTimestepTensor.AsReadOnlySpan<float>()[0];
        int odeSteps = odeStepsTensor.AsReadOnlySpan<int>()[0];

        using IBackend backend = CreateBackend(out string backendName);
        _out.WriteLine($"backend: {backendName}, ar={arTokens.Length} tokens, {noise.Length / 64} latent frames, {odeSteps} ODE steps");

        // The prefix cache is sized to EXACTLY the AR length so its buffers carry no unpopulated tail — the
        // acoustic stack concatenates them straight onto its own keys.
        using Yue2ArLm lm = new(Yue2Config.V1);
        lm.LoadWeights(weights.Ar);
        using IKvCache cache = lm.CreateCache(arTokens.Length);
        float[] logits = Yue2ArLm.AllocateLogits(Yue2Phase.Semantic);
        lm.Forward(backend, arTokens, posStart: 0, cache, logits, Yue2Phase.Semantic);
        (Tensor Key, Tensor Value)[] arPrefix = lm.ExportPrefix(cache);

        using Yue2AcousticTransformer acoustic = new(Yue2Config.V1);
        acoustic.LoadWeights(weights.Nar);

        // B1: one velocity evaluation on the reference's own state and timestep.
        float[] velocity = new float[noise.Length];
        acoustic.Velocity(backend, noise, rawTimestep, arPrefix, arTokens.Length, velocity);
        AssertAgree(expectedVelocity.AsReadOnlySpan<float>(), velocity, "velocity at t=1");

        // B2: the full 32-step midpoint solve from the same noise.
        float[] latents = Yue2FlowSolver.Solve(backend, acoustic, noise, arPrefix, arTokens.Length, odeSteps);
        AssertAgree(expectedLatents.AsReadOnlySpan<float>(), latents, $"{odeSteps}-step midpoint solve");
    }

    /// <summary>Correlation plus normalised RMS, the same bar the KV gate uses and for the same reason: the
    /// reference runs BF16 where we run F32.</summary>
    private void AssertAgree(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        double sumExpected = 0, sumActual = 0;
        for (int i = 0; i < expected.Length; i++) { sumExpected += expected[i]; sumActual += actual[i]; }
        double meanExpected = sumExpected / expected.Length, meanActual = sumActual / actual.Length;
        double cov = 0, varExpected = 0, varActual = 0, squaredError = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double a = expected[i] - meanExpected, b = actual[i] - meanActual;
            cov += a * b; varExpected += a * a; varActual += b * b;
            squaredError += (expected[i] - actual[i]) * (expected[i] - actual[i]);
        }
        double correlation = cov / Math.Sqrt(varExpected * varActual);
        double normalisedRms = Math.Sqrt(squaredError / expected.Length) / Math.Sqrt(varExpected / expected.Length);
        _out.WriteLine($"{label}: corr={correlation:F6} nrms={normalisedRms:P3}");
        Assert.True(correlation > 0.999, $"{label}: correlation {correlation:F6} below 0.999");
        Assert.True(normalisedRms < 0.05, $"{label}: normalised RMS {normalisedRms:P3} exceeds 5%");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void VaeDecode_MatchesReference()
    {
        if (Gated(out string checkpoint, out string referenceDir)) return;
        string fixtures = Path.Combine(referenceDir, "yue2_fixtures.safetensors");
        if (!File.Exists(fixtures)) return;

        using SafeTensorsLoader loader = new();
        loader.Load(checkpoint);
        using Yue2Weights weights = Yue2CheckpointConverter.Convert(loader.GetAllTensors());

        using SafeTensorsLoader reference = new();
        reference.Load(fixtures);
        using Tensor latents = reference.GetTensor("vae.latents");
        using Tensor expectedAudio = reference.GetTensor("vae.audio");
        using Tensor sampleRateTensor = reference.GetTensor("vae.sample_rate");
        int frames = (int)latents.Shape[0], channels = 64;

        OobleckConfig config = OobleckConfig.Yue2;
        Assert.Equal(1920, config.HopLength);
        Assert.Equal(sampleRateTensor.AsReadOnlySpan<int>()[0], config.SamplingRate);

        OobleckVae vae = new(config);
        vae.LoadWeights(weights.Vae);
        using IBackend backend = CreateBackend(out string backendName);

        // The decoder wants [1, channels, frames]; the fixture is frame-major [frames, channels].
        using Tensor input = new(new TensorShape(1, channels, frames), DType.F32);
        ReadOnlySpan<float> source = latents.AsReadOnlySpan<float>();
        Span<float> destination = input.AsSpan<float>();
        for (int f = 0; f < frames; f++)
        {
            for (int c = 0; c < channels; c++) destination[c * frames + f] = source[f * channels + c];
        }

        using Tensor audio = vae.Decode(backend, input);
        int produced = (int)audio.Shape[audio.Shape.Rank - 1];
        int expectedSamples = (int)expectedAudio.Shape[0];
        _out.WriteLine($"backend: {backendName}, {frames} frames -> {produced} samples (reference {expectedSamples}, frames*hop would be {frames * config.HopLength})");

        // Upstream leaves output_padding at zero, so the stride-5 stage loses a frame: the decode is SHORTER than
        // frames * 1920. A port that follows ComfyUI's added output_padding lands on 15360 here instead.
        Assert.Equal(expectedSamples, produced);
        Assert.True(produced < frames * config.HopLength, "the odd-stride stage should shorten the decode");

        // The decoder emits [1, channels, samples]; the fixture is [samples, channels].
        ReadOnlySpan<float> ours = audio.AsReadOnlySpan<float>();
        ReadOnlySpan<float> theirs = expectedAudio.AsReadOnlySpan<float>();
        float[] interleaved = new float[produced * 2];
        for (int sample = 0; sample < produced; sample++)
        {
            interleaved[sample * 2] = ours[sample];
            interleaved[sample * 2 + 1] = ours[produced + sample];
        }
        AssertAgree(theirs[..(produced * 2)], interleaved, "VAE decode");
    }

    [Fact]
    public void LogitProcessor_ReproducesReleaseRules()
    {
        // A4 runs on canned logits, so it needs no checkpoint and no GPU.
        const int vocab = Yue2Protocol.CodecOffset + Yue2Protocol.CodecSize;
        float[] scores = new float[vocab];
        // Strictly increasing, so no two logits tie. Ties are kept wholesale by the top-k threshold (which is what
        // torch.topk does), and a tied fixture would make the "exactly k survive" assertion meaningless.
        for (int i = 0; i < vocab; i++) scores[i] = i * 1e-4f;

        // The semantic phase may draw only codec ids; everything else is masked out.
        float[] semantic = [.. scores];
        Yue2LogitProcessor.Apply(semantic, Yue2Sampling.Semantic with { MinTokens = 0, TopK = 5, TopP = 1f, RepetitionPenalty = 1f },
            history: [], step: 0, Yue2Phase.Semantic, legacyOff: false, baseId: 0);
        for (int i = 0; i < Yue2Protocol.CodecOffset; i++)
        {
            if (i != Yue2Protocol.MusicEnd) Assert.True(float.IsNegativeInfinity(semantic[i]), $"id {i} should be masked");
        }
        Assert.Equal(5, semantic.Count(float.IsFinite) - (float.IsFinite(semantic[Yue2Protocol.MusicEnd]) ? 1 : 0));

        // Under min_tokens the end token cannot be drawn.
        float[] gated = [.. scores];
        Yue2LogitProcessor.Apply(gated, Yue2Sampling.Semantic with { MinTokens = 200, TopK = 5, TopP = 1f, RepetitionPenalty = 1f },
            history: [], step: 3, Yue2Phase.Semantic, legacyOff: false, baseId: 0);
        Assert.True(float.IsNegativeInfinity(gated[Yue2Protocol.MusicEnd]));

        // The windowed penalty is penalty^count over the window, and only over the window.
        float[] penalised = [.. scores];
        int repeated = Yue2Protocol.CodecOffset + 5;
        int stale = Yue2Protocol.CodecOffset + 6;
        int[] history = [stale, .. Enumerable.Repeat(repeated, 3)];
        Yue2LogitProcessor.Apply(penalised, Yue2Sampling.Semantic with { MinTokens = 0, TopK = int.MaxValue, TopP = 1f, RepetitionPenalty = 2f, PenaltyWindow = 3 },
            history, step: 4, Yue2Phase.Semantic, legacyOff: false, baseId: 0);
        float raw = scores[repeated];
        Assert.Equal(raw > 0 ? raw / 8f : raw * 8f, penalised[repeated], 4);
        Assert.Equal(scores[stale], penalised[stale], 4);   // outside the 3-wide window

        // The historical off path keeps three sorted entries out of the nucleus cut instead of one.
        float[] modern = [.. scores], legacy = [.. scores];
        Yue2Sampling nucleus = Yue2Sampling.Semantic with { MinTokens = 0, TopK = int.MaxValue, TopP = 1e-6f, RepetitionPenalty = 1f, Temperature = 1f };
        Yue2LogitProcessor.Apply(modern, nucleus, [], 0, Yue2Phase.Semantic, legacyOff: false, baseId: 0);
        Yue2LogitProcessor.Apply(legacy, nucleus, [], 0, Yue2Phase.Semantic, legacyOff: true, baseId: 0);
        Assert.True(legacy.Count(float.IsFinite) > modern.Count(float.IsFinite));
    }

    /// <summary>Greedy over the codec span of a semantic-window logit buffer; returns the absolute id.</summary>
    private static int ArgMaxCodec(ReadOnlySpan<float> window)
    {
        int baseId = Yue2Protocol.Window(Yue2Phase.Semantic).Base;
        int start = Yue2Protocol.CodecOffset - baseId;
        int best = start;
        float bestValue = float.NegativeInfinity;
        for (int i = start; i < start + Yue2Protocol.CodecSize; i++)
        {
            if (window[i] > bestValue) { bestValue = window[i]; best = i; }
        }
        return best + baseId;
    }

    /// <summary>The reference runs BF16 weights through torch's kernels and we run our own, so the bar is
    /// correlation plus argmax agreement rather than bitwise equality.</summary>
    private void AssertLogitsAgree(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        double sumExpected = 0, sumActual = 0;
        for (int i = 0; i < expected.Length; i++) { sumExpected += expected[i]; sumActual += actual[i]; }
        double meanExpected = sumExpected / expected.Length, meanActual = sumActual / actual.Length;
        double cov = 0, varExpected = 0, varActual = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double a = expected[i] - meanExpected, b = actual[i] - meanActual;
            cov += a * b; varExpected += a * a; varActual += b * b;
        }
        double correlation = cov / Math.Sqrt(varExpected * varActual);
        _out.WriteLine($"{label}: corr={correlation:F6}");
        Assert.True(correlation > 0.999, $"{label}: logit correlation {correlation:F6} below 0.999");
    }
}

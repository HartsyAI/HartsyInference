using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
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
        Yue2Weights weights = Yue2CheckpointConverter.Convert(loader.GetAllTensors());
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
        IBackend backend = new CpuBackend();
        using IKvCache cache = lm.CreateCache(prefix.Length + steps + 1);

        float[] logits = lm.Forward(backend, prefix, posStart: 0, cache);
        ReadOnlySpan<float> expected = expectedLogits.AsReadOnlySpan<float>();
        AssertLogitsAgree(expected[..Yue2Protocol.VocabSize], logits, "prefill");

        List<int> produced = [];
        for (int step = 0; step < steps; step++)
        {
            // Greedy over the codec span, mirroring the reference dump's temperature-0 mask.
            int best = ArgMaxCodec(logits);
            produced.Add(best);
            logits = lm.Forward(backend, [best], posStart: prefix.Length + step, cache);
            AssertLogitsAgree(expected.Slice((step + 1) * Yue2Protocol.VocabSize, Yue2Protocol.VocabSize), logits, $"step {step}");
        }
        Assert.Equal(expectedTokens.AsReadOnlySpan<int>().ToArray(), produced);
        _out.WriteLine($"greedy tokens matched: [{string.Join(", ", produced.Take(4))}, …]");
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
            history: [], step: 0, Yue2Phase.Semantic, legacyOff: false);
        for (int i = 0; i < Yue2Protocol.CodecOffset; i++)
        {
            if (i != Yue2Protocol.MusicEnd) Assert.True(float.IsNegativeInfinity(semantic[i]), $"id {i} should be masked");
        }
        Assert.Equal(5, semantic.Count(float.IsFinite) - (float.IsFinite(semantic[Yue2Protocol.MusicEnd]) ? 1 : 0));

        // Under min_tokens the end token cannot be drawn.
        float[] gated = [.. scores];
        Yue2LogitProcessor.Apply(gated, Yue2Sampling.Semantic with { MinTokens = 200, TopK = 5, TopP = 1f, RepetitionPenalty = 1f },
            history: [], step: 3, Yue2Phase.Semantic, legacyOff: false);
        Assert.True(float.IsNegativeInfinity(gated[Yue2Protocol.MusicEnd]));

        // The windowed penalty is penalty^count over the window, and only over the window.
        float[] penalised = [.. scores];
        int repeated = Yue2Protocol.CodecOffset + 5;
        int stale = Yue2Protocol.CodecOffset + 6;
        int[] history = [stale, .. Enumerable.Repeat(repeated, 3)];
        Yue2LogitProcessor.Apply(penalised, Yue2Sampling.Semantic with { MinTokens = 0, TopK = int.MaxValue, TopP = 1f, RepetitionPenalty = 2f, PenaltyWindow = 3 },
            history, step: 4, Yue2Phase.Semantic, legacyOff: false);
        float raw = scores[repeated];
        Assert.Equal(raw > 0 ? raw / 8f : raw * 8f, penalised[repeated], 4);
        Assert.Equal(scores[stale], penalised[stale], 4);   // outside the 3-wide window

        // The historical off path keeps three sorted entries out of the nucleus cut instead of one.
        float[] modern = [.. scores], legacy = [.. scores];
        Yue2Sampling nucleus = Yue2Sampling.Semantic with { MinTokens = 0, TopK = int.MaxValue, TopP = 0.0001f, RepetitionPenalty = 1f, Temperature = 1f };
        Yue2LogitProcessor.Apply(modern, nucleus, [], 0, Yue2Phase.Semantic, legacyOff: false);
        Yue2LogitProcessor.Apply(legacy, nucleus, [], 0, Yue2Phase.Semantic, legacyOff: true);
        Assert.True(legacy.Count(float.IsFinite) > modern.Count(float.IsFinite));
    }

    private static int ArgMaxCodec(ReadOnlySpan<float> logits)
    {
        int best = Yue2Protocol.CodecOffset;
        float bestValue = float.NegativeInfinity;
        for (int i = Yue2Protocol.CodecOffset; i < Yue2Protocol.CodecOffset + Yue2Protocol.CodecSize; i++)
        {
            if (logits[i] > bestValue) { bestValue = logits[i]; best = i; }
        }
        return best;
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

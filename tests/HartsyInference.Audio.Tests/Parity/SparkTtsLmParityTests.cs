using System.Text.Json;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.SparkTts;
using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Spark-TTS LM greedy parity on real weights (CPU F32) against upstream <c>model.generate(do_sample=False)</c>
/// on the controllable "numbers" prompt (fixture from <c>tools/parity/spark_lm_reference.py --mode greedy</c>). Needs
/// <c>HARTSY_SPARK_DIR</c> naming a Spark-TTS-0.5B root with <c>LLM/model.safetensors</c>.</summary>
[Trait("Category", "Integration")]
public sealed unsafe class SparkTtsLmParityTests(ITestOutputHelper output)
{
    private static readonly string FixturePath =
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "SparkTts", "spark_lm_greedy_numbers.json");

    [Fact]
    public void GreedyContinuation_MatchesUpstream()
    {
        string? root = Environment.GetEnvironmentVariable("HARTSY_SPARK_DIR");
        string weights = Path.Combine(root ?? string.Empty, "LLM", "model.safetensors");
        if (string.IsNullOrEmpty(root) || !File.Exists(weights))
        {
            Assert.False(Environment.GetEnvironmentVariable("HARTSY_REQUIRE_REAL_WEIGHTS") == "1",
                "HARTSY_SPARK_DIR is not set to a Spark-TTS-0.5B root.");
            return;
        }
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(FixturePath));
        JsonElement r = doc.RootElement;
        int[] prompt = Ints(r.GetProperty("prompt_ids"));
        int[] expectedIds = Ints(r.GetProperty("greedy_ids"));
        int[] topIds = Ints(r.GetProperty("first_top_ids"));
        float[] topLogits = r.GetProperty("first_top_logits").EnumerateArray().Select(e => e.GetSingle()).ToArray();

        SparkTtsConfig cfg = SparkTtsConfig.V0_5B;
        using SafeTensorsLoader loader = new();
        loader.Load(weights);
        using CpuBackend backend = new();
        using Qwen2Model model = new(cfg.Llm);
        model.LoadWeights(loader.GetAllTensors(), "model");

        int vocab = cfg.Llm.VocabSize;
        List<int> ids = new(expectedIds.Length);
        float maxTopDiff = 0f;
        using (IKvCache cache = model.CreateDecodeCache(prompt.Length + expectedIds.Length + 8))
        {
            Tensor? hidden = model.Forward(backend, prompt, batch: 1, posStart: 0, cache);
            for (int step = 0; step < expectedIds.Length; step++)
            {
                Tensor last = VibeVoiceOps.SliceLastFrame(hidden!, cfg.Llm.HiddenSize);
                hidden!.Dispose();
                using Tensor logits = model.ProjectLogits(backend, last, batch: 1, t: 1);
                last.Dispose();
                ReadOnlySpan<float> l = new((void*)logits.DataPointer, vocab);
                if (step == 0)
                {
                    for (int i = 0; i < topIds.Length; i++) maxTopDiff = MathF.Max(maxTopDiff, MathF.Abs(l[topIds[i]] - topLogits[i]));
                }
                int next = 0;
                float best = float.NegativeInfinity;
                for (int i = 0; i < vocab; i++) if (l[i] > best) { best = l[i]; next = i; }
                ids.Add(next);
                // Teacher-force upstream's token so one divergence does not cascade; the last step needs no successor.
                hidden = step + 1 < expectedIds.Length
                    ? model.Forward(backend, [expectedIds[step]], batch: 1, posStart: cache.CurrentLength, cache)
                    : null;
            }
            hidden?.Dispose();
        }
        int match = 0;
        for (int i = 0; i < expectedIds.Length; i++) if (ids[i] == expectedIds[i]) match++;
        output.WriteLine($"first-step top-{topIds.Length} logits max |diff| {maxTopDiff:G3}; greedy {match}/{expectedIds.Length}");
        Assert.True(maxTopDiff < 1e-2f, $"first-step logits differ by {maxTopDiff}");
        Assert.Equal(expectedIds, ids);

        using SparkTtsLm lm = new(cfg);
        lm.LoadWeights(loader.GetAllTensors());
        (List<int> global, List<int> semantic) = lm.GenerateAudioTokens(backend, prompt, expectedIds.Length, greedy: true);
        Assert.Equal(Ints(r.GetProperty("global")), global);
        Assert.Equal(Ints(r.GetProperty("semantic")), semantic);
    }

    /// <summary>Sampled generation (upstream defaults) on the repeated-"no" line must end on its own: upstream
    /// stops after 150 to 235 semantic tokens on seeds 0 to 2 (CPU F32, cap 1500). Opt in with
    /// <c>HARTSY_SPARK_SAMPLED=1</c>; each seed is a full CPU generation.</summary>
    [Fact]
    [Trait("Category", "Slow")]
    public void SampledRepeatedWords_StopsLikeUpstream()
    {
        string? root = Environment.GetEnvironmentVariable("HARTSY_SPARK_DIR");
        string weights = Path.Combine(root ?? string.Empty, "LLM", "model.safetensors");
        if (Environment.GetEnvironmentVariable("HARTSY_SPARK_SAMPLED") != "1" || !File.Exists(weights)) return;
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "Fixtures", "SparkTts", "spark_prompt_ids.json")));
        int[] prompt = doc.RootElement.GetProperty("cases").EnumerateArray()
            .First(c => c.GetProperty("mode").GetString() == "controllable"
                && c.GetProperty("text").GetString()!.StartsWith("No no", StringComparison.Ordinal))
            .GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        SparkTtsConfig cfg = SparkTtsConfig.V0_5B;
        using SafeTensorsLoader loader = new();
        loader.Load(weights);
        using CpuBackend backend = new();
        using SparkTtsLm lm = new(cfg);
        lm.LoadWeights(loader.GetAllTensors());
        int seeds = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_SPARK_SEEDS"), out int n) ? n : 3;
        List<int> counts = [];
        for (int seed = 0; seed < seeds; seed++)
        {
            System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
            (List<int> global, List<int> semantic) = lm.GenerateAudioTokens(backend, prompt, 1500, seed);
            output.WriteLine($"seed {seed}: {global.Count} global + {semantic.Count} semantic in {sw.Elapsed.TotalSeconds:F1}s");
            counts.Add(semantic.Count);
        }
        Assert.All(counts, c => Assert.InRange(c, 1, 600));
    }

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt32()).ToArray();
}

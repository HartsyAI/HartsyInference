using System.Diagnostics;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Cuda;
using HartsyInference.LLM.Ernie5;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>Real-weight checks on the pinned ERNIE-5.5-Flash-Base checkpoint (baidu/ERNIE-5.5-Flash-Base). They run only when
/// <c>ERNIE_REAL_DIR</c> names a directory holding <c>config.json</c> and <c>model.safetensors.index.json</c>, and the shards for
/// the forward check. Each is reported as skipped (never passed) otherwise.</summary>
/// <remarks>Environment: <c>ERNIE_REAL_DEVICE</c> = <c>cuda</c> (default), <c>vulkan</c> or <c>cpu</c>; <c>ERNIE_REAL_TOKENS</c> = a comma
/// list of token ids (default <c>1,2,3,4,5</c>). The checkpoint is a base model with no chat template, so the forward takes token ids.</remarks>
[Trait("Category", "Integration")]
public sealed class Ernie5RealCheckpointTests
{
    private const string DirVariable = "ERNIE_REAL_DIR";

    /// <summary>A fact reported as skipped (never passed) when <c>ERNIE_REAL_DIR</c> is not set.</summary>
    private sealed class RealCheckpointFactAttribute : FactAttribute
    {
        public RealCheckpointFactAttribute()
        {
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable(DirVariable)))
                Skip = $"SKIPPED: {DirVariable} is not set; the real-weight check did not run.";
        }
    }

    /// <summary>A fact reported as skipped (never passed) unless <c>ERNIE_REAL_DIR</c> holds every shard the index names.</summary>
    private sealed class RealShardsFactAttribute : FactAttribute
    {
        public RealShardsFactAttribute()
        {
            string? dir = Environment.GetEnvironmentVariable(DirVariable);
            if (string.IsNullOrEmpty(dir)) Skip = $"SKIPPED: {DirVariable} is not set; the real-weight check did not run.";
            else Skip = SkipReason(dir);
        }

        // Returns why the forward check cannot run, or null when every shard is present. Never throws: a discovery error would hide the reason.
        private static string? SkipReason(string dir)
        {
            string index = Path.Combine(dir, "model.safetensors.index.json");
            if (!File.Exists(index)) return $"SKIPPED: {index} is not on disk; the forward check did not run.";
            try
            {
                using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(index));
                int missing = 0;
                foreach (string shard in doc.RootElement.GetProperty("weight_map").EnumerateObject().Select(p => p.Value.GetString()!).Distinct())
                    if (!File.Exists(Path.Combine(dir, shard))) missing++;
                return missing == 0 ? null : $"SKIPPED: {missing} shard file(s) are not on disk in {dir}; the forward check did not run.";
            }
            catch (Exception ex) when (ex is JsonException or KeyNotFoundException or InvalidOperationException or IOException)
            {
                return $"SKIPPED: {index} is unreadable ({ex.GetType().Name}: {ex.Message}); the forward check did not run.";
            }
        }
    }

    [RealCheckpointFact]
    public void Real_IndexMapsEveryKey()
    {
        string dir = Environment.GetEnvironmentVariable(DirVariable)!;
        Ernie5Config config = Ernie5Config.Load(Path.Combine(dir, "config.json"));
        using JsonDocument index = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "model.safetensors.index.json")));
        List<string> keys = [];
        foreach (JsonProperty entry in index.RootElement.GetProperty("weight_map").EnumerateObject()) keys.Add(entry.Name);
        Ernie5WeightCoverage coverage = Ernie5WeightMap.Check(config, keys, mtp: true);
        coverage.RequireComplete();
        Assert.Empty(coverage.Unmapped);
    }

    [RealShardsFact]
    public void Real_PrefillAndDecode_ProduceFiniteLogits()
    {
        string dir = Environment.GetEnvironmentVariable(DirVariable)!;
        string device = (Environment.GetEnvironmentVariable("ERNIE_REAL_DEVICE") ?? "cuda").ToLowerInvariant();
        int[] tokens = (Environment.GetEnvironmentVariable("ERNIE_REAL_TOKENS") ?? "1,2,3,4,5")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse).ToArray();
        // Reference precision, the same settings as the `reference` knob profile: F32 GEMMs without TF32 or F16 inputs.
        using IBackend backend = device switch
        {
            "cuda" => new CudaBackend(0, Ernie5ModelGpuParityTests.PtxDirectory()) { HighPrecisionGemm = true },
            "vulkan" => new VulkanBackend { EnableF16Gemm = false },
            "cpu" => new CpuBackend(),
            _ => throw new ArgumentException($"ERNIE_REAL_DEVICE must be cuda, vulkan or cpu, not '{device}'."),
        };

        Stopwatch load = Stopwatch.StartNew();
        using Ernie5Model model = Ernie5Model.Load(dir, backend, new Ernie5LoadOptions { MaxSequenceTokens = tokens.Length + 2, ResidentWeights = false });
        load.Stop();
        using Ernie5SequenceState state = model.CreateState(tokens.Length + 2);

        Stopwatch prefill = Stopwatch.StartNew();
        float[] logits = model.Forward(tokens, state);
        prefill.Stop();
        AssertHealthy(logits, model.Config.VocabSize, "prefill");
        int next = Argmax(logits);

        Stopwatch decode = Stopwatch.StartNew();
        float[] step = model.Forward([next], state);
        decode.Stop();
        AssertHealthy(step, model.Config.VocabSize, "decode");

        Console.WriteLine($"ERNIE real run on {device}: load {load.Elapsed.TotalSeconds:F1}s, prefill {tokens.Length} tokens {prefill.Elapsed.TotalSeconds:F1}s, "
            + $"decode 1 token {decode.Elapsed.TotalSeconds:F1}s, argmax after prefill {next}.");
    }

    private static void AssertHealthy(float[] logits, int vocab, string stage)
    {
        // One row of vocab logits: Argmax needs exactly one row, or its index would not be a token id.
        Assert.Equal(vocab, logits.Length);
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            Assert.True(float.IsFinite(logits[i]), $"{stage} logit {i} is {logits[i]}.");
            min = MathF.Min(min, logits[i]);
            max = MathF.Max(max, logits[i]);
        }
        Assert.True(max > min, $"{stage} logits are constant, which means the forward did not compute.");
    }

    private static int Argmax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
}

using System.Diagnostics;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Moe;
using HartsyInference.Engine.Placement;
using HartsyInference.Cuda;
using HartsyInference.LLM.Ernie5;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>Real-weight checks on the pinned ERNIE-5.5-Flash-Base checkpoint (baidu/ERNIE-5.5-Flash-Base). They run only when
/// <c>ERNIE_REAL_DIR</c> names a directory holding <c>config.json</c> and <c>model.safetensors.index.json</c>, and the shards for
/// the forward check. Each is reported as skipped (never passed) otherwise.</summary>
/// <remarks>Environment: <c>ERNIE_REAL_RESIDENT</c> = <c>1</c> builds every layer once and preloads its per-token weights (default: layers
/// stream per forward); <c>ERNIE_REAL_EXPERT_CACHE</c> = <c>1</c> (with resident, CUDA) attaches the device expert cache before the first
/// forward and prints its hit share per decode step; <c>ERNIE_REAL_DECODE_STEPS</c> = greedy decode steps after the prefill, each timed (default 1);
/// <c>ERNIE_REAL_DEVICE</c> = <c>cuda</c> (default), <c>vulkan</c> or <c>cpu</c>; <c>ERNIE_REAL_TOKENS</c> = a comma
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
            catch (Exception ex)
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

        bool resident = Environment.GetEnvironmentVariable("ERNIE_REAL_RESIDENT") == "1";
        bool expertCache = Environment.GetEnvironmentVariable("ERNIE_REAL_EXPERT_CACHE") == "1";
        int decodeSteps = int.TryParse(Environment.GetEnvironmentVariable("ERNIE_REAL_DECODE_STEPS"), out int n) && n > 0 ? n : 1;
        int capacity = tokens.Length + decodeSteps + 1;
        Stopwatch load = Stopwatch.StartNew();
        using Ernie5Model model = Ernie5Model.Load(dir, backend, new Ernie5LoadOptions { MaxSequenceTokens = capacity, ResidentWeights = resident });
        load.Stop();
        using Ernie5SequenceState state = model.CreateState(capacity);
        // Declared after the model so it is disposed first; attached before the first forward, so no expert is promoted outside it.
        using MoeExpertOffload? offload = expertCache ? AttachExpertCache(backend, model, resident) : null;

        Stopwatch prefill = Stopwatch.StartNew();
        float[] logits = model.Forward(tokens, state);
        prefill.Stop();
        AssertHealthy(logits, model.Config.VocabSize, "prefill");
        int next = Argmax(logits);

        // Greedy decode; the first step pays every first-touch cost, so later steps are the steady state.
        List<string> steps = [];
        List<int> generated = [];
        int token = next;
        MoeOffloadStats before = offload?.Stats ?? default;
        for (int i = 0; i < decodeSteps; i++)
        {
            Stopwatch decode = Stopwatch.StartNew();
            float[] step = model.Forward([token], state);
            decode.Stop();
            AssertHealthy(step, model.Config.VocabSize, $"decode {i + 1}");
            if (offload is null)
            {
                steps.Add($"{decode.Elapsed.TotalSeconds:F2}s");
            }
            else
            {
                // This step's own hit share and admissions, not the running total since attach (which includes the prefill).
                MoeOffloadStats now = offload.Stats;
                long hit = now.ResidentRows - before.ResidentRows, all = hit + now.StreamedRows - before.StreamedRows + now.HostRows - before.HostRows;
                steps.Add($"{decode.Elapsed.TotalSeconds:F2}s hit {(all == 0 ? 0 : (double)hit / all):P0} +{now.Admitted - before.Admitted}");
                before = now;
            }
            token = Argmax(step);
            generated.Add(token);
        }

        if (offload is not null)
        {
            MoeOffloadStats stats = offload.Stats;
            Console.WriteLine($"ERNIE expert cache: resident rows {stats.ResidentRows}, streamed rows {stats.StreamedRows}, admitted {stats.Admitted}, "
                + $"resident share {stats.ResidentShare:P1}.");
        }
        Console.WriteLine($"ERNIE real run on {device} (resident {resident}, expert cache {expertCache}): load {load.Elapsed.TotalSeconds:F1}s, prefill {tokens.Length} tokens "
            + $"{prefill.Elapsed.TotalSeconds:F1}s, argmax after prefill {next}, decode steps [{string.Join(", ", steps)}], "
            + $"greedy tokens [{string.Join(",", generated)}].");
    }

    // The device-only expert cache, sized to what the dense preload left free less the same headroom.
    private static MoeExpertOffload AttachExpertCache(IBackend backend, Ernie5Model model, bool resident)
    {
        if (!resident) throw new ArgumentException("ERNIE_REAL_EXPERT_CACHE needs ERNIE_REAL_RESIDENT=1.");
        DeviceMemoryUsage usage = backend.GetDeviceMemoryUsage();
        long budget = Ernie5Model.PreloadBudget(usage.FreeBytes, usage.TotalBytes);
        return ExpertOffloadSetup.AttachDeviceOnly(backend, model, budget, "[ernie5 test]");
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

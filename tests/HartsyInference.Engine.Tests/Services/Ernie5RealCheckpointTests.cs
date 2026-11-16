using System.Diagnostics;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Moe;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Requests;
using HartsyInference.Cuda;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Ernie5;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>Real-weight checks on the pinned ERNIE-5.5-Flash-Base checkpoint (baidu/ERNIE-5.5-Flash-Base). They run only when
/// <c>ERNIE_REAL_DIR</c> names a directory holding <c>config.json</c> and <c>model.safetensors.index.json</c>, and the shards for
/// the forward check. Each is reported as skipped (never passed) otherwise.</summary>
/// <remarks>Environment: <c>ERNIE_REAL_RESIDENT</c> = <c>1</c> builds every layer once and preloads its per-token weights (default: layers
/// stream per forward); <c>ERNIE_REAL_EXPERT_CACHE</c> = <c>1</c> (with resident, CUDA) attaches the device expert cache before the first
/// forward and prints its hit share per decode step; <c>ERNIE_REAL_DECODE_STEPS</c> = greedy decode steps after the prefill, each timed
/// (default 1); <c>ERNIE_REAL_DEVICE</c> = <c>cuda</c> (default), <c>vulkan</c> or <c>cpu</c>; <c>ERNIE_REAL_TOKENS</c> = a comma list of
/// token ids (default <c>1,2,3,4,5</c>); <c>ERNIE_REAL_PROFILE</c> = a directory for the CUDA per-op table and the stage timer, per phase
/// (prefill, decode). On Linux each forward also reports its major page faults and storage reads. The checkpoint is a base model with no
/// chat template, so the forward takes token ids, and the Text service check sends a plain completion prompt (<c>ERNIE_REAL_DECODE_STEPS</c>
/// is its token limit there, default 16).</remarks>
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
        string? profileDir = Environment.GetEnvironmentVariable("ERNIE_REAL_PROFILE");
        using IDisposable? knobs = string.IsNullOrEmpty(profileDir) ? null : PushProfiling(profileDir);
        int capacity = tokens.Length + decodeSteps + 1;
        Stopwatch load = Stopwatch.StartNew();
        using Ernie5Model model = Ernie5Model.Load(dir, backend, new Ernie5LoadOptions { MaxSequenceTokens = capacity, ResidentWeights = resident });
        load.Stop();
        using Ernie5SequenceState state = model.CreateState(capacity);
        // Declared after the model so it is disposed first; attached before the first forward, so no expert is promoted outside it.
        using MoeExpertOffload? offload = expertCache ? AttachExpertCache(backend, model, resident) : null;

        ResetProfiles(backend);
        ProcessIo io = ProcessIo.Read();
        Stopwatch prefill = Stopwatch.StartNew();
        float[] logits = model.Forward(tokens, state);
        prefill.Stop();
        string prefillIo = ProcessIo.Read().Since(io);
        DumpProfiles(backend, profileDir, "prefill");
        AssertHealthy(logits, model.Config.VocabSize, "prefill");
        int next = Argmax(logits);

        // Greedy decode; the first step pays every first-touch cost, so later steps are the steady state.
        List<string> steps = [];
        List<int> generated = [];
        int token = next;
        MoeOffloadStats before = offload?.Stats ?? default;
        for (int i = 0; i < decodeSteps; i++)
        {
            io = ProcessIo.Read();
            Stopwatch decode = Stopwatch.StartNew();
            float[] step = model.Forward([token], state);
            decode.Stop();
            string stepIo = ProcessIo.Read().Since(io);
            AssertHealthy(step, model.Config.VocabSize, $"decode {i + 1}");
            if (offload is null)
            {
                steps.Add($"{decode.Elapsed.TotalSeconds:F2}s{stepIo}");
            }
            else
            {
                // This step's own hit share and admissions, not the running total since attach (which includes the prefill).
                MoeOffloadStats now = offload.Stats;
                long hit = now.ResidentRows - before.ResidentRows, all = hit + now.StreamedRows - before.StreamedRows + now.HostRows - before.HostRows;
                steps.Add($"{decode.Elapsed.TotalSeconds:F2}s{stepIo} hit {(all == 0 ? 0 : (double)hit / all):P0} +{now.Admitted - before.Admitted}");
                before = now;
            }
            token = Argmax(step);
            generated.Add(token);
        }
        DumpProfiles(backend, profileDir, "decode");

        if (offload is not null)
        {
            MoeOffloadStats stats = offload.Stats;
            Console.WriteLine($"ERNIE expert cache: resident rows {stats.ResidentRows}, streamed rows {stats.StreamedRows}, admitted {stats.Admitted}, "
                + $"resident share {stats.ResidentShare:P1}.");
        }
        Console.WriteLine($"ERNIE real run on {device} (resident {resident}, expert cache {expertCache}): load {load.Elapsed.TotalSeconds:F1}s, prefill {tokens.Length} tokens "
            + $"{prefill.Elapsed.TotalSeconds:F1}s{prefillIo}, argmax after prefill {next}, decode steps [{string.Join(", ", steps)}], "
            + $"greedy tokens [{string.Join(",", generated)}].");
    }

    [RealCheckpointFact]
    public void Real_CompletionPrompt_MatchesTheHfTokenizer()
    {
        string dir = Environment.GetEnvironmentVariable(DirVariable)!;
        (string? bos, string? eos) = HfChatAssets.SpecialTokenLiterals(dir);
        ILlmTokenizer tokenizer = HfChatAssets.LoadTokenizer(dir, [], bos, eos);
        RawCompletionTemplate template = HfChatAssets.LoadCompletionTemplate(dir);

        // tokenizer(prompt).input_ids from the Hugging Face tokenizers library on the published tokenizer.json: the post-processor puts
        // <|ERNIE_MASK|> (id 2) first, and digits group in threes from the right.
        Assert.Equal([2, 47712, 6057, 6463, 1706], Encode("Large language models are"));
        Assert.Equal([2, 2029, 10301, 1507, 13573, 1543, 1423, 1880, 17636, 1229], Encode("The capital of France is 12345."));

        // The wider set from tests/python-reference/dump_ernie55_tokenizer_cases.py, when ERNIE_TOKENIZER_CASES names its output.
        string? casesPath = Environment.GetEnvironmentVariable("ERNIE_TOKENIZER_CASES");
        if (string.IsNullOrEmpty(casesPath)) return;
        using JsonDocument cases = JsonDocument.Parse(File.ReadAllBytes(casesPath));
        int checkedCases = 0;
        foreach (JsonElement row in cases.RootElement.EnumerateArray())
        {
            string text = row.GetProperty("text").GetString()!;
            int[] expected = [.. row.GetProperty("ids").EnumerateArray().Select(id => id.GetInt32())];
            Assert.True(expected.SequenceEqual(Encode(text)), $"ids differ for {JsonSerializer.Serialize(text)}.");
            checkedCases++;
        }
        Console.WriteLine($"ERNIE tokenizer: {checkedCases} reference cases match.");

        int[] Encode(string prompt) => template.Encode(tokenizer, [new ChatMessage("user", prompt)], addGenerationPrompt: true);
    }

    [RealShardsFact]
    public async Task Real_TextService_GeneratesFromAPlainPrompt()
    {
        string dir = Environment.GetEnvironmentVariable(DirVariable)!;
        string device = (Environment.GetEnvironmentVariable("ERNIE_REAL_DEVICE") ?? "cuda") switch
        {
            "cuda" => "cuda:0",
            "vulkan" => "vulkan:0",
            string other => other,
        };
        int maxTokens = int.TryParse(Environment.GetEnvironmentVariable("ERNIE_REAL_DECODE_STEPS"), out int n) && n > 0 ? n : 16;
        using InferenceEngine engine = new(device);
        ModelSpec spec = new() { Requested = dir, Modality = Modality.Text, LocalPath = dir };
        TextRequest request = new()
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "Large language models are" }],
            Device = device,
            Temperature = 0,
            MaxTokens = maxTokens,
        };

        Stopwatch total = Stopwatch.StartNew();
        TextResult result = await engine.Text.GenerateAsync(spec, request);
        total.Stop();

        Console.WriteLine($"ERNIE Text service on {device}: {result.PromptTokens} prompt tokens, {result.CompletionTokens} generated, stop {result.Stop}, "
            + $"prefill {result.PrefillMilliseconds / 1000:F1}s, decode {result.DecodeMilliseconds / 1000:F1}s "
            + $"({(result.CompletionTokens > 1 ? result.DecodeMilliseconds / (result.CompletionTokens - 1) / 1000 : 0):F2}s/token), "
            + $"total with load {total.Elapsed.TotalSeconds:F1}s. Text: {JsonSerializer.Serialize(result.Text)}");
        Assert.Equal(5, result.PromptTokens);
        Assert.InRange(result.CompletionTokens, 1, maxTokens);
        Assert.False(string.IsNullOrWhiteSpace(result.Text));
    }

    // The CUDA per-op table (diagnostics.profile; expert uploads show as H2D_BIG) and the model stage timer (diagnostics.stageProfile), written
    // under profileDir. Both are off unless ERNIE_REAL_PROFILE names that directory. The stage timer syncs around each stage, so profiled
    // totals run slower than an unprofiled pass.
    private static IDisposable PushProfiling(string profileDir)
    {
        Directory.CreateDirectory(profileDir);
        return KnobProfileScope.Push(KnobProfile.Create("ernie5-real-profile")
            .With(EngineKnobs.Profile, true)
            .With(EngineKnobs.StageProfile, true)
            .With<string?>(EngineKnobs.ProfileOut, Path.Combine(profileDir, "ops")));
    }

    private static void ResetProfiles(IBackend backend)
    {
        backend.ResetOpProfile();
        StageProfiler.Reset();
    }

    // Writes ops.<phase> and stages.<phase>.tsv, then starts the next phase from zero.
    private static void DumpProfiles(IBackend backend, string? profileDir, string phase)
    {
        if (string.IsNullOrEmpty(profileDir)) return;
        backend.DumpOpProfile(phase);
        File.WriteAllText(Path.Combine(profileDir, $"stages.{phase}.tsv"), StageProfiler.Report());
        ResetProfiles(backend);
    }

    /// <summary>The process's major page faults and storage reads on Linux, where routed experts are read from the mapped checkpoint. A
    /// fault per step shows an expert read from disk rather than from the page cache. Zero and unreported elsewhere.</summary>
    /// <remarks><c>read_bytes</c> counts block-layer reads, so a FUSE or NFS network volume can report about 0 MB while every expert comes
    /// off the network; major faults still count those reads. High faults with about 0 MB read means the volume is not counted, not that
    /// the page cache served the reads.</remarks>
    private readonly record struct ProcessIo(long MajorFaults, long ReadBytes)
    {
        // Each counter is read on its own, -1 when unavailable, so a kernel without /proc/self/io still reports the faults.
        public static ProcessIo Read() => OperatingSystem.IsLinux() ? new ProcessIo(Counter(MajorFaultCount), Counter(ReadByteCount)) : new ProcessIo(-1, -1);

        private static long Counter(Func<long> read)
        {
            try
            {
                return read();
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FormatException or IndexOutOfRangeException)
            {
                return -1;
            }
        }

        // /proc/self/stat: the command is in parentheses and may hold spaces, so count fields after the last ')'; majflt is field 12.
        private static long MajorFaultCount()
        {
            string stat = File.ReadAllText("/proc/self/stat");
            string[] fields = stat[(stat.LastIndexOf(')') + 2)..].Split(' ');
            return long.Parse(fields[9], System.Globalization.CultureInfo.InvariantCulture);
        }

        private static long ReadByteCount()
        {
            string? line = File.ReadLines("/proc/self/io").FirstOrDefault(l => l.StartsWith("read_bytes:", StringComparison.Ordinal));
            if (line is null) throw new FormatException("/proc/self/io has no read_bytes line.");
            return long.Parse(line["read_bytes:".Length..].Trim(), System.Globalization.CultureInfo.InvariantCulture);
        }

        /// <summary>" (N major faults, M MB read)" since <paramref name="start"/>, with n/a for a counter either sample could not read;
        /// empty off Linux.</summary>
        public string Since(ProcessIo start)
        {
            if (!OperatingSystem.IsLinux()) return "";
            string faults = MajorFaults < 0 || start.MajorFaults < 0 ? "n/a" : (MajorFaults - start.MajorFaults).ToString(System.Globalization.CultureInfo.InvariantCulture);
            string read = ReadBytes < 0 || start.ReadBytes < 0 ? "n/a" : ((ReadBytes - start.ReadBytes) >> 20).ToString(System.Globalization.CultureInfo.InvariantCulture);
            return $" ({faults} major faults, {read} MB read)";
        }
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

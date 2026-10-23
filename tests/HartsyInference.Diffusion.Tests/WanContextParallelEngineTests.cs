using System.Globalization;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wan context parallelism through the full engine on real TI2V-5B weights, both ranks on their own card.
/// <para>Accuracy is judged per forward, not on the finished video: a few-step trajectory amplifies any numerics difference, so
/// two runs whose denoisers agree to F32 rounding still end far apart in frame SSIM (issue #552). The teacher-forced probe records
/// a single-GPU F32 run and replays its inputs at every step, once on one card with exact F32 attention as the control and once
/// across both ranks; context parallelism passes when each forward's relL2 is within three times the control's.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class WanContextParallelEngineTests : IDisposable
{
    private const double ControlMultiple = 3.0;

    private static readonly VideoRequest Request = new()
    {
        Prompt = "a red ball bouncing on a wooden table",
        Width = 480,
        Height = 480,
        Frames = 9,
        Steps = 6,
        CfgScale = 5.0f,
        Seed = 42,
    };

    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"wan-cp-probe-{Guid.NewGuid():N}");

    public WanContextParallelEngineTests(ITestOutputHelper output) => _output = output;

    /// <summary>The configured ranks run context-parallel (the decision is logged as active) and the generation completes.</summary>
    [Fact]
    public async Task ContextParallel_RealEngine_IsActiveAndCompletes()
    {
        if (!TryResolve(out ModelSpec spec)) return;
        VideoGenerationResult parallel = null!;
        string[] decisions = await CaptureDecisions(async () => parallel = await Generate(spec, contextParallel: true));
        Assert.Contains(decisions, d => d.Contains("active"));
        Assert.Equal(Request.Frames, parallel.Frames.Count);
    }

    /// <summary>Every step's context-parallel forward, on the single-GPU run's recorded inputs, is within three times the distance
    /// exact F32 attention puts between two single-GPU forwards.</summary>
    [Fact]
    public async Task ContextParallel_TeacherForcedForward_IsWithinThreeTimesTheF32AttentionControl()
    {
        if (!TryResolve(out ModelSpec spec)) return;
        string record = Path.Combine(_dir, "record"), control = Path.Combine(_dir, "control"), parallel = Path.Combine(_dir, "cp");
        await Probe(spec, "record", record, null, contextParallel: false);
        await Probe(spec, "replay", record, control, contextParallel: false, exactAttention: true);
        string[] decisions = await CaptureDecisions(() => Probe(spec, "replay", record, parallel, contextParallel: true));
        // A rank that fails its preload falls back to one GPU, whose replay would match the record and pass every ratio.
        Assert.Contains(decisions, d => d.Contains("active"));

        Dictionary<string, double> controlRows = Rows(control), parallelRows = Rows(parallel);
        Assert.Equal(2 * Request.Steps, controlRows.Count);
        Assert.Equal(controlRows.Keys.Order(), parallelRows.Keys.Order());
        foreach ((string key, double reference) in controlRows.OrderBy(r => r.Key, StringComparer.Ordinal))
        {
            double cp = parallelRows[key];
            _output.WriteLine($"{key}: CP relL2 {cp:E3}, control {reference:E3}, ratio {cp / reference:F2}");
            Assert.True(reference > 0, $"{key}: the control replay matched the record bit for bit, so it measures nothing.");
            Assert.True(cp > 0, $"{key}: the context-parallel replay matched the record bit for bit, so it ran on one GPU.");
            Assert.True(cp <= ControlMultiple * reference,
                $"{key}: the context-parallel forward is {cp:E3} from the single-GPU one, more than {ControlMultiple}x the {reference:E3} exact F32 attention gives.");
        }
    }

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    private bool TryResolve(out ModelSpec spec)
    {
        spec = null!;
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return false; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return false; }
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.WanVideo.Ti2V5B, TestPaths.WanVideo.Umt5Xxl)) return false;
        // TI2V-5B fp16 is ~10 GB REPLICATED per rank.
        foreach (GpuTopologyInfo gpu in CudaTopology.Probe())
        {
            _output.WriteLine($"GPU {gpu.Ordinal} ({gpu.Name}): {gpu.FreeMemoryBytes >> 20} MiB free");
            if (gpu.FreeMemoryBytes < 11L << 30) { _output.WriteLine($"SKIPPED: GPU {gpu.Ordinal} has < 11 GiB free."); return false; }
        }
        spec = ModelResolver.Resolve("wan", TestPaths.WanVideo.Ti2V5B, Modality.Video);
        if (spec.LocalPath is null) { _output.WriteLine("SKIPPED: wan not resolvable with the explicit path."); return false; }
        return true;
    }

    /// <summary>The <c>[ContextParallel]</c> decisions logged while <paramref name="run"/> generates.</summary>
    private async Task<string[]> CaptureDecisions(Func<Task> run)
    {
        List<string> decisions = [];
        Logs.SetLogger((level, message) =>
        {
            if (message.Contains("[ContextParallel]")) lock (decisions) decisions.Add(message);
            Console.Error.WriteLine($"[{level}] {message}");
        });
        try
        {
            await run();
        }
        finally
        {
            Logs.SetLogger(static (level, message) => Console.Error.WriteLine($"[{level}] {message}"));
        }
        string[] captured;
        lock (decisions) captured = [.. decisions];
        _output.WriteLine($"ContextParallel decisions: {string.Join(" | ", captured)}");
        return captured;
    }

    private async Task Probe(ModelSpec spec, string mode, string record, string? outDir, bool contextParallel, bool exactAttention = false)
    {
        // F32 on both arms: context parallelism runs Wan's F32 branches, and the record must be the same computation.
        KnobProfile profile = KnobProfile.Create("wan-cp-probe").With(EngineKnobs.WanF16, false)
            .With(EngineKnobs.DenoiserProbe, mode).With(EngineKnobs.DenoiserProbeDir, record).With(EngineKnobs.DenoiserProbeSteps, "all");
        if (outDir is not null) profile = profile.With(EngineKnobs.DenoiserProbeOut, outDir);
        if (exactAttention) profile = profile.With(EngineKnobs.SdpaNoF16, true);
        using IDisposable scope = profile.Push();
        await Generate(spec, contextParallel);
    }

    private static async Task<VideoGenerationResult> Generate(ModelSpec spec, bool contextParallel)
    {
        EngineOptions? options = contextParallel
            ? new EngineOptions { Placement = new PlacementConfig { ContextParallelDevices = ["cuda:0", "cuda:1"] } }
            : null;
        using InferenceEngine engine = options is null ? new InferenceEngine("cuda", 0) : new InferenceEngine("cuda", 0, options);
        return await engine.Video.GenerateAsync(spec, Request);
    }

    /// <summary>relL2 per <c>step/output</c> from a replay's <c>probe.tsv</c>.</summary>
    private static Dictionary<string, double> Rows(string outDir) =>
        File.ReadAllLines(Path.Combine(outDir, "probe.tsv")).Skip(1).Select(line => line.Split('\t'))
            .ToDictionary(r => $"step {r[0]}/{r[2]}", r => double.Parse(r[3], CultureInfo.InvariantCulture));
}

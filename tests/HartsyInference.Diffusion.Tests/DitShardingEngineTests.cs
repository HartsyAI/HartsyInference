using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>DiT sharding through the FULL engine path (<c>InferenceEngine</c> → recipe → pipeline) with
/// <see cref="PlacementConfig.EnableDitSharding"/> — the placement plumbing, the asymmetric preload/free, and the
/// per-step <c>ForwardSharded</c> routing, not each model's raw transformer primitive (that lives in the sibling
/// <c>*DitShardingTests</c> files). One shared harness parameterized per model instead of a near-identical file per
/// model (was: Chroma/Flux/HunyuanImage/Krea2/QwenImage/Sd3, ~1050 lines combined, each differing only in model id,
/// checkpoint, request shape, VRAM floors and the SSIM bar). <see cref="MiniMaxH3DitShardingEngineTests"/> stays a
/// separate file (video modality, tuple video+audio output, a 3-way same/cross-device comparison — not the same
/// shape as this 2-way image comparison). Qwen-Image's extra same-device and regime-matched gated tiers stay in
/// <see cref="QwenImageDitShardingEngineTests"/> since no other model has an equivalent fact to share them with.
///
/// <para>The SSIM gate is per-model, not a single constant — the two backends may take different fp8/F32 GEMM
/// paths across GPU generations (whichever ordinal lands on an SM 8.9+ card gets native fp8; the other dequants),
/// so sharded vs unsharded can legitimately diverge numerically even when both are correct. Qwen-Image's checkpoint
/// in particular has a documented outlier-channel precision regime that makes its default-regime SSIM
/// informational rather than a tight correctness bar (see the case table below) — <c>QwenImageDitShardingEngineTests</c>
/// carries the GATED regression tiers for that model.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class DitShardingEngineTests
{
    private readonly ITestOutputHelper _output;
    public DitShardingEngineTests(ITestOutputHelper output) => _output = output;

    /// <summary>One model's case for the "sharded ≈ unsharded" comparison. <paramref name="ResolverCheckpoint"/> is
    /// what's passed to <see cref="ModelResolver.Resolve"/> (null lets it resolve from the catalog);
    /// <paramref name="GatePath"/> is what <see cref="RealWeightGate"/> checks for existence — usually the same
    /// file, but Chroma/Krea2 gate on a specific known file while resolving the model by id alone.</summary>
    public sealed record CoherentCase(
        string Name, string ModelId, string? ResolverCheckpoint, string GatePath, Modality Modality,
        int Width, int Height, int Steps, float Cfg, int Seed,
        double? Floor0Gb, double? Floor1Gb, double SsimThreshold, string SplitDescription);

    public static TheoryData<CoherentCase> CoherentCases()
    {
        const string prompt = "a photograph of an astronaut riding a horse";
        _ = prompt;
        TheoryData<CoherentCase> data = new()
        {
            new CoherentCase("Chroma", "chroma", null, TestPaths.Chroma.V1, Modality.Image,
                1024, 1024, 6, 5.0f, 42, 12.0, 6.0, 0.75, "57-block flat loop split cuda:0 + cuda:1"),
            new CoherentCase("Flux", "flux1", TestPaths.Flux.Dev, TestPaths.Flux.Dev, Modality.Image,
                1024, 1024, 8, 1.0f, 42, 16.0, 7.0, 0.75, "57-block flat loop split cuda:0 + cuda:1"),
            new CoherentCase("HunyuanImage", "hunyuan-image", HunyuanImageCheckpoint, HunyuanImageCheckpoint, Modality.Image,
                1024, 1024, 8, 3.5f, 42, null, null, 0.75, "60-block flat loop split cuda:0 + cuda:1"),
            new CoherentCase("Krea2", "krea2", null,
                Path.Combine(TestPaths.Krea2.TurboDir, "krea2_turbo_fp8_scaled.safetensors"), Modality.Image,
                1024, 1024, 8, 1.0f, 42, 8.0, 8.0, 0.75, "DiT block loop split cuda:0 + cuda:1"),
            new CoherentCase("QwenImage", "qwen-image", TestPaths.QwenImageEdit.Edit2511Fp8, TestPaths.QwenImageEdit.Edit2511Fp8, Modality.Image,
                1024, 1024, 8, 1.0f, 42, 20.0, 9.0, 0.05, "60-block loop split cuda:0 + cuda:1"),
            new CoherentCase("Sd3", "sd3", TestPaths.Sd35.MediumTransformerOnly, TestPaths.Sd35.MediumTransformerOnly, Modality.Image,
                768, 768, 12, 4.5f, 42, 6.0, 6.0, 0.75, "JointBlock loop split cuda:0 + cuda:1"),
        };
        return data;
    }

    private static readonly string HunyuanImageCheckpoint =
        Path.Combine(TestPaths.ModelsDir, "Stable-Diffusion", "HunyuanImage", "HunyuanImage2.1-Q4_K_M.gguf");

    [Theory]
    [MemberData(nameof(CoherentCases))]
    public async Task DitSharding_RealEngine_ProducesCoherentImage_WithinToleranceOfUnsharded(CoherentCase c)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        if (!RealWeightGate.Require(_output.WriteLine, c.GatePath)) return;
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(DitShardingEngineTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        if (c.Floor0Gb is double f0 && c.Floor1Gb is double f1)
        {
            (double free0, double free1) = ProbeFreeGb(ptxDir);
            _output.WriteLine($"Free VRAM — ordinal 0: {free0:F2} GB, ordinal 1: {free1:F2} GB.");
            if (free0 < f0 || free1 < f1)
            {
                _output.WriteLine($"SKIPPED: insufficient free VRAM for baseline + sharded runs (need {f0:F0}/{f1:F0} GB).");
                return;
            }
        }

        ModelSpec spec = ModelResolver.Resolve(c.ModelId, c.ResolverCheckpoint, c.Modality);
        if (spec.LocalPath is null) { _output.WriteLine($"SKIPPED: {c.ModelId} not resolvable."); return; }

        ImageRequest request = new ImageRequest
        {
            Prompt = "a photograph of an astronaut riding a horse",
            Width = c.Width,
            Height = c.Height,
            Steps = c.Steps,
            CfgScale = c.Cfg,
            Seed = c.Seed,
        };

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[{c.Name}] [1/2] UNSHARDED baseline (single GPU, ordinal 0; may block-stream)...");
        ImageResult baseline;
        using (InferenceEngine unshardedEngine = new InferenceEngine("cuda", 0))
        {
            baseline = await unshardedEngine.Images.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  baseline: {baseline.Width}x{baseline.Height}, seed={baseline.Seed}, {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(baseline.Rgb, "baseline");

        sw.Restart();
        _output.WriteLine($"[{c.Name}] [2/2] SHARDED ({c.SplitDescription})...");
        PlacementConfig placement = new PlacementConfig { ShardDevices = ["cuda:0", "cuda:1"], EnableDitSharding = true };
        ImageResult sharded;
        using (InferenceEngine shardedEngine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement }))
        {
            sharded = await shardedEngine.Images.GenerateAsync(spec, request);
        }
        _output.WriteLine($"  sharded: {sharded.Width}x{sharded.Height}, seed={sharded.Seed}, {sw.Elapsed.TotalSeconds:F1}s");
        AssertCoherent(sharded.Rgb, "sharded");

        Assert.Equal(baseline.Width, sharded.Width);
        Assert.Equal(baseline.Height, sharded.Height);
        double ssim = Ssim.Compute(baseline.Rgb, sharded.Rgb, sharded.Width, sharded.Height);
        _output.WriteLine($"[{c.Name}] SSIM(baseline, sharded) = {ssim:F4} (gate > {c.SsimThreshold:F2})");
        Assert.True(ssim > c.SsimThreshold, $"[{c.Name}] sharded output diverged too far from the unsharded baseline " +
            $"(SSIM={ssim:F4}) — check the block-range hand-off (CopyFromPeer for shared activations/temb) rather " +
            "than assuming this is expected fp8-path drift.");
    }

    /// <summary>One model's case for the teardown-symmetry regression: with <c>HARTSY_KEEP_MODELS=0</c>, two
    /// consecutive sharded generations must not accumulate VRAM on either ordinal — the shard backend's block
    /// range is freed by the asymmetric free path, not the unsharded whole-DiT free that would silently no-op
    /// there. <c>HARTSY_KEEP_MODELS</c> is a <c>static readonly</c> read per pipeline type, so this only holds if
    /// nothing else in the process already touched that model's pipeline type first.</summary>
    public sealed record NonResidentCase(
        string Name, string ModelId, string? ResolverCheckpoint, string GatePath, Modality Modality,
        int Width, int Height, int Steps, float Cfg, int Seed, double? Floor0Gb, double? Floor1Gb);

    public static TheoryData<NonResidentCase> NonResidentCases() => new()
    {
        new NonResidentCase("Chroma", "chroma", null, TestPaths.Chroma.V1, Modality.Image,
            1024, 1024, 4, 5.0f, 42, 10.0, 5.0),
        new NonResidentCase("Krea2", "krea2", null,
            Path.Combine(TestPaths.Krea2.TurboDir, "krea2_turbo_fp8_scaled.safetensors"), Modality.Image,
            1024, 1024, 8, 1.0f, 42, null, null),
        new NonResidentCase("QwenImage", "qwen-image", TestPaths.QwenImageEdit.Edit2511Fp8, TestPaths.QwenImageEdit.Edit2511Fp8, Modality.Image,
            1024, 1024, 4, 1.0f, 42, 16.0, 8.0),
        new NonResidentCase("Sd3", "sd3", TestPaths.Sd35.MediumTransformerOnly, TestPaths.Sd35.MediumTransformerOnly, Modality.Image,
            768, 768, 6, 4.5f, 42, 6.0, 6.0),
    };

    [Theory]
    [MemberData(nameof(NonResidentCases))]
    public async Task DitSharding_NonResident_FreesShardBackend_NoAccumulationAcrossGenerations(NonResidentCase c)
    {
        Environment.SetEnvironmentVariable("HARTSY_KEEP_MODELS", "0");

        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 physical GPUs."); return; }
        if (!RealWeightGate.Require(_output.WriteLine, c.GatePath)) return;
        ModelSpec spec = ModelResolver.Resolve(c.ModelId, c.ResolverCheckpoint, c.Modality);
        if (spec.LocalPath is null) { _output.WriteLine($"SKIPPED: {c.ModelId} not resolvable."); return; }

        if (c.Floor0Gb is double f0 && c.Floor1Gb is double f1)
        {
            (double preFree0, double preFree1) = ProbeFreeGb();
            if (preFree0 < f0 || preFree1 < f1) { _output.WriteLine("SKIPPED: insufficient free VRAM."); return; }
        }

        ImageRequest request = new ImageRequest
        {
            Prompt = "a photograph of an astronaut riding a horse",
            Width = c.Width,
            Height = c.Height,
            Steps = c.Steps,
            CfgScale = c.Cfg,
            Seed = c.Seed,
        };
        PlacementConfig placement = new PlacementConfig { ShardDevices = ["cuda:0", "cuda:1"], EnableDitSharding = true };

        using InferenceEngine engine = new InferenceEngine("cuda", 0, new EngineOptions { Placement = placement });
        _output.WriteLine($"[{c.Name}] [1/2] First sharded generation (non-resident: weights freed after)...");
        await engine.Images.GenerateAsync(spec, request);
        (double free0AfterFirst, double free1AfterFirst) = ProbeFreeGb();
        _output.WriteLine($"  after gen 1 — ordinal 0: {free0AfterFirst:F2} GB free, ordinal 1: {free1AfterFirst:F2} GB free");

        _output.WriteLine($"[{c.Name}] [2/2] Second sharded generation (re-loads + re-frees both backends' ranges)...");
        await engine.Images.GenerateAsync(spec, request);
        (double free0AfterSecond, double free1AfterSecond) = ProbeFreeGb();
        _output.WriteLine($"  after gen 2 — ordinal 0: {free0AfterSecond:F2} GB free, ordinal 1: {free1AfterSecond:F2} GB free");

        double drift0 = free0AfterFirst - free0AfterSecond;
        double drift1 = free1AfterFirst - free1AfterSecond;
        _output.WriteLine($"[{c.Name}] Drift after gen 2 vs gen 1 — ordinal 0: {drift0:F2} GB, ordinal 1: {drift1:F2} GB");
        Assert.True(Math.Abs(drift0) < 0.5, $"[{c.Name}] ordinal 0 free VRAM drifted {drift0:F2} GB between generations — possible leak.");
        Assert.True(Math.Abs(drift1) < 0.5, $"[{c.Name}] ordinal 1 (shard backend) free VRAM drifted {drift1:F2} GB between " +
            "generations — the asymmetric free (EnumerateBlockRangeWeights on the shard backend) may not be firing.");
    }

    private static void AssertCoherent(byte[] rgb, string label)
    {
        int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
        Assert.True(nonZero > rgb.Length * 0.1, $"{label}: image is all black");
        Assert.True(nonFF > rgb.Length * 0.1, $"{label}: image is all white");
    }

    private static (double free0Gb, double free1Gb) ProbeFreeGb() =>
        ProbeFreeGb(Path.Combine(Path.GetDirectoryName(typeof(DitShardingEngineTests).Assembly.Location)!, "Ptx"));

    private static (double free0Gb, double free1Gb) ProbeFreeGb(string ptxDir)
    {
        using CudaBackend probe0 = new(deviceOrdinal: 0, ptxDir: ptxDir);
        using CudaBackend probe1 = new(deviceOrdinal: 1, ptxDir: ptxDir);
        (nuint free0, _) = probe0.Context.GetMemoryInfo();
        (nuint free1, _) = probe1.Context.GetMemoryInfo();
        return (free0 / (1024.0 * 1024.0 * 1024.0), free1 / (1024.0 * 1024.0 * 1024.0));
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Engine;
using HartsyInference.Engine.Recipes;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the honesty guarantee: a memory or placement setting that cannot take effect must say so by name, and one that can must not cry wolf.</summary>
/// <remarks>The failure this guards against is silence, which no assertion on a generation's OUTPUT can catch —
/// an ignored setting produces a perfectly good image. So the log line is the contract, and these capture it.</remarks>
public sealed class MemorySupportReportTests
{
    private static List<string> Capture(Action body)
    {
        List<string> lines = [];
        Logs.SetLogger((level, message) => lines.Add($"{level}|{message}"));
        try
        {
            body();
        }
        finally
        {
            Logs.SetLogger(static (level, message) => Console.Error.WriteLine($"[{level}] {message}"));
        }
        return lines;
    }

    /// <summary>A real CPU backend rather than a hand-rolled stub: IBackend's compute surface is ~100 members, and
    /// the report only reads Device, GetVramInfo and StreamingCache — all of which the real thing answers honestly
    /// (CPU genuinely has no streaming cache, which is one of the cases under test).</summary>
    private static IBackend Cpu() => BackendFactory.Create("cpu");

    private static RecipeContext Context(IBackend backend, VramPolicy? policy = null,
        IBackend? shard = null, IBackend? cfgParallel = null, IBackend? textEncoder = null)
        => new RecipeContext
        {
            CheckpointPath = "/dev/null",
            Backend = backend,
            VramPolicy = policy,
            DitShardBackend = shard,
            CfgParallelBackend = cfgParallel,
            TextEncoderBackend = textEncoder,
        };

    /// <summary>The 37-of-39 case: a family that declares nothing must name every configured knob it drops.</summary>
    [Fact]
    public void UndeclaredRecipe_NamesEveryConfiguredCapabilityItDrops()
    {
        using IBackend backend = Cpu();
        using IBackend second = Cpu();
        List<string> lines = Capture(() => MemorySupportReport.Report("Fake", Context(backend,
            VramPolicy.For(VramTier.Aggressive), shard: second, cfgParallel: second, textEncoder: second),
            MemoryCapabilities.None));

        string warnings = string.Join("\n", lines.Where(l => l.StartsWith("Warning|", StringComparison.Ordinal)));
        Assert.Contains("DiT sharding", warnings, StringComparison.Ordinal);
        Assert.Contains("CFG-parallel", warnings, StringComparison.Ordinal);
        Assert.Contains("Component placement", warnings, StringComparison.Ordinal);
        Assert.Contains("Weight streaming", warnings, StringComparison.Ordinal);
    }

    /// <summary>A family that DOES wire a capability must not warn about it — a channel that cries wolf gets ignored.</summary>
    [Fact]
    public void DeclaredCapabilities_ProduceNoWarning()
    {
        using IBackend backend = Cpu();
        using IBackend second = Cpu();
        // Balanced, not Aggressive: Aggressive pins WeightStreaming=On, which a CPU backend genuinely cannot do,
        // so it would (correctly) warn about the DEVICE and this test would be asserting the wrong thing.
        List<string> lines = Capture(() => MemorySupportReport.Report("Fake", Context(backend,
            VramPolicy.For(VramTier.Balanced), shard: second, cfgParallel: second, textEncoder: second),
            MemoryCapabilities.DitSharding | MemoryCapabilities.CfgParallel
            | MemoryCapabilities.ComponentPlacement | MemoryCapabilities.BlockStreaming
            | MemoryCapabilities.HalfPrecisionCaches | MemoryCapabilities.Chunking
            | MemoryCapabilities.PhaseUnload));

        Assert.DoesNotContain(lines, l => l.StartsWith("Warning|", StringComparison.Ordinal));
    }

    /// <summary>Auto is the default everywhere, so it must never warn — otherwise every generation of every model
    /// emits noise and the channel stops being read.</summary>
    [Fact]
    public void AutoPolicy_WarnsAboutNothing()
    {
        using IBackend backend = Cpu();
        List<string> lines = Capture(() => MemorySupportReport.Report("Fake",
            Context(backend, VramPolicy.For(VramTier.Auto)), MemoryCapabilities.None));

        Assert.DoesNotContain(lines, l => l.StartsWith("Warning|", StringComparison.Ordinal));
    }

    /// <summary>A device that cannot stream at all must be blamed instead of the model, which supports it fine.</summary>
    [Fact]
    public void BackendWithoutStreamingCache_BlamesTheDeviceNotTheModel()
    {
        using IBackend cpu = Cpu();
        List<string> lines = Capture(() => MemorySupportReport.Report("Fake",
            Context(cpu, VramPolicy.For(VramTier.Aggressive)), MemoryCapabilities.BlockStreaming));

        string warnings = string.Join("\n", lines.Where(l => l.StartsWith("Warning|", StringComparison.Ordinal)));
        Assert.Contains("no streaming weight cache", warnings, StringComparison.Ordinal);
        Assert.Contains("CUDA", warnings, StringComparison.Ordinal);
    }

    /// <summary>The always-on line: it has to state the tier and what the model can do, whether or not anything is wrong.</summary>
    [Fact]
    public void ResolvedPolicyLine_IsEmittedEvenWhenNothingIsMisconfigured()
    {
        using IBackend backend = Cpu();
        List<string> lines = Capture(() => MemorySupportReport.Report("Fake",
            Context(backend, VramPolicy.For(VramTier.Balanced)), MemoryCapabilities.BlockStreaming));

        string info = string.Join("\n", lines.Where(l => l.StartsWith("Info|", StringComparison.Ordinal)));
        Assert.Contains("[VRAM] Fake:", info, StringComparison.Ordinal);
        Assert.Contains("Balanced", info, StringComparison.Ordinal);
        Assert.Contains("BlockStreaming", info, StringComparison.Ordinal);
    }

    /// <summary>Describe names only what was pinned away from the tier, so the interesting part is not buried.</summary>
    [Fact]
    public void Describe_ListsOnlyLeversThatDifferFromTheTier()
    {
        Assert.Equal("Balanced", VramPolicy.For(VramTier.Balanced).Describe());

        VramPolicy pinned = VramPolicy.For(VramTier.Balanced) with { WeightStreaming = LeverState.On };
        string described = pinned.Describe();
        Assert.Contains("Balanced", described, StringComparison.Ordinal);
        Assert.Contains("WeightStreaming=On", described, StringComparison.Ordinal);
        Assert.DoesNotContain("KeepResident", described, StringComparison.Ordinal);
    }
}

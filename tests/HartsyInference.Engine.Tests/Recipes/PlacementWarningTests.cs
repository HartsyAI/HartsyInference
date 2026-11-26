using System.Collections.Concurrent;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Recipes;
using Xunit;

namespace HartsyInference.Engine.Tests.PlacementChecks;

/// <summary>The placement checks that the engine and the recipes both run: the warning is logged once, the refusal is still enforced by
/// <see cref="UnhonouredBackends.Refuse"/>, and the memory report leaves out the component-placement warning that the refusal already logged.</summary>
public sealed class PlacementWarningTests : IDisposable
{
    private readonly ConcurrentQueue<string> _warnings = new();

    public PlacementWarningTests()
    {
        // Other test classes log through the same sink, so each test filters for its own unique family or recipe name.
        Logs.SetLogger((level, message) => { if (level == LogLevel.Warning) _warnings.Enqueue(message); });
    }

    public void Dispose() => Logs.SetLogger((_, message) => Console.Error.WriteLine(message));

    private IEnumerable<string> WarningsFor(string tag) => _warnings.Where(message => message.Contains(tag, StringComparison.Ordinal));

    [Fact]
    public void TheTextEncoderWarningIsLoggedWhenTheCheckWarns()
    {
        string family = "PlacementCheck-" + Guid.NewGuid().ToString("N");
        using CpuBackend primary = new();
        using CpuBackend te = new();
        RecipeContext context = new() { CheckpointPath = "placement-check", Backend = primary, TextEncoderBackend = te };

        UnhonouredBackends.Refuse(context, family);

        Assert.Single(WarningsFor(family));
    }

    [Fact]
    public void ASecondCheckWithoutWarningLogsNothingAndStillPassesTheTextEncoder()
    {
        string family = "PlacementCheck-" + Guid.NewGuid().ToString("N");
        using CpuBackend primary = new();
        using CpuBackend te = new();
        RecipeContext context = new() { CheckpointPath = "placement-check", Backend = primary, TextEncoderBackend = te };

        UnhonouredBackends.Refuse(context, family, warn: false);

        Assert.Empty(WarningsFor(family));
    }

    [Fact]
    public void ASecondCheckWithoutWarningStillRefusesCfgParallel()
    {
        string family = "PlacementCheck-" + Guid.NewGuid().ToString("N");
        using CpuBackend primary = new();
        using CpuBackend cfg = new();
        RecipeContext context = new() { CheckpointPath = "placement-check", Backend = primary, CfgParallelBackend = cfg };

        Assert.Throws<NotSupportedException>(() => UnhonouredBackends.Refuse(context, family, warn: false));
    }

    [Fact]
    public void TheMemoryReportLeavesOutTheComponentWarningWhenTheRefusalAlreadyLoggedIt()
    {
        string name = "PlacementReport-" + Guid.NewGuid().ToString("N");
        using CpuBackend primary = new();
        using CpuBackend te = new();
        RecipeContext context = new() { CheckpointPath = "placement-check", Backend = primary, TextEncoderBackend = te };

        MemorySupportReport.Report(name, context, MemoryCapabilities.None, refusesUnhonoured: true);

        Assert.DoesNotContain(WarningsFor(name), message => message.Contains("Component placement", StringComparison.Ordinal));
    }

    [Fact]
    public void TheMemoryReportWarnsAboutComponentPlacementForAFamilyThatDoesNotRefuse()
    {
        string name = "PlacementReport-" + Guid.NewGuid().ToString("N");
        using CpuBackend primary = new();
        using CpuBackend te = new();
        RecipeContext context = new() { CheckpointPath = "placement-check", Backend = primary, TextEncoderBackend = te };

        MemorySupportReport.Report(name, context, MemoryCapabilities.None, refusesUnhonoured: false);

        Assert.Single(WarningsFor(name), message => message.Contains("Component placement", StringComparison.Ordinal));
    }
}

using System.Runtime.CompilerServices;
using HartsyInference.Core.Configuration;
using Xunit;

namespace HartsyInference.Tests.Shared;

/// <summary>Turns knobs on for a whole test run, for the GpuIntegration lanes: <c>HARTSY_GPU_TEST_KNOBS="numerics.tileFromDevice=true"</c>
/// (several as <c>id=value;id=value</c>). Linked into each test project that runs the GpuIntegration category. Unset, it does nothing.</summary>
/// <remarks>The values go through <see cref="KnobProfile.TrySet"/>, so they are parsed and range-checked like a <c>--set</c>, then applied as process
/// overrides with <see cref="KnobStore.ApplyRunProfile"/>. A malformed value fails the test host at load, not silently mid-lane.</remarks>
internal static class GpuRunKnobs
{
    internal const string EnvVar = "HARTSY_GPU_TEST_KNOBS";

    [ModuleInitializer]
    internal static void ApplyRunKnobs()
    {
        string? raw = Environment.GetEnvironmentVariable(EnvVar);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return;
        }
        KnobProfile profile = KnobProfile.Create("gpu-test-run");
        foreach (string pair in raw.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int eq = pair.IndexOf('=');
            if (eq <= 0)
            {
                throw new InvalidOperationException($"{EnvVar}: expected id=value, got '{pair}'.");
            }
            if (!profile.TrySet(pair[..eq], pair[(eq + 1)..], out KnobProfile next, out string? error))
            {
                throw new InvalidOperationException($"{EnvVar}: {error}");
            }
            profile = next;
        }
        KnobStore.ApplyRunProfile(profile);
    }
}

/// <summary>Proves the whole-run switch reached the knob store: every id in <c>HARTSY_GPU_TEST_KNOBS</c> resolves with source "run profile",
/// and <c>numerics.tileFromDevice</c>, when it is named, reads the value it was given. Does nothing when the variable is unset.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class GpuRunKnobsTests
{
    [Fact]
    public void TheWholeRunKnobsAreInForceForEveryTestInTheRun()
    {
        string? raw = Environment.GetEnvironmentVariable(GpuRunKnobs.EnvVar);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return;
        }
        foreach (string pair in raw.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            string id = pair[..pair.IndexOf('=')];
            Assert.Equal("run profile", KnobStore.SourceOf(id));
        }
        if (raw.Contains("numerics.tileFromDevice=true", StringComparison.Ordinal))
        {
            Assert.True(EngineKnobs.TileFromDevice.Value == true);
        }
    }

    [Fact]
    public void TheRunProfileSurvivesAReloadAndAClear()
    {
        string? raw = Environment.GetEnvironmentVariable(GpuRunKnobs.EnvVar);
        if (string.IsNullOrWhiteSpace(raw) || !raw.Contains("numerics.tileFromDevice=true", StringComparison.Ordinal))
        {
            return;
        }
        // Paint21 and other suites call KnobFile.Reload and KnobStore.Clear mid-run; neither may drop the whole-run profile.
        KnobFile.Reload();
        Assert.Equal("run profile", KnobStore.SourceOf("numerics.tileFromDevice"));
        Assert.True(EngineKnobs.TileFromDevice.Value == true);
        KnobStore.Clear(EngineKnobs.TileFromDevice);
        Assert.Equal("run profile", KnobStore.SourceOf("numerics.tileFromDevice"));
        Assert.True(EngineKnobs.TileFromDevice.Value == true);
    }
}

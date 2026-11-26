using System.Runtime.CompilerServices;
using HartsyInference.Core.Configuration;
using Xunit;

namespace HartsyInference.Tests.Shared;

/// <summary>Turns knobs on for a whole test run, for the GpuIntegration lanes: <c>HARTSY_GPU_TEST_KNOBS="numerics.tileFromDevice=true"</c>
/// (several as <c>id=value;id=value</c>). Linked into each test project that runs the GpuIntegration category. Unset, it does nothing.</summary>
/// <remarks>The values go through <see cref="KnobProfile.TrySet"/>, so they are parsed and range-checked like a <c>--set</c>, then applied as a
/// run profile with <see cref="KnobStore.ApplyRunProfile"/>. A malformed value, or an id that is retired with no replacement (which would
/// otherwise be dropped silently), fails the test host at load, not mid-lane.</remarks>
internal static class GpuRunKnobs
{
    internal const string EnvVar = "HARTSY_GPU_TEST_KNOBS";

    /// <summary>Parses <c>id=value;...</c>. False, with the reason, for a malformed pair or an id that applied nothing.</summary>
    internal static bool TryParse(string raw, out KnobProfile profile, out string? error)
    {
        profile = KnobProfile.Create("gpu-test-run");
        error = null;
        HashSet<string> named = new(StringComparer.Ordinal);
        foreach (string pair in raw.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int eq = pair.IndexOf('=');
            if (eq <= 0)
            {
                error = $"{EnvVar}: expected id=value, got '{pair}'.";
                return false;
            }
            string id = pair[..eq];
            int before = profile.Count;
            if (!profile.TrySet(id, pair[(eq + 1)..], out KnobProfile next, out error))
            {
                error = $"{EnvVar}: {error}";
                return false;
            }
            if (named.Add(id) && next.Count == before)
            {
                error = $"{EnvVar}: '{id}' is a retired setting with no replacement; nothing would be applied.";
                return false;
            }
            profile = next;
        }
        return true;
    }

    [ModuleInitializer]
    internal static void ApplyRunKnobs()
    {
        string? raw = Environment.GetEnvironmentVariable(EnvVar);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return;
        }
        if (!TryParse(raw, out KnobProfile profile, out string? error))
        {
            throw new InvalidOperationException(error);
        }
        KnobStore.ApplyRunProfile(profile);
    }
}

/// <summary>Serializes the tests that read or change the process-wide knob store, so a Reload or a host Set in one cannot race another's assert.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public static class GlobalKnobStore
{
    public const string Name = "GlobalKnobStore";
}

/// <summary>Proves the whole-run switch reached the knob store: every id in <c>HARTSY_GPU_TEST_KNOBS</c> resolves with source "run profile",
/// and <c>numerics.tileFromDevice</c>, when it is named, reads the value it was given. Does nothing when the variable is unset.</summary>
[Trait("Category", "GpuIntegration")]
[Collection(GlobalKnobStore.Name)]
public sealed class GpuRunKnobsTests
{
    private static string? RunNames(string id)
    {
        string? raw = Environment.GetEnvironmentVariable(GpuRunKnobs.EnvVar);
        return string.IsNullOrWhiteSpace(raw) || !raw.Contains(id, StringComparison.Ordinal) ? null : raw;
    }

    [Fact]
    public void TheWholeRunKnobsAreInForceForEveryTestInTheRun()
    {
        string? raw = Environment.GetEnvironmentVariable(GpuRunKnobs.EnvVar);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return;
        }
        Assert.True(GpuRunKnobs.TryParse(raw, out KnobProfile profile, out string? error), error);
        foreach (string pair in raw.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            Assert.Equal("run profile", KnobStore.SourceOf(pair[..pair.IndexOf('=')]));
        }
        if (raw.Contains("numerics.tileFromDevice=true", StringComparison.Ordinal))
        {
            Assert.True(EngineKnobs.TileFromDevice.Value == true);
        }
    }

    [Fact]
    public void TheRunProfileSurvivesAReloadAndAClear()
    {
        if (RunNames("numerics.tileFromDevice=true") is null)
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

    [Fact]
    public void AHostSetDoesNotBeatTheRunProfile()
    {
        if (RunNames("numerics.tileFromDevice=true") is null)
        {
            return;
        }
        // Order: request profile, run profile, host overrides. A host Set to off must not turn the lane's knob back off.
        KnobStore.Set(EngineKnobs.TileFromDevice, false);
        Assert.Equal("run profile", KnobStore.SourceOf("numerics.tileFromDevice"));
        Assert.True(EngineKnobs.TileFromDevice.Value == true);
    }

    [Fact]
    public void ARetiredIdWithNoReplacementIsRefusedNotDropped()
    {
        // numerics.sdpaV2 is retired with no replacement: TrySet accepts it and applies nothing, so the parse must refuse it.
        Assert.False(GpuRunKnobs.TryParse("numerics.sdpaV2=1", out _, out string? error));
        Assert.Contains("retired", error, StringComparison.Ordinal);
    }
}

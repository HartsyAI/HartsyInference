using System.Text.Json;
using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Execution;
using HartsyInference.BenchmarkRunner.Serialization;
using HartsyInference.Core.Configuration;
using Xunit;

namespace HartsyInference.BenchmarkRunner.Tests;

/// <summary>Campaign provenance must cover the real engine registry, including nullable defaults.</summary>
public sealed class HardwareTests
{
    [Fact]
    public void ResumeAcceptsRoundTrippedProvenanceButRejectsChangedValues()
    {
        EnvironmentRecord original = Hardware.Capture(Hardware.Probe("cpu"));
        string json = JsonSerializer.Serialize(original, BenchJson.Default.EnvironmentRecord);
        EnvironmentRecord restored = JsonSerializer.Deserialize(json, BenchJson.Default.EnvironmentRecord)!;
        Assert.True(Campaign.SameEnvironment(restored, original));
        Assert.False(Campaign.SameEnvironment(restored with { CpuCount = original.CpuCount + 1 }, original));
        restored.Settings[restored.Settings.Keys.First()] = "changed";
        Assert.False(Campaign.SameEnvironment(restored, original));
    }

    [Fact]
    public void NullDefaultsSuppressLocalOverridesForTheCampaign()
    {
        KnobStore.Set(EngineKnobs.Fp8Native, true);
        try
        {
            using (Hardware.Defaults(out _).Push())
                Assert.Null(EngineKnobs.Fp8Native.Value);
            Assert.True(EngineKnobs.Fp8Native.Value);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.Fp8Native);
        }
    }

    [Fact]
    public void DefaultsPinEveryRegisteredSettingIncludingNulls()
    {
        KnobProfile profile = Hardware.Defaults(out SortedDictionary<string, string> settings);
        Assert.Equal(KnobRegistry.All.Count(), profile.Count);
        Assert.Equal(profile.Count, settings.Count);
        using IDisposable scope = profile.Push();
        foreach (object knob in KnobRegistry.All)
        {
            switch (knob)
            {
                case Knob<bool?> value:
                    Assert.Equal(value.Default, value.Value);
                    break;
                case Knob<int?> value:
                    Assert.Equal(value.Default, value.Value);
                    break;
                case Knob<float?> value:
                    Assert.Equal(value.Default, value.Value);
                    break;
            }
        }
    }
}

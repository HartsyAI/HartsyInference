using HartsyInference.Core.Configuration;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Forces <c>AudioRuntime</c>'s host-RAM eviction path for the duration of a scope by raising <c>vram.audioEvictBelowGb</c> far above any real machine, then restores the knob exactly (previous override or cleared). The floor only reads on Linux (<c>/proc/meminfo</c>), so tests that assert an eviction happened guard on <see cref="HostPressureObservable"/>.</summary>
internal static class AudioEvictionPressure
{
    /// <summary>True where the runtime can read free host RAM and therefore act on the forced floor.</summary>
    public static bool HostPressureObservable { get; } = File.Exists("/proc/meminfo");

    /// <summary>Raises the floor; dispose restores the knob.</summary>
    public static IDisposable Force()
    {
        bool hadOverride = KnobStore.HasOverride(EngineKnobs.AudioEvictBelowGb);
        long previous = EngineKnobs.AudioEvictBelowGb.Value;
        KnobStore.Set(EngineKnobs.AudioEvictBelowGb, 1_000_000_000L);
        return new Restore(hadOverride, previous);
    }

    private sealed class Restore(bool hadOverride, long previous) : IDisposable
    {
        public void Dispose()
        {
            if (hadOverride)
            {
                KnobStore.Set(EngineKnobs.AudioEvictBelowGb, previous);
            }
            else
            {
                KnobStore.Clear(EngineKnobs.AudioEvictBelowGb);
            }
        }
    }
}

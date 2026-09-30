using HartsyInference.Core.Configuration;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Forces <c>AudioRuntime</c>'s host-RAM eviction path for the duration of a scope by raising <c>vram.audioEvictBelowGb</c> far above any real machine, then restores the knob exactly (previous override or cleared). The floor only reads on Linux (<c>/proc/meminfo</c>), so tests that assert an eviction happened guard on <see cref="HostPressureObservable"/>.</summary>
internal static class AudioEvictionPressure
{
    /// <summary>True where the runtime can read free host RAM and therefore act on the forced floor.</summary>
    public static bool HostPressureObservable { get; } = File.Exists("/proc/meminfo");

    /// <summary>Raises the floor; dispose restores the knob.</summary>
    public static IDisposable Force() => SetFloor(1_000_000_000L);

    /// <summary>Drops the floor to 1 GB (the knob coerces zero back to its default) so a box genuinely short of RAM
    /// cannot evict a test's fake runners; dispose restores the knob.</summary>
    public static IDisposable Relax() => SetFloor(1L);

    private static IDisposable SetFloor(long floorGb)
    {
        bool hadOverride = KnobStore.HasOverride(EngineKnobs.AudioEvictBelowGb);
        long previous = EngineKnobs.AudioEvictBelowGb.Value;
        KnobStore.Set(EngineKnobs.AudioEvictBelowGb, floorGb);
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

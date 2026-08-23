using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Layer-activation dump hooks for the Hunyuan3D reference-diff harness; a no-op unless <c>HARTSYINFERENCE_HUNYUAN3D_DEBUG_DIR</c> is set, so it's free on the hot path in normal runs.</summary>
/// <remarks>Deliberately unwired: the parity pass it feeds is planned but not yet run (docs/Research/HUNYUAN3D_2_ARCHITECTURE.md names these hooks as its mechanism), unlike the sibling *DebugDump classes whose models are already validated.</remarks>
public static unsafe class Hunyuan3DDebugDump
{
    private static readonly object Lock = new();
    private static string? s_dir;
    private static bool s_initialized;

    /// <summary>True when a dump directory is configured.</summary>
    public static bool Enabled
    {
        get
        {
            if (!s_initialized)
                lock (Lock)
                    if (!s_initialized)
                    {
                        s_dir = Environment.GetEnvironmentVariable("HARTSYINFERENCE_HUNYUAN3D_DEBUG_DIR");
                        if (!string.IsNullOrEmpty(s_dir)) Directory.CreateDirectory(s_dir);
                        s_initialized = true;
                    }
            return !string.IsNullOrEmpty(s_dir);
        }
    }

    /// <summary>Writes <paramref name="t"/> as a raw F32 blob <c>&lt;tag&gt;.bin</c> when enabled.</summary>
    public static void Dump(string tag, Tensor t)
    {
        if (!Enabled) return;
        long n = t.ElementCount;
        byte[] bytes = new byte[n * 4];
        new ReadOnlySpan<byte>((byte*)t.DataPointer, (int)(n * 4)).CopyTo(bytes);
        File.WriteAllBytes(Path.Combine(s_dir!, $"{tag}.bin"), bytes);
    }
}

using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Layer-activation dump hooks for the Hunyuan3D reference-diff harness (the project's standard
/// 🔧→✅ loop). Enabled by setting <c>HARTSYINFERENCE_HUNYUAN3D_DEBUG_DIR</c> to an output directory; each
/// <see cref="Dump"/> writes a raw little-endian F32 blob named <c>&lt;tag&gt;.bin</c> that
/// <c>diff_hunyuan3d_layers.py</c> compares against the Python reference. No-op when the env var is unset, so
/// it's free on the hot path in normal runs.</summary>
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

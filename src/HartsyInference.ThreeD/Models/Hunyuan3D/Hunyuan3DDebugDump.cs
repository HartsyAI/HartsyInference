using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Layer-activation dump hooks for the Hunyuan3D reference-diff harness; a no-op unless
/// <c>diagnostics.hunyuan3dDebugDir</c> is set, so it's free on the hot path in normal runs.</summary>
/// <remarks>Deliberately unwired: the parity pass it feeds is planned but not yet run (docs/Research/HUNYUAN3D_2_ARCHITECTURE.md names these hooks as its mechanism), unlike the sibling *DebugDump classes whose models are already validated.</remarks>
public static unsafe class Hunyuan3DDebugDump
{
    /// <summary>Dump root directory, or null when dumping is disabled.</summary>
    /// <remarks>Resolved per access. A double-checked one-shot cache held it before, which froze a Runtime
    /// setting at whatever the first <see cref="Enabled"/> read saw — the same defect as the sibling sinks, in
    /// the one shape a lint over field initializers cannot see.</remarks>
    private static string? Dir
    {
        get
        {
            string? dir = EngineKnobs.Hunyuan3dDebugDir.Value;
            return string.IsNullOrEmpty(dir) ? null : dir;
        }
    }

    /// <summary>True when a dump directory is configured.</summary>
    public static bool Enabled => Dir is not null;

    /// <summary>Writes <paramref name="t"/> as a raw F32 blob <c>&lt;tag&gt;.bin</c> when enabled.</summary>
    public static void Dump(string tag, Tensor t)
    {
        if (Dir is not string dir) return;
        Directory.CreateDirectory(dir);
        long n = t.ElementCount;
        byte[] bytes = new byte[n * 4];
        new ReadOnlySpan<byte>((byte*)t.DataPointer, (int)(n * 4)).CopyTo(bytes);
        File.WriteAllBytes(Path.Combine(dir, $"{tag}.bin"), bytes);
    }
}

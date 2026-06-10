using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Optional layer-by-layer debug dump for LTX-Video. When <c>LTX_DEBUG_DIR</c> is set, writes each named tensor as raw little-endian F32 to <c>{dir}/layers/{safe_name}.bin</c>; zero-cost otherwise. Mirrors <see cref="LanceDebugDump"/> — for first-run Python layer-diff validation.</summary>
internal static unsafe class LtxVideoDebugDump
{
    private static readonly string? s_dumpDir = ResolveDir();
    private static bool s_initialized;
    private static readonly object s_lock = new();

    private static string? ResolveDir()
    {
        string? dir = Environment.GetEnvironmentVariable("LTX_DEBUG_DIR");
        return string.IsNullOrEmpty(dir) ? null : dir;
    }

    private static void EnsureInit()
    {
        if (s_initialized) return;
        lock (s_lock)
        {
            if (s_initialized) return;
            if (s_dumpDir is not null) Directory.CreateDirectory(Path.Combine(s_dumpDir, "layers"));
            s_initialized = true;
        }
    }

    public static void Dump(string name, Tensor t)
    {
        if (s_dumpDir is null) return;
        EnsureInit();
        WriteRawF32(Path.Combine(s_dumpDir, "layers", name.Replace('.', '_') + ".bin"), t);
    }

    public static void DumpOutput(Tensor t)
    {
        if (s_dumpDir is null) return;
        EnsureInit();
        WriteRawF32(Path.Combine(s_dumpDir, "output_velocity.bin"), t);
    }

    private static void WriteRawF32(string path, Tensor t)
    {
        long count = t.Shape.ElementCount;
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        byte[] buffer = new byte[count * sizeof(float)];
        fixed (byte* dst = buffer) Buffer.MemoryCopy((float*)f32.DataPointer, dst, buffer.Length, buffer.Length);
        File.WriteAllBytes(path, buffer);
        if (!ReferenceEquals(f32, t)) f32.Dispose();
    }
}

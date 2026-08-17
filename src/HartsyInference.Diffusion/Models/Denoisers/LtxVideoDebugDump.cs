using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Optional layer-by-layer debug dump for LTX-Video. When <c>LTX_DEBUG_DIR</c> is set, writes each named
/// tensor as raw little-endian F32 to <c>{dir}/layers/{safe_name}.bin</c>; zero-cost otherwise. Mirrors <see
/// cref="LanceDebugDump"/> — for first-run Python layer-diff validation.</summary>
internal static unsafe class LtxVideoDebugDump
{
    private static readonly string? _dumpDir = ResolveDir();
    private static bool _initialized;
    private static readonly object _lock = new();

    private static string? ResolveDir()
    {
        string? dir = Environment.GetEnvironmentVariable("LTX_DEBUG_DIR");
        return string.IsNullOrEmpty(dir) ? null : dir;
    }

    private static void EnsureInit()
    {
        if (_initialized) return;
        lock (_lock)
        {
            if (_initialized) return;
            if (_dumpDir is not null) Directory.CreateDirectory(Path.Combine(_dumpDir, "layers"));
            _initialized = true;
        }
    }

    public static void Dump(string name, Tensor t)
    {
        if (_dumpDir is null) return;
        EnsureInit();
        WriteRawF32(Path.Combine(_dumpDir, "layers", name.Replace('.', '_') + ".bin"), t);
    }

    public static void DumpOutput(Tensor t)
    {
        if (_dumpDir is null) return;
        EnsureInit();
        WriteRawF32(Path.Combine(_dumpDir, "output_velocity.bin"), t);
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

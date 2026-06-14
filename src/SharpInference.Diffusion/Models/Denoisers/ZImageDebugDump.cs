using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Optional layer-by-layer debug dump for Z-Image. When the environment variable <c>Z_IMAGE_DEBUG_DIR</c> is set, the transformer writes each named tensor as raw F32 to that directory under <c>layers/&lt;safe_name&gt;.bin</c>. Disabled (zero-cost) otherwise. Used to diff against a Python (diffusers) reference produced by <c>dump_zimage_full_forward.py</c>.</summary>
internal static unsafe class ZImageDebugDump
{
    private static readonly string? _dumpDir = ResolveDir();
    private static bool _initialized;
    private static readonly object _lock = new();

    public static bool Enabled => _dumpDir is not null;

    private static string? ResolveDir()
    {
        string? dir = Environment.GetEnvironmentVariable("Z_IMAGE_DEBUG_DIR");
        return string.IsNullOrEmpty(dir) ? null : dir;
    }

    private static void EnsureInit()
    {
        if (_initialized) return;
        lock (_lock)
        {
            if (_initialized) return;
            if (_dumpDir is not null)
                Directory.CreateDirectory(Path.Combine(_dumpDir, "layers"));
            _initialized = true;
        }
    }

    /// <summary>Writes the tensor's data as raw F32 little-endian to <c>{dumpDir}/layers/{safeName}.bin</c>. <paramref name="name"/> is the diffusers-side capture name (e.g. <c>"layers.5"</c>); dots are replaced with underscores in the filename.</summary>
    public static void Dump(string name, Tensor t)
    {
        if (_dumpDir is null) return;
        EnsureInit();

        string safe = name.Replace('.', '_');
        string path = Path.Combine(_dumpDir, "layers", safe + ".bin");
        WriteRawF32(path, t);
    }

    /// <summary>Writes the final transformer output (velocity) at <c>{dumpDir}/output_velocity.bin</c>.</summary>
    public static void DumpOutput(Tensor t)
    {
        if (_dumpDir is null) return;
        EnsureInit();
        WriteRawF32(Path.Combine(_dumpDir, "output_velocity.bin"), t);
    }

    private static void WriteRawF32(string path, Tensor t)
    {
        long count = t.Shape.ElementCount;
        if (t.DType == DType.F32)
        {
            byte[] buffer = new byte[count * sizeof(float)];
            fixed (byte* dst = buffer)
            {
                Buffer.MemoryCopy((float*)t.DataPointer, dst, buffer.Length, buffer.Length);
            }
            File.WriteAllBytes(path, buffer);
        }
        else
        {
            // Cast through F32 for diff comparison.
            using Tensor cast = t.CastTo(DType.F32);
            byte[] buffer = new byte[count * sizeof(float)];
            fixed (byte* dst = buffer)
            {
                Buffer.MemoryCopy((float*)cast.DataPointer, dst, buffer.Length, buffer.Length);
            }
            File.WriteAllBytes(path, buffer);
        }
    }
}

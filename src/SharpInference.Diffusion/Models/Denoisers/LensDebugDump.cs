using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Optional layer-by-layer debug dump for Microsoft Lens. When <c>LENS_DEBUG_DIR</c> is set, the transformer writes each named tensor as raw F32 little-endian to that directory under <c>layers/&lt;safe_name&gt;.bin</c>. Disabled (zero-cost) otherwise. Mirrors <see cref="QwenImageDebugDump"/> / <see cref="Sd3DebugDump"/> — used by <c>tests/python-reference/dump_lens_full_forward.py</c> + <c>diff_lens_layers.py</c> + <c>LensDiffTests</c> to validate the transformer against the diffusers reference.</summary>
internal static unsafe class LensDebugDump
{
    private static readonly string? _dumpDir = ResolveDir();
    private static bool _initialized;
    private static readonly object _lock = new();

    public static bool Enabled => _dumpDir is not null;

    private static string? ResolveDir()
    {
        string? dir = Environment.GetEnvironmentVariable("LENS_DEBUG_DIR");
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

    /// <summary>Writes the tensor's data as raw F32 little-endian to <c>{dumpDir}/layers/{safeName}.bin</c>.</summary>
    public static void Dump(string name, Tensor t)
    {
        if (_dumpDir is null) return;
        EnsureInit();

        string safe = name.Replace('.', '_');
        string path = Path.Combine(_dumpDir, "layers", safe + ".bin");
        WriteRawF32(path, t);
    }

    /// <summary>Writes the final transformer output (predicted velocity) at <c>{dumpDir}/output_velocity.bin</c>.</summary>
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

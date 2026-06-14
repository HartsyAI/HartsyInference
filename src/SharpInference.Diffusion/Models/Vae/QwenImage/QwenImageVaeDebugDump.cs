using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae.QwenImage;

/// <summary>Optional layer-by-layer debug dump for the Qwen-Image VAE decoder. When
/// <c>QWEN_VAE_DEBUG_DIR</c> is set, the decoder writes each named stage tensor as raw F32 to
/// <c>{dumpDir}/layers/{safe_name}.bin</c>. Disabled (zero-cost) otherwise. Used to diff against
/// the Python <see cref="diffusers"/> reference produced by <c>dump_qwen_image_vae.py</c>.
/// Mirrors <c>AnimaDebugDump</c>.</summary>
internal static unsafe class QwenImageVaeDebugDump
{
    private static readonly string? _dumpDir = ResolveDir();
    private static bool _initialized;
    private static readonly object _lock = new();

    public static bool Enabled => _dumpDir is not null;

    private static string? ResolveDir()
    {
        string? dir = Environment.GetEnvironmentVariable("QWEN_VAE_DEBUG_DIR");
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

        string safe = name.Replace('.', '_').Replace('/', '_');
        string path = Path.Combine(_dumpDir, "layers", safe + ".bin");
        WriteRawF32(path, t);
    }

    /// <summary>Writes the final decoded image at <c>{dumpDir}/output_image.bin</c>.</summary>
    public static void DumpOutput(Tensor t)
    {
        if (_dumpDir is null) return;
        EnsureInit();
        WriteRawF32(Path.Combine(_dumpDir, "output_image.bin"), t);
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

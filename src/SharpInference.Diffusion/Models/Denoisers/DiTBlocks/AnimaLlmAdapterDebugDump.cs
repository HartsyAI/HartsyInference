using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Optional layer-by-layer debug dump for the Anima <c>net.llm_adapter</c>. When
/// <c>ANIMA_LLM_ADAPTER_DEBUG_DIR</c> is set, the adapter writes each named stage tensor as raw F32 to
/// <c>{dumpDir}/layers/{safe_name}.bin</c>. Disabled (zero-cost) otherwise. Used to diff against the
/// Python reference produced by <c>dump_anima_llm_adapter.py</c>. Mirrors <c>QwenImageVaeDebugDump</c>.</summary>
internal static unsafe class AnimaLlmAdapterDebugDump
{
    private static readonly string? s_dumpDir = ResolveDir();
    private static bool s_initialized;
    private static readonly object s_lock = new();

    public static bool Enabled => s_dumpDir is not null;

    private static string? ResolveDir()
    {
        string? dir = Environment.GetEnvironmentVariable("ANIMA_LLM_ADAPTER_DEBUG_DIR");
        return string.IsNullOrEmpty(dir) ? null : dir;
    }

    private static void EnsureInit()
    {
        if (s_initialized) return;
        lock (s_lock)
        {
            if (s_initialized) return;
            if (s_dumpDir is not null)
                Directory.CreateDirectory(Path.Combine(s_dumpDir, "layers"));
            s_initialized = true;
        }
    }

    /// <summary>Writes <paramref name="t"/>'s data as raw F32 little-endian to <c>{dumpDir}/layers/{safeName}.bin</c>.</summary>
    public static void Dump(string name, Tensor t)
    {
        if (s_dumpDir is null) return;
        EnsureInit();
        string safe = name.Replace('.', '_').Replace('/', '_');
        string path = Path.Combine(s_dumpDir, "layers", safe + ".bin");
        WriteRawF32(path, t);
    }

    private static void WriteRawF32(string path, Tensor t)
    {
        long count = t.Shape.ElementCount;
        byte[] buffer = new byte[count * sizeof(float)];
        if (t.DType == DType.F32)
        {
            fixed (byte* dst = buffer)
                Buffer.MemoryCopy((float*)t.DataPointer, dst, buffer.Length, buffer.Length);
        }
        else
        {
            using Tensor cast = t.CastTo(DType.F32);
            fixed (byte* dst = buffer)
                Buffer.MemoryCopy((float*)cast.DataPointer, dst, buffer.Length, buffer.Length);
        }
        File.WriteAllBytes(path, buffer);
    }
}

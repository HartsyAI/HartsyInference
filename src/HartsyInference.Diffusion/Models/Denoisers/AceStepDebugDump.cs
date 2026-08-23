using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Optional layer-by-layer debug dump for the ACE-Step v1 DiT. When <c>ACE_STEP_DEBUG_DIR</c> is set, the
/// transformer writes each named tensor as raw little-endian F32 to <c>{dir}/layers/{safe_name}.bin</c>; zero-cost
/// otherwise. Used to diff against the Python reference (<c>dump_ace_step_dit.py</c> / <c>diff_ace_step_layers.py</c>).
/// Same pattern as <see cref="Ideogram4DebugDump"/>.</summary>
internal static unsafe class AceStepDebugDump
{
    // Resolved per call (NOT cached): the DiT/DCAE/vocoder parity tests each set ACE_STEP_DEBUG_DIR to their own
    // dir, and a static-cached value would pin whichever test ran first — so the others would dump to the wrong
    // place and fail. Reading the env each call keeps every test's dumps in its own directory.
    private static string? CurrentDir()
    {
        string? dir = Environment.GetEnvironmentVariable("ACE_STEP_DEBUG_DIR");
        return string.IsNullOrEmpty(dir) ? null : dir;
    }

    /// <summary>Writes the tensor's data as raw F32 to <c>{dumpDir}/layers/{safeName}.bin</c>.</summary>
    public static void Dump(string name, Tensor t)
    {
        string? dir = CurrentDir();
        if (dir is null) return;
        Directory.CreateDirectory(Path.Combine(dir, "layers"));
        WriteRawF32(Path.Combine(dir, "layers", name.Replace('.', '_') + ".bin"), t);
    }

    /// <summary>Writes the final velocity at <c>{dumpDir}/output_velocity.bin</c>.</summary>
    public static void DumpOutput(Tensor t)
    {
        string? dir = CurrentDir();
        if (dir is null) return;
        Directory.CreateDirectory(Path.Combine(dir, "layers"));
        WriteRawF32(Path.Combine(dir, "output_velocity.bin"), t);
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

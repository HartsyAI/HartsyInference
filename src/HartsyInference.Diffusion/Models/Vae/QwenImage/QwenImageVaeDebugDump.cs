using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Models.Vae.QwenImage;

/// <summary>Optional layer-by-layer debug dump for the Qwen-Image VAE decoder. When
/// <c>QWEN_VAE_DEBUG_DIR</c> is set, the decoder writes each named stage tensor as raw F32 to
/// <c>{dumpDir}/layers/{safe_name}.bin</c>. Disabled (zero-cost) otherwise. Used to diff against
/// the Python <see cref="diffusers"/> reference produced by <c>dump_qwen_image_vae.py</c>.
/// Mirrors <c>AnimaDebugDump</c>.</summary>
internal static class QwenImageVaeDebugDump
{
    private static readonly DebugDumpSink _sink = new DebugDumpSink("QWEN_VAE_DEBUG_DIR");

    public static bool Enabled => _sink.Enabled;

    /// <summary>Writes the tensor's data as raw F32 little-endian to <c>{dumpDir}/layers/{safeName}.bin</c>.</summary>
    public static void Dump(string name, Tensor t)
    {
        string? dir = _sink.Dir;
        if (dir is null) return;
        _sink.EnsureLayersDir(dir);
        _sink.WriteRawF32(Path.Combine(dir, "layers", name.Replace('.', '_').Replace('/', '_') + ".bin"), t);
    }

    /// <summary>Writes the final decoded image at <c>{dumpDir}/output_image.bin</c>.</summary>
    public static void DumpOutput(Tensor t)
    {
        string? dir = _sink.Dir;
        if (dir is null) return;
        _sink.EnsureLayersDir(dir);
        _sink.WriteRawF32(Path.Combine(dir, "output_image.bin"), t);
    }
}

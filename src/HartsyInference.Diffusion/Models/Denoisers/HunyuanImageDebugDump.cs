using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Optional layer-by-layer debug dump for Hunyuan Image. When the environment variable <c>HUNYUAN_IMAGE_DEBUG_DIR</c> is set, the transformer writes each named tensor as raw F32 little-endian to that directory under <c>layers/&lt;safe_name&gt;.bin</c>. Disabled (zero-cost) otherwise. Mirrors <see cref="Sd3DebugDump"/> / <see cref="QwenImageDebugDump"/> — used to diff against a Python diffusers reference dump.</summary>
internal static class HunyuanImageDebugDump
{
    private static readonly DebugDumpSink _sink = new DebugDumpSink("HUNYUAN_IMAGE_DEBUG_DIR");

    public static bool Enabled => _sink.Enabled;

    /// <summary>Writes the tensor's data as raw F32 little-endian to <c>{dumpDir}/layers/{safeName}.bin</c>.</summary>
    public static void Dump(string name, Tensor t)
    {
        string? dir = _sink.Dir;
        if (dir is null) return;
        _sink.EnsureLayersDir(dir);
        _sink.WriteRawF32(Path.Combine(dir, "layers", name.Replace('.', '_') + ".bin"), t);
    }

    /// <summary>Writes the final transformer output (velocity) at <c>{dumpDir}/output_velocity.bin</c>.</summary>
    public static void DumpOutput(Tensor t)
    {
        string? dir = _sink.Dir;
        if (dir is null) return;
        _sink.EnsureLayersDir(dir);
        _sink.WriteRawF32(Path.Combine(dir, "output_velocity.bin"), t);
    }
}

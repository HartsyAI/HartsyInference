using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Optional layer-by-layer debug dump for Microsoft Lens. When <c>LENS_DEBUG_DIR</c> is set, the transformer writes each named tensor as raw F32 little-endian to that directory under <c>layers/&lt;safe_name&gt;.bin</c>. Disabled (zero-cost) otherwise. Mirrors <see cref="QwenImageDebugDump"/> / <see cref="Sd3DebugDump"/> — used by <c>tests/python-reference/dump_lens_full_forward.py</c> + <c>diff_lens_layers.py</c> + <c>LensDiffTests</c> to validate the transformer against the diffusers reference.</summary>
internal static class LensDebugDump
{
    private static readonly DebugDumpSink _sink = new DebugDumpSink("LENS_DEBUG_DIR");

    public static bool Enabled => _sink.Enabled;

    /// <summary>Writes the tensor's data as raw F32 little-endian to <c>{dumpDir}/layers/{safeName}.bin</c>.</summary>
    public static void Dump(string name, Tensor t)
    {
        string? dir = _sink.Dir;
        if (dir is null) return;
        _sink.EnsureLayersDir(dir);
        _sink.WriteRawF32(Path.Combine(dir, "layers", name.Replace('.', '_') + ".bin"), t);
    }

    /// <summary>Writes the final transformer output (predicted velocity) at <c>{dumpDir}/output_velocity.bin</c>.</summary>
    public static void DumpOutput(Tensor t)
    {
        string? dir = _sink.Dir;
        if (dir is null) return;
        _sink.EnsureLayersDir(dir);
        _sink.WriteRawF32(Path.Combine(dir, "output_velocity.bin"), t);
    }
}

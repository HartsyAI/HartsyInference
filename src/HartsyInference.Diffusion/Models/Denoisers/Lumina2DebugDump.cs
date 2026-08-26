using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Optional layer-by-layer debug dump for Lumina-Image-2.0. When the environment variable <c>LUMINA2_DEBUG_DIR</c> is set, the transformer writes each named tensor as raw F32 to that directory under <c>layers/&lt;safe_name&gt;.bin</c>. Disabled (zero-cost) otherwise. Used to diff against a Python (diffusers) reference produced by an equivalent <c>dump_lumina2_full_forward.py</c>.</summary>
internal static class Lumina2DebugDump
{
    private static readonly DebugDumpSink _sink = new DebugDumpSink(EngineKnobs.Lumina2DebugDir);

    /// <summary>Writes the tensor's data as raw F32 little-endian to <c>{dumpDir}/layers/{safeName}.bin</c>. <paramref name="name"/> is the diffusers-side capture name (e.g. <c>"layers.5"</c>); dots are replaced with underscores in the filename.</summary>
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

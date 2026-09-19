using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Optional layer-by-layer debug dump for the ACE-Step v1 DiT. When <c>diagnostics.aceStepDebugDir</c> is set, the
/// transformer writes each named tensor as raw little-endian F32 to <c>{dir}/layers/{safe_name}.bin</c>; zero-cost
/// otherwise. Used to diff against the Python reference (<c>dump_ace_step_dit.py</c> / <c>diff_ace_step_layers.py</c>).
/// Same pattern as <see cref="Ideogram4DebugDump"/>.</summary>
internal static class AceStepDebugDump
{
    private static readonly DebugDumpSink _sink = new DebugDumpSink(EngineKnobs.AceStepDebugDir);

    /// <summary>Writes the tensor's data as raw F32 to <c>{dumpDir}/layers/{safeName}.bin</c>.</summary>
    public static void Dump(string name, Tensor t)
    {
        string? dir = _sink.Dir;
        if (dir is null) return;
        _sink.EnsureLayersDir(dir);
        _sink.WriteRawF32(Path.Combine(dir, "layers", name.Replace('.', '_') + ".bin"), t);
    }

    /// <summary>Writes the final velocity at <c>{dumpDir}/output_velocity.bin</c>.</summary>
    public static void DumpOutput(Tensor t)
    {
        string? dir = _sink.Dir;
        if (dir is null) return;
        _sink.EnsureLayersDir(dir);
        _sink.WriteRawF32(Path.Combine(dir, "output_velocity.bin"), t);
    }
}

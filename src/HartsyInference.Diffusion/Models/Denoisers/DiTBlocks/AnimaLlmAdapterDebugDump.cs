using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Optional layer-by-layer debug dump for the Anima <c>net.llm_adapter</c>. When
/// <c>ANIMA_LLM_ADAPTER_DEBUG_DIR</c> is set, the adapter writes each named stage tensor as raw F32 to
/// <c>{dumpDir}/layers/{safe_name}.bin</c>. Disabled (zero-cost) otherwise. Used to diff against the
/// Python reference produced by <c>dump_anima_llm_adapter.py</c>. Mirrors <c>QwenImageVaeDebugDump</c>.</summary>
internal static class AnimaLlmAdapterDebugDump
{
    private static readonly DebugDumpSink _sink = new DebugDumpSink("ANIMA_LLM_ADAPTER_DEBUG_DIR");

    public static bool Enabled => _sink.Enabled;

    /// <summary>Writes <paramref name="t"/>'s data as raw F32 little-endian to <c>{dumpDir}/layers/{safeName}.bin</c>.</summary>
    public static void Dump(string name, Tensor t)
    {
        string? dir = _sink.Dir;
        if (dir is null) return;
        _sink.EnsureLayersDir(dir);
        _sink.WriteRawF32(Path.Combine(dir, "layers", name.Replace('.', '_').Replace('/', '_') + ".bin"), t);
    }
}

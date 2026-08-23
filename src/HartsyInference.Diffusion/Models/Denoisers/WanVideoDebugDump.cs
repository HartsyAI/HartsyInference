using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Optional layer-by-layer debug dump for Wan-family DiTs. When <c>WAN_DEBUG_DIR</c> is set, writes each
/// named tensor as raw little-endian F32 to <c>{dir}/layers/{tag}_{safe_name}.bin</c> plus a <c>shapes.txt</c>
/// sidecar; zero-cost otherwise. The tag (initially <c>WAN_DEBUG_TAG</c>, overridable via <see cref="SetTag"/>)
/// keeps the CFG cond/uncond forwards of the same step from colliding. For Python layer-diff validation.</summary>
public static unsafe class WanVideoDebugDump
{
    private static readonly DebugDumpSink _sink = new DebugDumpSink("WAN_DEBUG_DIR");
    private static string _tag = Environment.GetEnvironmentVariable("WAN_DEBUG_TAG") is { Length: > 0 } t ? t + "_" : "";
    private static readonly object _lock = new();
    private static readonly HashSet<string> _shapesWritten = new();

    /// <summary>True when <c>WAN_DEBUG_DIR</c> is set — callers gate any dump-only tensor prep on this.</summary>
    public static bool Enabled => _sink.Enabled;

    /// <summary>Prefixes subsequent dump names with <c>{tag}_</c> (null/empty clears). The CFG pipeline sets
    /// <c>cond</c>/<c>uncond</c> around each branch forward. No-op when dumping is disabled.</summary>
    public static void SetTag(string? tag)
    {
        if (!_sink.Enabled) return;
        _tag = string.IsNullOrEmpty(tag) ? "" : tag + "_";
    }

    public static void Dump(string name, Tensor t)
    {
        string? dir = _sink.Dir;
        if (dir is null) return;
        _sink.EnsureLayersDir(dir);
        string safeName = (_tag + name).Replace('.', '_');
        _sink.WriteRawF32(Path.Combine(dir, "layers", safeName + ".bin"), t);
        // Shape sidecar so the Python layer-diff reference knows how to reshape each raw-F32 blob.
        int[] dims = new int[t.Shape.Rank];
        for (int i = 0; i < dims.Length; i++) dims[i] = (int)t.Shape[i];
        AppendShape(dir, safeName, dims);
    }

    /// <summary>Dumps a small host-side float array (per-group timesteps, scalars) as a rank-1 blob.</summary>
    public static void DumpValues(string name, ReadOnlySpan<float> values)
    {
        string? dir = _sink.Dir;
        if (dir is null) return;
        _sink.EnsureLayersDir(dir);
        string safeName = (_tag + name).Replace('.', '_');
        byte[] buffer = new byte[values.Length * sizeof(float)];
        fixed (float* src = values)
        fixed (byte* dst = buffer) Buffer.MemoryCopy(src, dst, buffer.Length, buffer.Length);
        File.WriteAllBytes(Path.Combine(dir, "layers", safeName + ".bin"), buffer);
        AppendShape(dir, safeName, [values.Length]);
    }

    public static void DumpOutput(Tensor t)
    {
        string? dir = _sink.Dir;
        if (dir is null) return;
        _sink.EnsureLayersDir(dir);
        _sink.WriteRawF32(Path.Combine(dir, _tag + "output_velocity.bin"), t);
    }

    private static void AppendShape(string dir, string safeName, int[] dims)
    {
        lock (_lock)
        {
            // Multi-step runs overwrite the same .bin each step — keep shapes.txt to one line per name.
            if (_shapesWritten.Add(safeName))
                File.AppendAllText(Path.Combine(dir, "shapes.txt"), $"{safeName} {string.Join(",", dims)}\n");
        }
    }
}

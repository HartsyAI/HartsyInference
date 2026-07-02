using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Optional layer-by-layer debug dump for Wan-Video. When <c>WAN_DEBUG_DIR</c> is set, writes each named tensor as raw little-endian F32 to <c>{dir}/layers/{safe_name}.bin</c>; zero-cost otherwise. For first-run Python layer-diff validation.</summary>
internal static unsafe class WanVideoDebugDump
{
    private static readonly string? _dumpDir = Environment.GetEnvironmentVariable("WAN_DEBUG_DIR") is { Length: > 0 } d ? d : null;
    private static bool _initialized;
    private static readonly object _lock = new();

    private static void EnsureInit()
    {
        if (_initialized) return;
        lock (_lock)
        {
            if (_initialized) return;
            if (_dumpDir is not null) Directory.CreateDirectory(Path.Combine(_dumpDir, "layers"));
            _initialized = true;
        }
    }

    public static void Dump(string name, Tensor t)
    {
        if (_dumpDir is null) return;
        EnsureInit();
        WriteRawF32(Path.Combine(_dumpDir, "layers", name.Replace('.', '_') + ".bin"), t);
        // Shape sidecar so the Python layer-diff reference knows how to reshape each raw-F32 blob.
        int[] dims = new int[t.Shape.Rank];
        for (int i = 0; i < dims.Length; i++) dims[i] = (int)t.Shape[i];
        lock (_lock) File.AppendAllText(Path.Combine(_dumpDir, "shapes.txt"), $"{name.Replace('.', '_')} {string.Join(",", dims)}\n");
    }

    public static void DumpOutput(Tensor t)
    {
        if (_dumpDir is null) return;
        EnsureInit();
        WriteRawF32(Path.Combine(_dumpDir, "output_velocity.bin"), t);
    }

    private static void WriteRawF32(string path, Tensor t)
    {
        long count = t.Shape.ElementCount;
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        byte[] buffer = new byte[count * sizeof(float)];
        fixed (byte* dst = buffer) Buffer.MemoryCopy((float*)f32.DataPointer, dst, buffer.Length, buffer.Length);
        File.WriteAllBytes(path, buffer);
        if (!ReferenceEquals(f32, t)) f32.Dispose();
    }
}

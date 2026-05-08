using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Debug-dump hook for OmniGen 2's main forward pass. Activates when the
/// <c>OMNIGEN2_DEBUG_DIR</c> environment variable points to a directory; otherwise no-ops.
/// Emits raw F32 binaries one per intermediate tensor for layer-by-layer diff against
/// a Python reference dump.</summary>
public static class OmniGen2DebugDump
{
    private static readonly string? _dir = Environment.GetEnvironmentVariable("OMNIGEN2_DEBUG_DIR");
    private static int _layerSeq;

    /// <summary>Whether the debug dump is currently active.</summary>
    public static bool IsActive => _dir is not null && Directory.Exists(_dir);

    /// <summary>Dumps a tensor under a label-derived filename. No-ops when inactive.</summary>
    public static unsafe void Dump(string label, Tensor tensor)
    {
        if (!IsActive) return;
        Tensor casted = tensor.DType == DType.F32 ? tensor : tensor.CastTo(DType.F32);
        try
        {
            int seq = Interlocked.Increment(ref _layerSeq);
            string fileName = $"{seq:D4}_{label}.bin";
            string path = Path.Combine(_dir!, fileName);

            long byteCount = casted.Shape.ElementCount * sizeof(float);
            byte[] buffer = new byte[byteCount];
            fixed (byte* dest = buffer)
            {
                Buffer.MemoryCopy((void*)casted.DataPointer, dest, byteCount, byteCount);
            }
            File.WriteAllBytes(path, buffer);
        }
        finally
        {
            if (!ReferenceEquals(casted, tensor)) casted.Dispose();
        }
    }
}

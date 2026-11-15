using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Memory.Host;

/// <summary>Records the file a borrowed tensor was mapped from, so its bytes can be read back with positional reads.</summary>
public static class TensorFileExtents
{
    private const int MaxViewDepth = 8;

    private static readonly ConditionalWeakTable<Tensor, FileExtent> Extents = new();

    /// <summary>Records that the first byte of <paramref name="tensor"/> sits at <paramref name="fileOffset"/> in <paramref name="path"/>.</summary>
    public static void Register(Tensor tensor, string path, long fileOffset)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentException.ThrowIfNullOrEmpty(path);
        ArgumentOutOfRangeException.ThrowIfNegative(fileOffset);
        Extents.AddOrUpdate(tensor, new FileExtent(path, fileOffset));
    }

    /// <summary>Finds the file of <paramref name="tensor"/>, or of the registered tensor it views, and the file offset of its first byte.</summary>
    public static unsafe bool TryGet(Tensor tensor, out string path, out long fileOffset)
    {
        Tensor? node = tensor;
        for (int depth = 0; depth < MaxViewDepth && node is not null; depth++)
        {
            if (Extents.TryGetValue(node, out FileExtent? extent))
            {
                // A view starts at a byte offset inside its registered root, so the file offset follows the pointer delta.
                long delta = (long)((nint)tensor.DataPointer - (nint)node.DataPointer);
                path = extent.Path;
                fileOffset = extent.FileOffset + delta;
                return true;
            }
            node = node.KeepAliveOwner as Tensor;
        }
        path = string.Empty;
        fileOffset = 0;
        return false;
    }

    private sealed record FileExtent(string Path, long FileOffset);
}

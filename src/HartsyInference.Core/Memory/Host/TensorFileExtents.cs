using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Memory.Host;

/// <summary>Records the file a borrowed tensor was mapped from, so its bytes can be read back with positional reads.</summary>
public static class TensorFileExtents
{
    private const int MaxViewDepth = 8;

    private static readonly ConditionalWeakTable<Tensor, FileExtent> Extents = new();

    /// <summary>Records that the first byte of <paramref name="tensor"/> sits at <paramref name="fileOffset"/> in the file <paramref name="owner"/> maps.</summary>
    public static void Register(Tensor tensor, MmapHandle owner, long fileOffset)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentNullException.ThrowIfNull(owner);
        ArgumentOutOfRangeException.ThrowIfNegative(fileOffset);
        Extents.AddOrUpdate(tensor, new FileExtent(owner, fileOffset));
    }

    /// <summary>Finds the mapping of <paramref name="tensor"/>, or of the registered tensor it views, and the file offset of its first byte.</summary>
    public static unsafe bool TryGet(Tensor tensor, [MaybeNullWhen(false)] out MmapHandle owner, out long fileOffset)
    {
        Tensor? node = tensor;
        for (int depth = 0; depth < MaxViewDepth && node is not null; depth++)
        {
            if (Extents.TryGetValue(node, out FileExtent? extent))
            {
                // A view starts at a byte offset inside its registered root, so the file offset follows the pointer delta.
                long delta = (long)((nint)tensor.DataPointer - (nint)node.DataPointer);
                owner = extent.Owner;
                fileOffset = extent.FileOffset + delta;
                return true;
            }
            node = node.KeepAliveOwner as Tensor;
        }
        owner = null;
        fileOffset = 0;
        return false;
    }

    private sealed record FileExtent(MmapHandle Owner, long FileOffset);
}

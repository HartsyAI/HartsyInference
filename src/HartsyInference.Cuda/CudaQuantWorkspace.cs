using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.Cuda;

/// <summary>A bounded ring of BF16 dequant slots for recipe weights (MXFP4 or FP8 with E8M0 scales): the fallback that unpacks one expert matrix at a time
/// on the compute stream, bypassing <c>IBackend.Linear</c>'s F16 cast cache, whose per-tensor copies would outgrow the expert budget.</summary>
/// <remarks>Slots are allocated on first use and freed at dispose. Dequant and the GEMM that reads it are queued on the same stream, so a returned slot
/// can be refilled at once. The ring is bounded: renting with every slot out throws rather than growing.</remarks>
public sealed class CudaQuantWorkspace : IDisposable
{
    /// <summary>Bytes of the largest routed-expert matrix in BF16.</summary>
    public const long DefaultSlotBytes = 23_592_960;

    /// <summary>Slots in the ring.</summary>
    public const int DefaultSlots = 2;

    private readonly CudaBackend _backend;
    private readonly ulong[] _slots;
    private readonly bool[] _rented;
    private readonly object _gate = new();
    private int _next;
    private bool _disposed;

    internal CudaQuantWorkspace(CudaBackend backend, long slotBytes = DefaultSlotBytes, int slots = DefaultSlots)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(slotBytes);
        ArgumentOutOfRangeException.ThrowIfLessThan(slots, 1);
        _backend = backend;
        SlotBytes = slotBytes;
        _slots = new ulong[slots];
        _rented = new bool[slots];
    }

    /// <summary>Capacity of one slot in bytes.</summary>
    public long SlotBytes { get; }

    /// <summary>Slots currently leased.</summary>
    public int Rented
    {
        get
        {
            lock (_gate) return _rented.Count(static r => r);
        }
    }

    /// <summary>Unpacks one matrix of a live <paramref name="lease"/>, so the weight and scale are pinned for the duration of the call and the queued kernel.</summary>
    public QuantWorkspaceLease Dequantize(ExpertLease lease, ExpertMatrix matrix)
    {
        ArgumentNullException.ThrowIfNull(lease);
        ArgumentNullException.ThrowIfNull(matrix);
        if (lease.IsReleased) throw new InvalidOperationException("The expert lease was released; its weights may be evicted.");
        bool owned = lease.Weights.Any(w => ReferenceEquals(w.W1, matrix) || ReferenceEquals(w.W2, matrix) || ReferenceEquals(w.W3, matrix));
        if (!owned) throw new ArgumentException("The matrix is not part of the lease.", nameof(matrix));
        return Dequantize(matrix);
    }

    /// <summary>Unpacks <paramref name="matrix"/> to BF16 into the next free slot. The caller must keep the owning expert pinned; prefer the lease overload.</summary>
    public QuantWorkspaceLease Dequantize(ExpertMatrix matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        QuantRecipe recipe = matrix.Recipe ?? throw new NotSupportedException("The expert matrix carries no quant recipe; it is already a dense weight.");
        Tensor scale = recipe.Scale ?? throw new InvalidOperationException("The recipe has no scale tensor.");
        Validate(recipe, matrix.Weight, scale);
        if (!_backend.TryGetResidentPointer(matrix.Weight, out ulong packed) || !_backend.TryGetResidentPointer(scale, out ulong scalePtr))
            throw new InvalidOperationException("Weight or scale is not device resident; acquire the expert from the cache first.");

        long bytes = recipe.LogicalRows * recipe.LogicalCols * 2;
        if (bytes > SlotBytes)
            throw new NotSupportedException($"A {recipe.LogicalRows}x{recipe.LogicalCols} BF16 matrix is {bytes} bytes; a workspace slot holds {SlotBytes}.");
        int slot;
        ulong buffer;
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            slot = -1;
            for (int i = 0; i < _slots.Length; i++)
            {
                int candidate = (_next + i) % _slots.Length;
                if (_rented[candidate]) continue;
                slot = candidate;
                break;
            }
            if (slot < 0) throw new InvalidOperationException($"All {_slots.Length} dequant workspace slots are leased.");
            if (_slots[slot] == 0) _slots[slot] = _backend.AllocateWorkspace((nuint)SlotBytes);
            _rented[slot] = true;
            _next = (slot + 1) % _slots.Length;
            buffer = _slots[slot];
        }
        try
        {
            _backend.LaunchRecipeDequant(recipe, packed, scalePtr, buffer);
        }
        catch
        {
            Return(slot);
            throw;
        }
        return new QuantWorkspaceLease(this, slot, buffer, recipe.LogicalRows, recipe.LogicalCols);
    }

    internal void Return(int slot)
    {
        lock (_gate) _rented[slot] = false;
    }

    /// <summary>Frees every slot after the compute stream has finished with them.</summary>
    public void Dispose()
    {
        ulong[] toFree;
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            toFree = [.. _slots];
            Array.Clear(_slots);
        }
        _backend.FreeWorkspace(toFree);
    }

    private static void Validate(QuantRecipe recipe, Tensor packed, Tensor scale)
    {
        if (recipe.Encoding is not (QuantEncoding.Mxfp4E8M0 or QuantEncoding.Fp8E4M3BlockE8M0))
            throw new NotSupportedException($"Device dequant covers Mxfp4E8M0 and Fp8E4M3BlockE8M0; recipe is {recipe.Encoding}.");
        if (recipe.ScaleLayout != ScaleLayout.RowMajorBlocks)
            throw new NotSupportedException($"Device dequant reads {ScaleLayout.RowMajorBlocks} scales; recipe is {recipe.ScaleLayout}.");
        if (recipe.ScaleDType != DType.F8E8M0 && recipe.ScaleDType != DType.U8)
            throw new NotSupportedException($"Device dequant needs E8M0 scale bytes; recipe has {recipe.ScaleDType}.");
        if (scale.Shape.Rank != 2) throw new ArgumentException($"Scale must be rank 2; got {scale.Shape}.");
        (long scaleRows, long scaleCols) = recipe.Geometry.ScaleShape(recipe.LogicalRows, recipe.LogicalCols);
        if (scale.Shape[0] < scaleRows || scale.Shape[1] < recipe.ScaleColOffset + scaleCols)
            throw new ArgumentException($"Scale {scale.Shape} is smaller than the [{scaleRows}, {recipe.ScaleColOffset + scaleCols}] the {recipe.Geometry} geometry needs.");
        long expectedPacked = recipe.LogicalRows * recipe.LogicalCols / recipe.ElementsPerByte;
        long actualPacked = packed.DType.ComputeByteCount(packed.ElementCount);
        if (actualPacked != expectedPacked)
            throw new ArgumentException($"Packed weight is {actualPacked} bytes; the recipe describes {expectedPacked}.");
        if (recipe.Encoding == QuantEncoding.Mxfp4E8M0 && (recipe.Geometry.BlockCols % 2 != 0 || recipe.LogicalCols % 2 != 0))
            throw new NotSupportedException($"Mxfp4 needs even block width and columns; got {recipe.Geometry} over {recipe.LogicalCols} columns.");
    }
}

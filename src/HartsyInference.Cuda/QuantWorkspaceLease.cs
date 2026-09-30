namespace HartsyInference.Cuda;

/// <summary>One dequantized BF16 matrix in a <see cref="CudaQuantWorkspace"/> slot; valid on the compute stream until disposed.</summary>
public sealed class QuantWorkspaceLease : IDisposable
{
    private readonly CudaQuantWorkspace _owner;
    private readonly int _slot;
    private int _returned;

    internal QuantWorkspaceLease(CudaQuantWorkspace owner, int slot, ulong devicePointer, long rows, long cols)
    {
        _owner = owner;
        _slot = slot;
        DevicePointer = devicePointer;
        Rows = rows;
        Cols = cols;
    }

    /// <summary>Device address of the row-major BF16 <c>[Rows, Cols]</c> matrix.</summary>
    public ulong DevicePointer { get; }

    /// <summary>Output features.</summary>
    public long Rows { get; }

    /// <summary>Input features.</summary>
    public long Cols { get; }

    /// <summary>Bytes of the BF16 matrix.</summary>
    public long Bytes => Rows * Cols * 2;

    /// <summary>Returns the slot; work queued on the compute stream before this call still reads it safely, because a later dequant into the slot is queued behind that work.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _returned, 1) == 0) _owner.Return(_slot);
    }
}

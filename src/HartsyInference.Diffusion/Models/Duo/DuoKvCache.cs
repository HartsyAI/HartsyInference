namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>One layer's key and value rows of the committed history, oldest first, in a flat row-major <c>[rows, width]</c> buffer.
/// The caller appends each committed chunk and evicts the oldest rows when a sliding history window is full; a row is never rewritten in place.
/// <b>CPU reference only.</b> It is managed <c>float[]</c> storage, not backend tensors, and nothing in the production path uses it yet.
/// <see cref="Denoisers.DiTBlocks.FrameKvCache"/> is not reused because it stores one slot per latent frame as pinned backend tensors whose dtype
/// and shape the attention path fixes. Duo needs row-granular sliding eviction over the packed sequence, which a frame slot cannot express.
/// A backend-backed row cache is a later step; whether <c>FrameKvCache</c> can host it is open.
/// Rows are stored verbatim. The intended contract is that keys are post-QK-norm and post-RoPE, as the attention layer reads them, and values
/// as produced by the V projection. This class does not enforce that, and the order has not been checked against upstream.
/// Because keys are post-RoPE, each row's rotary position is fixed when it is committed. Eviction never re-rotates the remaining rows, so their
/// positions do not restart at 0. A caller that stored pre-RoPE keys would have to re-rotate after eviction, which this contract rules out.</summary>
public sealed class DuoKvCache
{
    private readonly float[] _keys;
    private readonly float[] _values;

    public DuoKvCache(int capacityRows, int width)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacityRows, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        CapacityRows = capacityRows;
        Width = width;
        _keys = new float[checked(capacityRows * width)];
        _values = new float[checked(capacityRows * width)];
    }

    public int CapacityRows { get; }

    /// <summary>Floats per row: the key and value width, <c>kvHeads * headDim</c>.</summary>
    public int Width { get; }

    /// <summary>Rows currently held.</summary>
    public int Rows { get; private set; }

    /// <summary>The held keys, oldest row first.</summary>
    public ReadOnlySpan<float> Keys => _keys.AsSpan(0, Rows * Width);

    /// <summary>The held values, oldest row first.</summary>
    public ReadOnlySpan<float> Values => _values.AsSpan(0, Rows * Width);

    /// <summary>Appends whole rows after the newest one.</summary>
    /// <exception cref="ArgumentException">The spans differ in length or are not whole rows.</exception>
    /// <exception cref="InvalidOperationException">The rows would exceed the capacity; evict first.</exception>
    public void Append(ReadOnlySpan<float> keys, ReadOnlySpan<float> values)
    {
        if (keys.Length != values.Length || keys.Length % Width != 0)
        {
            throw new ArgumentException($"Append takes equal key and value spans of whole {Width}-float rows; got {keys.Length} and {values.Length}.");
        }
        int rows = keys.Length / Width;
        if (Rows + rows > CapacityRows)
        {
            throw new InvalidOperationException(
                $"Appending {rows} rows to {Rows} would exceed the {CapacityRows}-row capacity; evict the oldest rows first.");
        }
        keys.CopyTo(_keys.AsSpan(Rows * Width));
        values.CopyTo(_values.AsSpan(Rows * Width));
        Rows += rows;
    }

    /// <summary>Drops <paramref name="count"/> rows starting at row <paramref name="first"/>, shifting the later rows down to close the gap.
    /// Rows before <paramref name="first"/> keep their place, so a reference or shared span can stay while a clean block goes.</summary>
    public void EvictRange(int first, int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(first);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if (first + count > Rows)
        {
            throw new ArgumentOutOfRangeException(nameof(count), count, $"Rows [{first}, {first + count}) are not all held; {Rows} rows are held.");
        }
        int tail = Rows - (first + count);
        Array.Copy(_keys, (first + count) * Width, _keys, first * Width, tail * Width);
        Array.Copy(_values, (first + count) * Width, _values, first * Width, tail * Width);
        Rows -= count;
    }

    /// <summary>Forgets every row; the buffers are reused.</summary>
    public void Clear() => Rows = 0;
}

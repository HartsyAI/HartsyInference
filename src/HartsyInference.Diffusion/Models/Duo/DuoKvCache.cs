namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>One layer's key and value rows of the committed history, oldest first, in a flat row-major <c>[rows, width]</c> buffer.
/// The caller appends each committed chunk and evicts the oldest rows when a sliding history window is full; a row is never rewritten in place.</summary>
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

    /// <summary>Drops the <paramref name="rows"/> oldest rows, shifting the rest to the front.</summary>
    public void EvictOldest(int rows)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        if (rows > Rows)
        {
            throw new ArgumentOutOfRangeException(nameof(rows), rows, $"Only {Rows} rows are held.");
        }
        int keep = Rows - rows;
        Array.Copy(_keys, rows * Width, _keys, 0, keep * Width);
        Array.Copy(_values, rows * Width, _values, 0, keep * Width);
        Rows = keep;
    }

    /// <summary>Forgets every row; the buffers are reused.</summary>
    public void Clear() => Rows = 0;
}

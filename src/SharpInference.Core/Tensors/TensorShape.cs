namespace SharpInference.Core.Tensors;

/// <summary>Describes the shape (dimensions) and strides of a tensor, up to 6 dimensions. Row-major strides.</summary>
public readonly struct TensorShape : IEquatable<TensorShape>
{
    /// <summary>Maximum number of dimensions supported.</summary>
    public const int MaxRank = 6;

    private readonly long _dim0, _dim1, _dim2, _dim3, _dim4, _dim5;
    private readonly long _stride0, _stride1, _stride2, _stride3, _stride4, _stride5;

    /// <summary>Number of dimensions in this shape.</summary>
    public int Rank { get; }

    /// <summary>Total number of elements across all dimensions.</summary>
    public long ElementCount { get; }

    /// <summary>Creates a tensor shape from the given dimensions. Strides are computed as row-major.</summary>
    public TensorShape(ReadOnlySpan<long> dims)
    {
        if (dims.Length > MaxRank)
            throw new ArgumentException($"TensorShape supports at most {MaxRank} dimensions, got {dims.Length}.");

        Rank = dims.Length;

        // Store dimensions
        _dim0 = dims.Length > 0 ? dims[0] : 0;
        _dim1 = dims.Length > 1 ? dims[1] : 0;
        _dim2 = dims.Length > 2 ? dims[2] : 0;
        _dim3 = dims.Length > 3 ? dims[3] : 0;
        _dim4 = dims.Length > 4 ? dims[4] : 0;
        _dim5 = dims.Length > 5 ? dims[5] : 0;

        // Compute row-major strides and total element count
        ElementCount = 1;
        Span<long> strides = stackalloc long[MaxRank];
        for (int i = Rank - 1; i >= 0; i--)
        {
            strides[i] = ElementCount;
            ElementCount *= dims[i];
        }

        _stride0 = Rank > 0 ? strides[0] : 0;
        _stride1 = Rank > 1 ? strides[1] : 0;
        _stride2 = Rank > 2 ? strides[2] : 0;
        _stride3 = Rank > 3 ? strides[3] : 0;
        _stride4 = Rank > 4 ? strides[4] : 0;
        _stride5 = Rank > 5 ? strides[5] : 0;
    }

    /// <summary>Convenience constructor for 1D shape.</summary>
    public TensorShape(long d0) : this([d0]) { }

    /// <summary>Convenience constructor for 2D shape.</summary>
    public TensorShape(long d0, long d1) : this([d0, d1]) { }

    /// <summary>Convenience constructor for 3D shape.</summary>
    public TensorShape(long d0, long d1, long d2) : this([d0, d1, d2]) { }

    /// <summary>Convenience constructor for 4D shape (NCHW).</summary>
    public TensorShape(long d0, long d1, long d2, long d3) : this([d0, d1, d2, d3]) { }

    /// <summary>Gets the size of the given dimension.</summary>
    public long this[int index] => index switch
    {
        0 => _dim0,
        1 => _dim1,
        2 => _dim2,
        3 => _dim3,
        4 => _dim4,
        5 => _dim5,
        _ => throw new IndexOutOfRangeException($"Dimension index {index} out of range for rank {Rank}."),
    };

    /// <summary>Gets the stride for the given dimension.</summary>
    public long Stride(int index) => index switch
    {
        0 => _stride0,
        1 => _stride1,
        2 => _stride2,
        3 => _stride3,
        4 => _stride4,
        5 => _stride5,
        _ => throw new IndexOutOfRangeException($"Stride index {index} out of range for rank {Rank}."),
    };

    /// <summary>Copies dimension sizes into the provided span.</summary>
    public void CopyDimsTo(Span<long> destination)
    {
        if (destination.Length < Rank)
            throw new ArgumentException($"Destination span length {destination.Length} is less than rank {Rank}.", nameof(destination));

        for (int i = 0; i < Rank; i++)
            destination[i] = this[i];
    }

    /// <summary>Copies strides into the provided span.</summary>
    public void CopyStridesTo(Span<long> destination)
    {
        if (destination.Length < Rank)
            throw new ArgumentException($"Destination span length {destination.Length} is less than rank {Rank}.", nameof(destination));

        for (int i = 0; i < Rank; i++)
            destination[i] = Stride(i);
    }

    /// <summary>Returns a new shape with a dimension of size 1 inserted at the given position.</summary>
    public TensorShape Unsqueeze(int dim)
    {
        if (dim < 0 || dim > Rank)
            throw new ArgumentOutOfRangeException(nameof(dim), $"Unsqueeze dim {dim} out of range for rank {Rank}.");

        if (Rank >= MaxRank)
            throw new InvalidOperationException($"Cannot unsqueeze: already at max rank {MaxRank}.");

        Span<long> newDims = stackalloc long[Rank + 1];
        for (int i = 0; i < dim; i++)
            newDims[i] = this[i];
        newDims[dim] = 1;
        for (int i = dim; i < Rank; i++)
            newDims[i + 1] = this[i];

        return new TensorShape(newDims);
    }

    /// <summary>Returns a new shape with a dimension of size 1 removed at the given position.</summary>
    public TensorShape Squeeze(int dim)
    {
        if (dim < 0 || dim >= Rank)
            throw new ArgumentOutOfRangeException(nameof(dim), $"Squeeze dim {dim} out of range for rank {Rank}.");

        if (this[dim] != 1)
            throw new InvalidOperationException($"Cannot squeeze dimension {dim} with size {this[dim]} (must be 1).");

        Span<long> newDims = stackalloc long[Rank - 1];
        int j = 0;
        for (int i = 0; i < Rank; i++)
        {
            if (i != dim)
                newDims[j++] = this[i];
        }

        return new TensorShape(newDims);
    }

    /// <summary>Returns a new shape with dimensions permuted according to the given order.</summary>
    public TensorShape Permute(ReadOnlySpan<int> order)
    {
        if (order.Length != Rank)
            throw new ArgumentException($"Permutation length {order.Length} must match rank {Rank}.", nameof(order));

        Span<long> newDims = stackalloc long[Rank];
        for (int i = 0; i < Rank; i++)
        {
            if (order[i] < 0 || order[i] >= Rank)
                throw new ArgumentOutOfRangeException(nameof(order), $"Permutation index {order[i]} out of range for rank {Rank}.");
            newDims[i] = this[order[i]];
        }

        return new TensorShape(newDims);
    }

    public bool Equals(TensorShape other) =>
        Rank == other.Rank &&
        _dim0 == other._dim0 && _dim1 == other._dim1 && _dim2 == other._dim2 &&
        _dim3 == other._dim3 && _dim4 == other._dim4 && _dim5 == other._dim5;

    public override bool Equals(object? obj) => obj is TensorShape other && Equals(other);

    public override int GetHashCode() => HashCode.Combine(Rank, _dim0, _dim1, _dim2, _dim3);

    public static bool operator ==(TensorShape left, TensorShape right) => left.Equals(right);

    public static bool operator !=(TensorShape left, TensorShape right) => !left.Equals(right);

    public override string ToString()
    {
        Span<long> dims = stackalloc long[Rank];
        for (int i = 0; i < Rank; i++) dims[i] = this[i];
        return $"[{string.Join(", ", dims.ToArray())}]";
    }
}

using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Layout;

/// <summary>A rectangular window onto a tensor's storage: up to six extents, a non-negative element stride per axis
/// (0 broadcasts the axis) and an element offset. Views compose like NumPy's <c>as_strided</c> without touching
/// data: reshape a contiguous view, select or slice an axis, permute, broadcast. A <see cref="StridedCopySpec"/>
/// pairs a source view with a destination view for <see cref="IBackend.CopyStrided"/>.</summary>
public readonly struct StridedView
{
    /// <summary>The most axes a view holds. More than <see cref="TensorShape.MaxRank"/>, because a pixel shuffle splits a
    /// five-axis tensor's channel and spatial axes into eight.</summary>
    public const int MaxRank = 8;

    private readonly StridedExtents _dims;
    private readonly StridedExtents _strides;

    /// <summary>A view over <paramref name="dims"/> with explicit strides and offset, all in elements.</summary>
    public StridedView(ReadOnlySpan<long> dims, ReadOnlySpan<long> strides, long offset = 0)
    {
        if (dims.Length < 1 || dims.Length > MaxRank)
            throw new ArgumentException($"A strided view has 1 to {MaxRank} axes; got {dims.Length}.", nameof(dims));
        if (strides.Length != dims.Length)
            throw new ArgumentException($"A strided view needs one stride per axis; got {strides.Length} strides for {dims.Length} axes.", nameof(strides));
        if (offset < 0)
            throw new ArgumentOutOfRangeException(nameof(offset), offset, "A strided view's offset must not be negative.");
        for (int axis = 0; axis < dims.Length; axis++)
        {
            if (dims[axis] <= 0)
                throw new ArgumentOutOfRangeException(nameof(dims), dims[axis], $"Axis {axis} of a strided view must have a positive extent.");
            if (strides[axis] < 0)
                throw new ArgumentOutOfRangeException(nameof(strides), strides[axis], $"Axis {axis} of a strided view must have a non-negative stride.");
        }
        Rank = dims.Length;
        Offset = offset;
        dims.CopyTo(_dims);
        strides.CopyTo(_strides);
    }

    /// <summary>Number of axes.</summary>
    public int Rank { get; }

    /// <summary>Element offset of the view's first element.</summary>
    public long Offset { get; }

    /// <summary>Extent of one axis, outermost first.</summary>
    public long this[int axis]
    {
        get
        {
            CheckAxis(axis);
            return _dims[axis];
        }
    }

    /// <summary>Element stride of one axis; 0 broadcasts the axis.</summary>
    public long Stride(int axis)
    {
        CheckAxis(axis);
        return _strides[axis];
    }

    /// <summary>Writes the extents into <paramref name="destination"/>, which must hold <see cref="Rank"/> entries.</summary>
    public void CopyDimsTo(Span<long> destination) => ((ReadOnlySpan<long>)_dims)[..Rank].CopyTo(destination);

    /// <summary>Writes the strides into <paramref name="destination"/>, which must hold <see cref="Rank"/> entries.</summary>
    public void CopyStridesTo(Span<long> destination) => ((ReadOnlySpan<long>)_strides)[..Rank].CopyTo(destination);

    /// <summary>Product of the extents.</summary>
    public long ElementCount
    {
        get
        {
            long count = 1;
            for (int axis = 0; axis < Rank; axis++) count = checked(count * _dims[axis]);
            return count;
        }
    }

    /// <summary>The largest element index the view addresses: offset plus every axis's last step.</summary>
    public long MaxElementIndex
    {
        get
        {
            long index = Offset;
            for (int axis = 0; axis < Rank; axis++) index = checked(index + (_dims[axis] - 1) * _strides[axis]);
            return index;
        }
    }

    /// <summary>True when the view walks its storage in row-major order with unit inner stride (axes of extent 1 excepted).</summary>
    public bool IsContiguous
    {
        get
        {
            long expected = 1;
            for (int axis = Rank - 1; axis >= 0; axis--)
            {
                if (_dims[axis] == 1) continue;
                if (_strides[axis] != expected) return false;
                expected *= _dims[axis];
            }
            return true;
        }
    }

    /// <summary>True when both views have the same extents on every axis.</summary>
    public bool HasSameDims(in StridedView other)
    {
        if (other.Rank != Rank) return false;
        for (int axis = 0; axis < Rank; axis++)
        {
            if (other._dims[axis] != _dims[axis]) return false;
        }
        return true;
    }

    /// <summary>The whole of a tensor of <paramref name="shape"/>, row-major.</summary>
    public static StridedView Of(TensorShape shape)
    {
        Span<long> dims = stackalloc long[shape.Rank];
        shape.CopyDimsTo(dims);
        return Of(dims);
    }

    /// <summary>A row-major view over <paramref name="dims"/> starting at element 0.</summary>
    public static StridedView Of(ReadOnlySpan<long> dims)
    {
        Span<long> strides = stackalloc long[dims.Length];
        long stride = 1;
        for (int axis = dims.Length - 1; axis >= 0; axis--)
        {
            strides[axis] = stride;
            stride = checked(stride * Math.Max(dims[axis], 1));
        }
        return new StridedView(dims, strides);
    }

    /// <summary>A vector of length <c>dims[axis]</c> broadcast over every other axis of <paramref name="dims"/>: stride 1 along <paramref name="axis"/>, 0 elsewhere.</summary>
    public static StridedView VectorAlong(ReadOnlySpan<long> dims, int axis)
    {
        if ((uint)axis >= (uint)dims.Length)
            throw new ArgumentOutOfRangeException(nameof(axis), axis, $"Axis must be in [0,{dims.Length}).");
        Span<long> strides = stackalloc long[dims.Length];
        strides[axis] = 1;
        return new StridedView(dims, strides);
    }

    /// <summary>A row-major view with this view's extents, starting at element 0.</summary>
    public StridedView Packed()
    {
        Span<long> dims = stackalloc long[Rank];
        CopyDimsTo(dims);
        return Of(dims);
    }

    /// <summary>The same elements under new extents; the view must be contiguous and the element count unchanged.</summary>
    public StridedView Reshape(ReadOnlySpan<long> dims)
    {
        if (!IsContiguous)
            throw new InvalidOperationException($"Only a contiguous view can be reshaped; {this} is not.");
        StridedView packed = Of(dims);
        if (packed.ElementCount != ElementCount)
            throw new ArgumentException($"Cannot reshape {ElementCount} elements to {packed.ElementCount}.", nameof(dims));
        Span<long> strides = stackalloc long[packed.Rank];
        packed.CopyStridesTo(strides);
        return new StridedView(dims, strides, Offset);
    }

    /// <summary>Splits <paramref name="axis"/> of extent <c>outer·inner</c> into two axes, the outer one stepping by
    /// <paramref name="inner"/> strides; valid on any view, contiguous or not.</summary>
    public StridedView Split(int axis, long outer, long inner)
    {
        CheckAxis(axis);
        if (outer <= 0 || inner <= 0 || checked(outer * inner) != _dims[axis])
            throw new ArgumentException($"Axis {axis} of extent {_dims[axis]} does not split into {outer} × {inner}.", nameof(outer));
        if (Rank == MaxRank)
            throw new InvalidOperationException($"A strided view holds at most {MaxRank} axes.");
        Span<long> dims = stackalloc long[Rank + 1];
        Span<long> strides = stackalloc long[Rank + 1];
        int k = 0;
        for (int a = 0; a < Rank; a++)
        {
            if (a == axis)
            {
                dims[k] = outer;
                strides[k] = checked(_strides[a] * inner);
                k++;
                dims[k] = inner;
                strides[k] = _strides[a];
                k++;
                continue;
            }
            dims[k] = _dims[a];
            strides[k] = _strides[a];
            k++;
        }
        return new StridedView(dims, strides, Offset);
    }

    /// <summary>Fixes <paramref name="axis"/> at <paramref name="index"/> and drops it.</summary>
    public StridedView Select(int axis, long index)
    {
        CheckAxis(axis);
        if (Rank == 1)
            throw new InvalidOperationException("Selecting the only axis would leave an empty view; use Slice with length 1.");
        if ((ulong)index >= (ulong)_dims[axis])
            throw new ArgumentOutOfRangeException(nameof(index), index, $"Index must be in [0,{_dims[axis]}) on axis {axis}.");
        Span<long> dims = stackalloc long[Rank - 1];
        Span<long> strides = stackalloc long[Rank - 1];
        int k = 0;
        for (int a = 0; a < Rank; a++)
        {
            if (a == axis) continue;
            dims[k] = _dims[a];
            strides[k] = _strides[a];
            k++;
        }
        return new StridedView(dims, strides, checked(Offset + index * _strides[axis]));
    }

    /// <summary>Keeps <paramref name="length"/> positions of <paramref name="axis"/> from <paramref name="start"/>, every <paramref name="step"/>-th one.</summary>
    public StridedView Slice(int axis, long start, long length, long step = 1)
    {
        CheckAxis(axis);
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length), length, "A slice keeps at least one position.");
        if (step <= 0)
            throw new ArgumentOutOfRangeException(nameof(step), step, "A slice step must be positive.");
        if (start < 0 || checked(start + (length - 1) * step) >= _dims[axis])
            throw new ArgumentOutOfRangeException(nameof(start), start, $"Slice [{start}, +{length} by {step}) leaves axis {axis} of extent {_dims[axis]}.");
        Span<long> dims = stackalloc long[Rank];
        Span<long> strides = stackalloc long[Rank];
        CopyDimsTo(dims);
        CopyStridesTo(strides);
        dims[axis] = length;
        strides[axis] = checked(strides[axis] * step);
        return new StridedView(dims, strides, checked(Offset + start * _strides[axis]));
    }

    /// <summary>Reorders the axes: result axis <c>k</c> is this view's axis <c>order[k]</c>.</summary>
    public StridedView Permute(ReadOnlySpan<int> order)
    {
        if (order.Length != Rank)
            throw new ArgumentException($"A permutation of a rank-{Rank} view needs {Rank} entries; got {order.Length}.", nameof(order));
        Span<bool> seen = stackalloc bool[Rank];
        Span<long> dims = stackalloc long[Rank];
        Span<long> strides = stackalloc long[Rank];
        for (int k = 0; k < Rank; k++)
        {
            int source = order[k];
            if ((uint)source >= (uint)Rank || seen[source])
                throw new ArgumentException($"Permutation entry {k} is {source}; each axis in [0,{Rank}) must appear once.", nameof(order));
            seen[source] = true;
            dims[k] = _dims[source];
            strides[k] = _strides[source];
        }
        return new StridedView(dims, strides, Offset);
    }

    /// <summary>Inserts an axis of extent <paramref name="count"/> and stride 0 at position <paramref name="axis"/> (0 to <see cref="Rank"/>), so every position reads the same elements.</summary>
    public StridedView Broadcast(int axis, long count)
    {
        if ((uint)axis > (uint)Rank)
            throw new ArgumentOutOfRangeException(nameof(axis), axis, $"A new axis goes at a position in [0,{Rank}].");
        if (Rank == MaxRank)
            throw new InvalidOperationException($"A strided view holds at most {MaxRank} axes.");
        Span<long> dims = stackalloc long[Rank + 1];
        Span<long> strides = stackalloc long[Rank + 1];
        int k = 0;
        for (int a = 0; a <= Rank; a++)
        {
            if (a == axis)
            {
                dims[a] = count;
                strides[a] = 0;
                continue;
            }
            dims[a] = _dims[k];
            strides[a] = _strides[k];
            k++;
        }
        return new StridedView(dims, strides, Offset);
    }

    /// <summary>Inserts an axis of extent 1 at position <paramref name="axis"/>.</summary>
    public StridedView Unsqueeze(int axis) => Broadcast(axis, 1);

    public override string ToString()
    {
        long[] dims = new long[Rank];
        long[] strides = new long[Rank];
        CopyDimsTo(dims);
        CopyStridesTo(strides);
        return $"dims [{string.Join(",", dims)}] strides [{string.Join(",", strides)}] offset {Offset}";
    }

    private void CheckAxis(int axis)
    {
        if ((uint)axis >= (uint)Rank)
            throw new ArgumentOutOfRangeException(nameof(axis), axis, $"Axis must be in [0,{Rank}).");
    }
}

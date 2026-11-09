using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Copy, staging and dtype helpers shared by the SANA-WM layers.</summary>
/// <remarks>Activations are plain tensors that a device backend keeps resident by tensor identity. A host view of one
/// (<see cref="Tensor.Reshape"/>, <see cref="Tensor.SliceRows"/>) reads it back, and an op that writes into a view leaves the
/// parent unchanged, so the layers allocate every activation in the shape its consumer reads and move row ranges with
/// <see cref="IBackend.SliceRowsGeneric"/> and <see cref="IBackend.ScatterRowsGeneric"/>. Views are taken of host weights only.</remarks>
internal static class SanaWmTensors
{
    /// <summary>A new tensor of <paramref name="shape"/> holding <paramref name="source"/>'s elements from
    /// <paramref name="elementOffset"/>; a device copy on device backends. The caller owns the result.</summary>
    internal static Tensor Slice(IBackend backend, Tensor source, long elementOffset, TensorShape shape)
    {
        if (elementOffset < 0 || elementOffset + shape.ElementCount > source.ElementCount)
            throw new ArgumentOutOfRangeException(nameof(elementOffset), $"{shape} at {elementOffset} is outside {source.Shape}.");
        Tensor slice = new(shape, source.DType);
        try
        {
            backend.SliceRowsGeneric(slice, source, RowOffset(elementOffset, shape[shape.Rank - 1]));
            return slice;
        }
        catch
        {
            slice.Dispose();
            throw;
        }
    }

    /// <summary>A copy of <paramref name="source"/> in <paramref name="shape"/>, which must hold the same element count.</summary>
    internal static Tensor Copy(IBackend backend, Tensor source, TensorShape shape)
    {
        if (shape.ElementCount != source.ElementCount)
            throw new ArgumentException($"A copy of {source.Shape} cannot have shape {shape}.", nameof(shape));
        return Slice(backend, source, 0, shape);
    }

    /// <summary>Writes <paramref name="source"/> into <paramref name="destination"/> from element <paramref name="elementOffset"/>. The
    /// first write allocates the destination's device buffer without uploading its host bytes.</summary>
    internal static void Write(IBackend backend, Tensor destination, Tensor source, long elementOffset)
    {
        if (elementOffset < 0 || elementOffset + source.ElementCount > destination.ElementCount)
            throw new ArgumentOutOfRangeException(nameof(elementOffset), $"{source.Shape} at {elementOffset} does not fit {destination.Shape}.");
        backend.ScatterRowsGeneric(destination, source, RowOffset(elementOffset, source.Shape[source.Shape.Rank - 1]));
    }

    /// <summary>Makes a host-built tensor an activation of <paramref name="backend"/>, so every later reader shares one device
    /// copy. Takes ownership of <paramref name="host"/>; on the CPU it is returned as it is.</summary>
    internal static Tensor Stage(IBackend backend, Tensor host)
    {
        if (backend.Device.IsCpu)
            return host;
        try
        {
            return Copy(backend, host, host.Shape);
        }
        finally
        {
            host.Dispose();
        }
    }

    /// <summary>A new host F32 tensor with every element set to <paramref name="value"/>.</summary>
    internal static Tensor Filled(TensorShape shape, float value)
    {
        Tensor tensor = new(shape, DType.F32);
        tensor.AsSpan<float>().Fill(value);
        return tensor;
    }

    /// <summary>Returns <paramref name="source"/> when it is F32, otherwise a host F32 copy recorded in <paramref name="owned"/>. For
    /// small parameters only: a converted weight matrix would be a second, F32-sized copy of the model.</summary>
    internal static Tensor HostF32(Tensor source, List<Tensor> owned)
    {
        if (source.DType == DType.F32)
            return source;
        Tensor converted = source.CastTo(DType.F32);
        owned.Add(converted);
        return converted;
    }

    /// <summary>A new F32 copy of an activation. The CPU takes the dtype-aware host cast, because the IBackend default
    /// <see cref="IBackend.CastToF32"/> reads its input as F16; device backends cast on the device.</summary>
    internal static Tensor ToF32(IBackend backend, Tensor source)
    {
        if (backend.Device.IsCpu)
            return source.CastTo(DType.F32);
        Tensor converted = new(source.Shape, DType.F32);
        try
        {
            backend.CastToF32(converted, source);
            return converted;
        }
        catch
        {
            converted.Dispose();
            throw;
        }
    }

    /// <summary>A new BF16 copy of an F32 activation, cast on the backend.</summary>
    internal static Tensor ToBf16(IBackend backend, Tensor source)
    {
        Tensor converted = new(source.Shape, DType.BF16);
        try
        {
            backend.CastToBf16(converted, source);
            return converted;
        }
        catch
        {
            converted.Dispose();
            throw;
        }
    }

    /// <summary>Throws unless <paramref name="tensor"/> has the given dtype and exact dimensions.</summary>
    internal static void Require(Tensor tensor, string name, DType dtype, ReadOnlySpan<long> dims)
    {
        if (tensor.DType != dtype || tensor.Shape.Rank != dims.Length)
            throw new ArgumentException($"{name} must be {dtype.Name} with rank {dims.Length}; got {tensor.DType} {tensor.Shape}.", name);
        for (int i = 0; i < dims.Length; i++)
        {
            if (tensor.Shape[i] != dims[i])
                throw new ArgumentException($"{name} axis {i} must be {dims[i]}; got {tensor.Shape}.", name);
        }
    }

    /// <summary>Throws unless <paramref name="tensor"/> is an F32 or BF16 matrix <c>[rows, columns]</c>.</summary>
    internal static void RequireMatrix(Tensor tensor, string name, long rows, long columns)
    {
        if ((tensor.DType != DType.F32 && tensor.DType != DType.BF16) || tensor.Shape.Rank != 2 || tensor.Shape[0] != rows
            || tensor.Shape[1] != columns)
            throw new ArgumentException($"{name} must be F32 or BF16 [{rows}, {columns}]; got {tensor.DType} {tensor.Shape}.", name);
    }

    /// <summary>Disposes every tensor in <paramref name="tensors"/> and clears the list.</summary>
    internal static void DisposeAll(List<Tensor> tensors)
    {
        foreach (Tensor tensor in tensors)
            tensor.Dispose();
        tensors.Clear();
    }

    private static int RowOffset(long elementOffset, long rowWidth)
    {
        if (rowWidth <= 0 || elementOffset % rowWidth != 0)
            throw new ArgumentException($"Element offset {elementOffset} is not a whole number of {rowWidth}-element rows.", nameof(elementOffset));
        return checked((int)(elementOffset / rowWidth));
    }
}

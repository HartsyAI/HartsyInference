using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Dtype and placement helpers shared by the SANA-WM block and DiT.</summary>
internal static class SanaWmTensors
{
    /// <summary>Returns <paramref name="source"/> itself when it is F32, otherwise an F32 copy that the caller records in
    /// <paramref name="owned"/> and disposes.</summary>
    internal static Tensor ToF32(IBackend backend, Tensor source, List<Tensor> owned)
    {
        if (source.DType == DType.F32)
            return source;

        Tensor converted = backend.Device.IsCpu ? source.CastTo(DType.F32) : ConvertOnDevice(backend, source);
        owned.Add(converted);
        return converted;
    }

    /// <summary>Copies a host F32 tensor to the backend device. The caller owns the result.</summary>
    internal static Tensor Upload(IBackend backend, Tensor host)
    {
        Tensor device = new(host.Shape, DType.F32, backend.Device);
        bool built = false;
        try
        {
            backend.CopyTo(device, host);
            built = true;
            return device;
        }
        finally
        {
            if (!built)
                device.Dispose();
        }
    }

    /// <summary>Copies a device F32 tensor to a new host F32 tensor. The caller owns the result.</summary>
    internal static Tensor Readback(IBackend backend, Tensor device)
    {
        Tensor host = new(device.Shape, DType.F32, DeviceKind.Cpu);
        bool built = false;
        try
        {
            backend.CopyTo(host, device);
            built = true;
            return host;
        }
        finally
        {
            if (!built)
                host.Dispose();
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

    /// <summary>Disposes every tensor in <paramref name="tensors"/> and clears the list.</summary>
    internal static void DisposeAll(List<Tensor> tensors)
    {
        foreach (Tensor tensor in tensors)
            tensor.Dispose();
        tensors.Clear();
    }

    private static Tensor ConvertOnDevice(IBackend backend, Tensor source)
    {
        Tensor converted = new(source.Shape, DType.F32, backend.Device);
        bool built = false;
        try
        {
            backend.CastToF32(converted, source);
            built = true;
            return converted;
        }
        finally
        {
            if (!built)
                converted.Dispose();
        }
    }
}

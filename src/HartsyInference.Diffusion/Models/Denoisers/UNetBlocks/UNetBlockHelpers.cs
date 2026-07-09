using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;

/// <summary>Shared helpers for UNet block implementations.</summary>
public static class UNetBlockHelpers
{
    /// <summary>Clones a tensor without leaving the device. <c>Tensor.To(Device)</c> reads <c>DataPointer</c>, which forces a full GPU→host sync + host memcpy of the activation (and a re-upload when the clone is consumed) — for the UNet skip connections that was ~9 pipeline drains and ~50 MB of PCIe round-trips per forward. A unit-scale kernel pass produces the same independent copy entirely on-device.</summary>
    public static Tensor CloneOnDevice(IBackend backend, Tensor source)
    {
        Tensor copy = new Tensor(source.Shape, source.DType);
        backend.Scale(copy, source, 1.0f);
        return copy;
    }
}

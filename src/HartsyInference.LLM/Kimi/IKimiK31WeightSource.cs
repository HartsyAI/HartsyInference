using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Kimi;

/// <summary>Where <see cref="KimiK31WeightLoader"/> reads a Kimi K3.1 checkpoint's tensors: safetensors on disk, or any store with the
/// checkpoint's names.</summary>
public interface IKimiK31WeightSource
{
    /// <summary>Whether the source holds a tensor called <paramref name="name"/>.</summary>
    bool Has(string name);

    /// <summary>The tensor as the caller disposes it, in any dtype; the loader converts it to F32 where it needs F32.</summary>
    /// <exception cref="KeyNotFoundException">The source has no tensor <paramref name="name"/>.</exception>
    Tensor Read(string name);
}

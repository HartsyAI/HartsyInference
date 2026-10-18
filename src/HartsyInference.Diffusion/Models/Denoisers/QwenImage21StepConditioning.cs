using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>One denoise step's conditioning: the timestep embedding and the modulation every block reads. Both CFG
/// branches share the step's timestep, so it is built once by <see cref="QwenImage21Transformer.PrepareStep"/> and
/// passed to each branch. A struct, so the denoise loop does not allocate per step; release it once with
/// <c>using</c>.</summary>
public readonly struct QwenImage21StepConditioning(Tensor temb, QwenImage21Modulation modulation) : IDisposable
{
    /// <summary>The timestep embedding, <c>[1, hidden]</c> F32; the final layer reads it.</summary>
    public Tensor Temb { get; } = temb;

    /// <summary>The shared modulation every block reads.</summary>
    public QwenImage21Modulation Modulation { get; } = modulation;

    /// <inheritdoc/>
    public void Dispose()
    {
        Temb.Dispose();
        Modulation.Dispose();
    }
}

using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Features;

/// <summary>Reference images prepared by <see cref="ReferenceImageResolver"/>, in picture order; owns every tensor.</summary>
public sealed class ResolvedReferences : IDisposable
{
    private int _disposed;

    /// <summary>VAE copies <c>[1, 3, H, W]</c> in <c>[-1, 1]</c>, sides multiples of the policy's latent multiple.</summary>
    public required IReadOnlyList<Tensor> Latent { get; init; }

    /// <summary>Vision-tower copies <c>[1, 3, H, W]</c> in <c>[0, 1]</c>; empty when the policy makes none.</summary>
    public required IReadOnlyList<Tensor> Vision { get; init; }

    /// <summary>References.</summary>
    public int Count => Latent.Count;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Tensor tensor in Latent) tensor.Dispose();
        foreach (Tensor tensor in Vision) tensor.Dispose();
    }
}

using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Requests;

/// <summary>Reference images for in-context conditioning, in picture order, each prepared twice: once for the VAE (the
/// DiT's reference tokens) and once for the text encoder's vision tower. The tensors are borrowed; the caller disposes
/// them after the generation.</summary>
public sealed record ReferenceImageInputs
{
    /// <summary>VAE copies <c>[1, 3, H, W]</c> in <c>[-1, 1]</c>, sides multiples of the VAE token stride.</summary>
    public required IReadOnlyList<Tensor> Latent { get; init; }

    /// <summary>Vision-tower copies <c>[1, 3, H, W]</c> in <c>[0, 1]</c>, same count and order as <see cref="Latent"/>;
    /// empty when the text encoder takes no images.</summary>
    public IReadOnlyList<Tensor> Vision { get; init; } = [];

    /// <summary>The attention layout and cache budget.</summary>
    public ReferenceConditioningOptions Options { get; init; } = ReferenceConditioningOptions.Default;

    /// <summary>Reference images.</summary>
    public int Count => Latent.Count;

    /// <summary>Throws unless the two lists line up and every tensor is a single <c>[1, 3, H, W]</c> image.</summary>
    public void Validate(string family)
    {
        ArgumentNullException.ThrowIfNull(Latent);
        ArgumentNullException.ThrowIfNull(Vision);
        if (Latent.Count == 0)
            throw new ArgumentException($"{family}: reference conditioning needs at least one image.");
        if (Vision.Count != 0 && Vision.Count != Latent.Count)
            throw new ArgumentException($"{family}: {Latent.Count} VAE references but {Vision.Count} vision references.");
        foreach (Tensor image in Latent) RequireImage(image, family);
        foreach (Tensor image in Vision) RequireImage(image, family);
    }

    private static void RequireImage(Tensor image, string family)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.Shape.Rank != 4 || image.Shape[0] != 1 || image.Shape[1] != 3)
            throw new ArgumentException($"{family}: a reference image must be [1, 3, H, W], got {image.Shape}.");
    }
}

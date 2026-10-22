using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes;

/// <summary>The resolved sampling values an image pipeline runs with. A pipeline receives these alongside the
/// request, so it can never see an unset tunable or substitute a default of its own.</summary>
public sealed record ImageEffectiveSettings
{
    /// <summary>Output width in pixels before any family-specific alignment.</summary>
    public required int Width { get; init; }

    /// <summary>Output height in pixels before any family-specific alignment.</summary>
    public required int Height { get; init; }

    /// <summary>Denoising steps.</summary>
    public required int Steps { get; init; }

    /// <summary>Guidance scale.</summary>
    public required float CfgScale { get; init; }

    /// <summary>The settings of a request already resolved against its family's <see cref="ImageDefaults"/>.</summary>
    /// <exception cref="ArgumentException">A tunable is still unset: the request skipped resolution.</exception>
    public static ImageEffectiveSettings Of(ImageRequest resolved)
    {
        ArgumentNullException.ThrowIfNull(resolved);
        if (resolved.Width is not int width || resolved.Height is not int height || resolved.Steps is not int steps
            || resolved.CfgScale is not float cfgScale)
        {
            throw new ArgumentException("The image request has unset tunables: resolve it against the family's "
                + "ImageDefaults (ImagesService does) before handing it to a pipeline.", nameof(resolved));
        }
        return new ImageEffectiveSettings { Width = width, Height = height, Steps = steps, CfgScale = cfgScale };
    }
}

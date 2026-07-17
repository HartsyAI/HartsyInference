using HartsyInference.Core.Backends;
using Xunit;

namespace HartsyInference.Diffusion.Tests.GenHarness;

/// <summary>Minimal, model-agnostic generation request the harness hands to each case's delegate. Kept separate from
/// the engine's <c>TextToImageRequest</c> so a delegate is free to map these knobs onto whatever its pipeline needs.</summary>
public sealed record ImageGenRequest
{
    public required string Prompt { get; init; }
    public int Width { get; init; } = 128;
    public int Height { get; init; } = 128;
    public int Steps { get; init; } = 3;
    public float CfgScale { get; init; } = 7.0f;
    public int Seed { get; init; } = 42;
}

/// <summary>What a case's generate delegate returns: raw RGB plus the realized dimensions and the seed actually used.</summary>
public sealed record GenImage(byte[] Rgb, int Width, int Height, int Seed);

/// <summary>One registered model in the generation matrix. A case is self-describing: it knows how to tell whether its
/// weights are present (<see cref="IsAvailable"/>) and how to turn a request into pixels (<see cref="Generate"/>).
/// Registering a new model is a single entry in <see cref="ImageModelManifest"/> — the harness owns selection,
/// skipping, backend choice, validity assertions, deterministic-matrix expansion, and seeded fuzz.</summary>
public sealed record ImageGenCase
{
    /// <summary>Stable, unique display name — also the MemberData key, so it must be deterministic and file-safe.</summary>
    public required string Name { get; init; }

    /// <summary>True only when every file this case needs (checkpoint, tokenizer, VAE, …) is present on disk.</summary>
    public required Func<bool> IsAvailable { get; init; }

    /// <summary>Loads the model on the given backend and renders one image. Called only after <see cref="IsAvailable"/>.</summary>
    public required Func<IBackend, ImageGenRequest, GenImage> Generate { get; init; }
}

/// <summary>Shared assertions for "did this actually render a plausible image" — one place so every model in the matrix
/// is held to the same bar (right dimensions, correct byte count, and neither a flat black nor flat white frame, which
/// is how a broken VAE / NaN latent / dead pipeline typically presents).</summary>
public static class ImageValidity
{
    public static void AssertRenderable(GenImage image, ImageGenRequest request)
    {
        Assert.Equal(request.Width, image.Width);
        Assert.Equal(request.Height, image.Height);
        Assert.Equal(request.Width * request.Height * 3, image.Rgb.Length);

        int nonZero = 0;
        int nonFf = 0;
        foreach (byte b in image.Rgb)
        {
            if (b != 0) nonZero++;
            if (b != 255) nonFf++;
        }
        float nonZeroPct = nonZero / (float)image.Rgb.Length * 100f;
        float nonFfPct = nonFf / (float)image.Rgb.Length * 100f;
        Assert.True(nonZeroPct > 10f, $"Image appears to be all black ({nonZeroPct:F1}% non-zero bytes).");
        Assert.True(nonFfPct > 10f, $"Image appears to be all white ({nonFfPct:F1}% non-255 bytes).");
    }
}

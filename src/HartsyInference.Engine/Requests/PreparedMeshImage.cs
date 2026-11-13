namespace HartsyInference.Engine.Requests;

/// <summary>An image as a 3D model will see it, with what the preparation did, so a host can show "this is what the model gets" and let the user fix a bad cutout.</summary>
public sealed record PreparedMeshImage
{
    /// <summary>The image the model is fed: RGB, and the alpha plane too for a model that reads the cutout itself (Paint, SAM 3D). For TRELLIS.2 and Pixal3D it is the premultiplied square their conditioner resizes.</summary>
    public required ImageData Image { get; init; }

    /// <summary>The source image with the alpha the preparation used (the caller's or the matte), so the user can correct it and send it back; null when the model takes no image or prepares its own input where the cutout cannot be shown (HairDeck, SAM 3D). For TRELLIS.2 and Pixal3D it is shrunk to 1024 px on the long edge, the size their pipelines matte at.</summary>
    public ImageData? Cutout { get; init; }

    /// <summary>True when the engine matted the image because it carried no alpha.</summary>
    public bool Matted { get; init; }

    /// <summary>What the preparation did or why it did nothing, one line each.</summary>
    public IReadOnlyList<string> Notes { get; init; } = [];
}

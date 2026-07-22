namespace HartsyInference.Engine.Requests;

/// <summary>Which vision operation a <see cref="VisionRequest"/> performs.</summary>
public enum VisionMode
{
    /// <summary>Produce an image embedding.</summary>
    Embed,

    /// <summary>Detect objects (bounding boxes), optionally text-conditioned.</summary>
    Detect,

    /// <summary>Produce segmentation masks, optionally text-conditioned.</summary>
    Segment,

    /// <summary>Produce a relative-depth grayscale map (DepthAnything-V2).</summary>
    Depth,

    /// <summary>Produce a ControlNet-style soft-edge map (HED).</summary>
    Edge,

    /// <summary>Produce a ControlNet-style line-art map (realistic or coarse variant).</summary>
    Lineart,

    /// <summary>Produce a ControlNet-style surface-normal map (NormalBAE).</summary>
    Normal,

    /// <summary>Produce an ADE20K semantic-segmentation palette map (UperNet-ConvNeXt).</summary>
    SegMap,

    /// <summary>Produce the foreground image with the background removed (RMBG-1.4).</summary>
    BackgroundRemoval,
}

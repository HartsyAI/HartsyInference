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
}

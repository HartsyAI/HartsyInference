namespace HartsyInference.Engine.Vision;

/// <summary>Which detector/segmenter a vision target string routes to.</summary>
public enum VisionTargetKind
{
    /// <summary>Closed-set COCO detection with RT-DETR (the default when no text query is given).</summary>
    RtDetr,

    /// <summary>Class-prompted detection with a named YOLO checkpoint (<c>yolo-MODEL[-INDEX][:CLASS]</c>).</summary>
    Yolo,

    /// <summary>Open-vocabulary, text-conditioned detection with Grounding DINO.</summary>
    GroundingDino,

    /// <summary>Free-text segmentation with CLIPSeg.</summary>
    ClipSeg,
}

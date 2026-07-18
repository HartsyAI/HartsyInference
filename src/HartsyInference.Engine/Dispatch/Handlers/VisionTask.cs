namespace HartsyInference.Engine.Dispatch.Handlers;

/// <summary>The vision task a loaded model performs, inferred from the model id.</summary>
public enum VisionTask
{
    /// <summary>Image embedding (CLIP).</summary>
    Embed,

    /// <summary>Object detection (YOLO).</summary>
    Detect,
}

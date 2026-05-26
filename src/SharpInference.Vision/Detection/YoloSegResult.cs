namespace SharpInference.Vision.Detection;

/// <summary>A detection plus its associated binary segmentation mask in original-image coordinates.
/// The mask is a flat HxW grid of 0/1 bytes (row-major).</summary>
public sealed class YoloSegResult
{
    /// <summary>Detection box / confidence / class id.</summary>
    public required YoloDetection Detection { get; init; }

    /// <summary>Binary mask, row-major <c>[height × width]</c> in <i>source-image</i> coordinates.
    /// Pixels inside the instance are 1; outside are 0.</summary>
    public required byte[] Mask { get; init; }

    /// <summary>Mask width (= source image width).</summary>
    public required int Width { get; init; }

    /// <summary>Mask height (= source image height).</summary>
    public required int Height { get; init; }
}

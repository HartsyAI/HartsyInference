using System.Globalization;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Renders the depth plane an <see cref="ImageResult"/> carries, for consumers that deliver it as an image.</summary>
public static class DepthImage
{
    /// <summary>Meta key holding the sky ceiling of the model that produced <see cref="ImageResult.Depth"/>; below 1, pixels
    /// at +1 are sky.</summary>
    public const string SkyCeilingMeta = "sky_ceiling";

    /// <summary>Meta key naming the task that produced the result.</summary>
    public const string TaskMeta = "task";

    /// <summary>The depth plane as colour-coded RGB24 (near red, far blue, sky black), or null when the result has none.</summary>
    public static byte[]? Colourize(ImageResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        if (result.Depth is not { } depth || depth.Length != (long)result.Width * result.Height)
            return null;
        float skyCeiling = result.Meta.TryGetValue(SkyCeilingMeta, out string? text)
            && float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed) ? parsed : 1f;
        return IrisZDepth.Colourize(depth, IrisZDepth.SkyMask(depth, skyCeiling));
    }

    /// <summary>The colour-coded depth to deliver beside the image, or null when there is no depth or the image itself
    /// already is the depth rendering (the <c>depth</c> task).</summary>
    public static byte[]? Companion(ImageResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        if (result.Meta.TryGetValue(TaskMeta, out string? task) && string.Equals(task, "depth", StringComparison.OrdinalIgnoreCase))
            return null;
        return Colourize(result);
    }
}

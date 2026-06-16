using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation;

/// <summary>Text-prompted image segmentation (e.g. CLIPSeg). Produces a soft mask for a free-text prompt over an image, the missing half of the <c>&lt;segment:text&gt;</c> story (SAM is point/box-prompted; YOLO is class-prompted). The returned <c>float[]</c> feeds <c>RegionMask.FromBitmap</c> for regional conditioning.</summary>
public interface ITextSegmenter
{
    /// <summary>Returns a row-major <c>[height*width]</c> soft mask in <c>[0, 1]</c> for <paramref name="prompt"/> over <paramref name="image"/> (<c>[1, 3, H, W]</c>). <paramref name="width"/>/<paramref name="height"/> report the mask resolution.</summary>
    float[] Segment(IBackend backend, Tensor image, string prompt, out int width, out int height);
}

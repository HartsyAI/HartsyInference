using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Vision;

/// <summary>Pixel packings the detectors need beyond <see cref="Features.FeatureImaging"/>: the ImageNet-normalized
/// <c>[1,3,H,W]</c> tensor shared by Grounding DINO (DETR preprocessing) and SAM 2.</summary>
public static class VisionTensors
{
    /// <summary>ImageNet channel means used by the DETR-family preprocessors.</summary>
    public static readonly float[] ImageNetMean = [0.485f, 0.456f, 0.406f];

    /// <summary>ImageNet channel standard deviations used by the DETR-family preprocessors.</summary>
    public static readonly float[] ImageNetStd = [0.229f, 0.224f, 0.225f];

    /// <summary>Packs interleaved RGB24 into a <c>[1,3,H,W]</c> F32 tensor normalized as <c>(x/255 - mean) / std</c>.</summary>
    public static unsafe Tensor ImageNetNormalized(ReadOnlySpan<byte> rgb, int width, int height)
    {
        Tensor tensor = new Tensor(new TensorShape(1, 3, height, width), DType.F32);
        float* dp = (float*)tensor.DataPointer;
        int spatial = width * height;
        const float Inv255 = 1f / 255f;
        for (int c = 0; c < 3; c++)
        {
            float mean = ImageNetMean[c];
            float invStd = 1f / ImageNetStd[c];
            long chOff = (long)c * spatial;
            for (int i = 0; i < spatial; i++)
            {
                dp[chOff + i] = (rgb[i * 3 + c] * Inv255 - mean) * invStd;
            }
        }
        return tensor;
    }
}

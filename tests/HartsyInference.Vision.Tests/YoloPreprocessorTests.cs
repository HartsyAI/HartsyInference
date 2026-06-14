using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Tests for the YOLO letterbox preprocessor — no model weights required.</summary>
public sealed class YoloPreprocessorTests
{
    [Fact]
    public void Preprocess_SquareInput_ProducesExpectedShape()
    {
        YoloPreprocessor pre = new(targetSize: 640);
        byte[] rgb = new byte[640 * 640 * 3];
        Array.Fill<byte>(rgb, 100);

        (Tensor input, YoloPreprocessor.Transform tr) = pre.Preprocess(rgb, 640, 640);
        try
        {
            Assert.Equal(4, input.Shape.Rank);
            Assert.Equal(1, input.Shape[0]);
            Assert.Equal(3, input.Shape[1]);
            Assert.Equal(640, input.Shape[2]);
            Assert.Equal(640, input.Shape[3]);
            Assert.Equal(DType.F32, input.DType);
            Assert.Equal(1f, tr.Scale);
            Assert.Equal(0, tr.PadLeft);
            Assert.Equal(0, tr.PadTop);
        }
        finally { input.Dispose(); }
    }

    [Fact]
    public void Preprocess_WideInput_LetterboxesWithVerticalPadding()
    {
        // 1280 wide, 640 tall → scale = 640/1280 = 0.5 → resized 640×320
        // 320 rows of image + 320 rows of padding (160 top, 160 bottom) → 640×640 canvas.
        YoloPreprocessor pre = new(targetSize: 640);
        byte[] rgb = new byte[1280 * 640 * 3];
        Array.Fill<byte>(rgb, 200);

        (Tensor input, YoloPreprocessor.Transform tr) = pre.Preprocess(rgb, 1280, 640);
        try
        {
            Assert.Equal(640, tr.PaddedWidth);
            Assert.Equal(640, tr.PaddedHeight);
            Assert.Equal(640, tr.ResizedWidth);
            Assert.Equal(320, tr.ResizedHeight);
            Assert.Equal(0.5f, tr.Scale);
            Assert.Equal(0, tr.PadLeft);
            Assert.Equal(160, tr.PadTop);

            // Top-left corner (in the padding region) should be the pad value.
            ReadOnlySpan<float> data = input.AsReadOnlySpan<float>();
            float padNorm = YoloPreprocessor.PadValue / 255f;
            int plane = 640 * 640;
            // R channel, row 0, col 0
            Assert.InRange(data[0 * plane + 0], padNorm - 1e-4f, padNorm + 1e-4f);
            // R channel, row 320 (in image region), col 320
            float imgVal = 200f / 255f;
            int imgPos = (160 + 159) * 640 + 320; // bottom row of image area
            Assert.InRange(data[0 * plane + imgPos], imgVal - 1e-3f, imgVal + 1e-3f);
        }
        finally { input.Dispose(); }
    }

    [Fact]
    public void Preprocess_TallInput_LetterboxesWithHorizontalPadding()
    {
        // 320 wide, 640 tall → scale = 640/640 = 1.0 (limited by height) → resized 320×640
        // 320 cols of image + 320 cols of padding (160 left, 160 right) → 640×640.
        YoloPreprocessor pre = new(targetSize: 640);
        byte[] rgb = new byte[320 * 640 * 3];
        Array.Fill<byte>(rgb, 50);

        (Tensor input, YoloPreprocessor.Transform tr) = pre.Preprocess(rgb, 320, 640);
        try
        {
            Assert.Equal(640, tr.PaddedWidth);
            Assert.Equal(640, tr.PaddedHeight);
            Assert.Equal(320, tr.ResizedWidth);
            Assert.Equal(640, tr.ResizedHeight);
            Assert.Equal(1f, tr.Scale);
            Assert.Equal(160, tr.PadLeft);
            Assert.Equal(0, tr.PadTop);
        }
        finally { input.Dispose(); }
    }

    [Fact]
    public void Transform_Invert_RoundTrips_BoxFromOriginalToLetterboxAndBack()
    {
        YoloPreprocessor pre = new(targetSize: 640);
        YoloPreprocessor.Transform tr = pre.ComputeTransform(srcWidth: 1280, srcHeight: 720);
        // 1280 → 640 (scale 0.5), 720 → 360 (scale 0.5). Padding centered top/bottom (140 each).

        // Pick an arbitrary source-space box, map it forward into letterbox coords, then invert.
        float srcX1 = 200f, srcY1 = 100f, srcX2 = 500f, srcY2 = 400f;
        // Forward: letterboxCoord = src * scale + pad
        float lx1 = srcX1 * tr.Scale + tr.PadLeft;
        float ly1 = srcY1 * tr.Scale + tr.PadTop;
        float lx2 = srcX2 * tr.Scale + tr.PadLeft;
        float ly2 = srcY2 * tr.Scale + tr.PadTop;

        (float x1, float y1, float x2, float y2) = tr.Invert(lx1, ly1, lx2, ly2);

        Assert.InRange(x1, srcX1 - 1e-3f, srcX1 + 1e-3f);
        Assert.InRange(y1, srcY1 - 1e-3f, srcY1 + 1e-3f);
        Assert.InRange(x2, srcX2 - 1e-3f, srcX2 + 1e-3f);
        Assert.InRange(y2, srcY2 - 1e-3f, srcY2 + 1e-3f);
    }

    [Fact]
    public void Transform_Invert_ClampsToSourceImageBounds()
    {
        YoloPreprocessor pre = new(targetSize: 640);
        YoloPreprocessor.Transform tr = pre.ComputeTransform(srcWidth: 800, srcHeight: 600);
        // A detection that spills into the padding region should clamp to the image edge.
        // PadLeft = 0 (since 800>600 → letterbox is horizontal-fit), pad is top/bottom.
        // Pad above the image: y < tr.PadTop. Inverse of a point at y=0 is negative → clamp to 0.
        (float x1, float y1, float x2, float y2) = tr.Invert(-50f, -50f, 1000f, 1000f);
        Assert.Equal(0f, x1);
        Assert.Equal(0f, y1);
        Assert.Equal(800f, x2);
        Assert.Equal(600f, y2);
    }

    [Fact]
    public void Preprocess_StridePadAlign_ProducesNonSquareCanvas()
    {
        // Wide input + stridePadAlign=true should leave only minimal top/bottom padding rounded
        // to a stride multiple, producing a non-square (e.g. 640×384) canvas — Ultralytics
        // auto=True behavior.
        YoloPreprocessor pre = new(targetSize: 640, stride: 32, stridePadAlign: true);
        (_, YoloPreprocessor.Transform tr) = pre.Preprocess(new byte[1280 * 640 * 3], 1280, 640);
        Assert.Equal(640, tr.PaddedWidth);
        Assert.Equal(320, tr.ResizedHeight);
        // Pad rows = (640 - 320) % 32 = 320 % 32 = 0 → canvas height equals resized height.
        Assert.Equal(320, tr.PaddedHeight);
    }

    [Fact]
    public void Constructor_RejectsBadStride()
    {
        Assert.Throws<ArgumentException>(() => new YoloPreprocessor(targetSize: 640, stride: 0));
        Assert.Throws<ArgumentException>(() => new YoloPreprocessor(targetSize: 640, stride: 30)); // 640 % 30 != 0
    }
}

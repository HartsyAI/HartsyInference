using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Annotators;
using HartsyInference.Vision.Codec;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Unit tier: HED / lineart pre- and post-processing math (no weights, no GPU).</summary>
public sealed unsafe class AnnotatorPreprocessorTests
{
    [Fact]
    public void RgbToTensor255_KeepsRawRangeInPlanarLayout()
    {
        byte[] rgb = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        using Tensor t = ImageTensor.RgbToTensor255(rgb, 2, 2);
        Assert.Equal(new TensorShape(1, 3, 2, 2), t.Shape);
        ReadOnlySpan<float> s = t.AsReadOnlySpan<float>();
        Assert.Equal([10f, 40f, 70f, 100f], s[..4].ToArray());
        Assert.Equal([20f, 50f, 80f, 110f], s[4..8].ToArray());
        Assert.Equal([30f, 60f, 90f, 120f], s[8..].ToArray());
    }

    [Fact]
    public void QuantizeU8_TruncatesLikeNumpyAstype()
    {
        Assert.Equal(254, HedPreprocessor.QuantizeU8(0.9999f));
        Assert.Equal(255, HedPreprocessor.QuantizeU8(1.0f));
        Assert.Equal(255, HedPreprocessor.QuantizeU8(1.5f));
        Assert.Equal(0, HedPreprocessor.QuantizeU8(0.001f));
        Assert.Equal(127, HedPreprocessor.QuantizeU8(0.5f));
        Assert.Equal(0, HedPreprocessor.QuantizeU8(0f));
    }

    [Fact]
    public void SafeStep_MatchesControlnetAuxFormula()
    {
        // y = float(int(x * 3)) / 2
        Assert.Equal(0f, HedPreprocessor.SafeStep(0.32f));
        Assert.Equal(0.5f, HedPreprocessor.SafeStep(0.34f));
        Assert.Equal(1f, HedPreprocessor.SafeStep(0.7f));
        Assert.Equal(1.5f, HedPreprocessor.SafeStep(1.0f));
    }

    [Fact]
    public void GaussianKernel_IsNormalizedAndSymmetric()
    {
        double[] k = HedPreprocessor.GaussianKernel(25, 3.0);
        Assert.Equal(1.0, k.Sum(), 12);
        for (int i = 0; i < 12; i++)
            Assert.Equal(k[i], k[24 - i], 15);
        Assert.True(k[12] > k[11]);
    }

    [Fact]
    public void GaussianBlur_ConstantImageIsInvariant()
    {
        float[] src = new float[16 * 9];
        Array.Fill(src, 42f);
        float[] blurred = HedPreprocessor.GaussianBlur(src, 16, 9, 3.0, 25);
        foreach (float v in blurred)
            Assert.Equal(42f, v, 3);
    }

    [Fact]
    public void ScribbleFromSoftEdge_KeepsRidgeDropsFlat()
    {
        // A bright vertical ridge should survive NMS + binarization; the dark background must not.
        // Width 8 keeps the σ=3-blurred peak above the 127 NMS threshold (255·Σ central taps ≈ 200).
        const int w = 64, h = 64;
        byte[] gray = new byte[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 27; x <= 34; x++)
                gray[y * w + x] = 255;
        byte[] scribble = HedPreprocessor.ScribbleFromSoftEdge(gray, w, h);
        Assert.Equal(255, scribble[32 * w + 31]);
        Assert.Equal(0, scribble[32 * w + 2]);
        Assert.All(scribble, b => Assert.True(b == 0 || b == 255));
    }

    [Fact]
    public void ReflectionPad2d_MirrorsWithoutRepeatingBorder()
    {
        using Tensor t = new(new TensorShape(1, 1, 2, 3), DType.F32);
        Span<float> s = t.AsSpan<float>();
        s[0] = 1; s[1] = 2; s[2] = 3;
        s[3] = 4; s[4] = 5; s[5] = 6;

        using Tensor padded = LineartGenerator.ReflectionPad2d(t, 1);
        Assert.Equal(new TensorShape(1, 1, 4, 5), padded.Shape);
        float[] expected =
        [
            5, 4, 5, 6, 5,
            2, 1, 2, 3, 2,
            5, 4, 5, 6, 5,
            2, 1, 2, 3, 2,
        ];
        Assert.Equal(expected, padded.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void ReflectionPad2d_RejectsPadNotSmallerThanDims()
    {
        using Tensor t = new(new TensorShape(1, 1, 3, 3), DType.F32);
        Assert.Throws<ArgumentException>(() => LineartGenerator.ReflectionPad2d(t, 3));
    }

    [Fact]
    public void LineartPostprocess_InvertsOnU8Grid()
    {
        using Tensor line = new(new TensorShape(1, 1, 1, 4), DType.F32);
        Span<float> s = line.AsSpan<float>();
        s[0] = 1f;      // white bg → 0 (black) after inversion
        s[1] = 0f;      // black line → 1 (white)
        s[2] = 0.5f;    // 127 → 128/255
        s[3] = 0.9999f; // truncates to 254 → 1/255
        float[] unit = LineartPreprocessor.PostprocessToUnit(line);
        Assert.Equal(0f, unit[0]);
        Assert.Equal(1f, unit[1]);
        Assert.Equal(128f / 255f, unit[2], 6);
        Assert.Equal(1f / 255f, unit[3], 6);
    }

    [Fact]
    public void ZeroPad2dSame_PadsMoreOnBottomRight()
    {
        // k=3, s=2 on even dims: TF SAME pads (0 top/left, 1 bottom/right).
        using Tensor t = new(new TensorShape(1, 1, 2, 2), DType.F32);
        Span<float> s = t.AsSpan<float>();
        s[0] = 1; s[1] = 2; s[2] = 3; s[3] = 4;
        using Tensor padded = NormalBaeModel.ZeroPad2dSame(t, 3, 2, 2, 2);
        Assert.Equal(new TensorShape(1, 1, 3, 3), padded.Shape);
        float[] expected = [1, 2, 0, 3, 4, 0, 0, 0, 0];
        Assert.Equal(expected, padded.AsReadOnlySpan<float>().ToArray());

        // k=5, s=2 on even dims: total 3 → (1 top/left, 2 bottom/right).
        using Tensor padded5 = NormalBaeModel.ZeroPad2dSame(t, 5, 2, 2, 2);
        Assert.Equal(new TensorShape(1, 1, 5, 5), padded5.Shape);
        Assert.Equal(0f, padded5.AsReadOnlySpan<float>()[0]);
        Assert.Equal(1f, padded5.AsReadOnlySpan<float>()[6]);
        Assert.Equal(4f, padded5.AsReadOnlySpan<float>()[12]);
    }

    [Fact]
    public void NormNormalize_UnitLengthXyzAndEluKappa()
    {
        using Tensor t = new(new TensorShape(1, 4, 1, 2), DType.F32);
        Span<float> s = t.AsSpan<float>();
        s[0] = 3f; s[1] = 0f;      // x
        s[2] = 0f; s[3] = -2f;     // y
        s[4] = 4f; s[5] = 0f;      // z
        s[6] = 2f; s[7] = -1f;     // kappa
        NormalBaeModel.NormNormalize(t);
        ReadOnlySpan<float> r = t.AsReadOnlySpan<float>();
        Assert.Equal(0.6f, r[0], 5);
        Assert.Equal(0.8f, r[4], 5);
        Assert.Equal(-1f, r[3], 5);
        Assert.Equal(3.01f, r[6], 5);                       // kappa > 0: k + 1.01
        Assert.Equal(MathF.Exp(-1f) - 1f + 1.01f, r[7], 5); // kappa < 0: elu
    }

    [Fact]
    public void NormalBaePostprocess_MapsNormalsToRgb()
    {
        using Tensor t = new(new TensorShape(1, 4, 1, 1), DType.F32);
        Span<float> s = t.AsSpan<float>();
        s[0] = -1f; s[1] = 0f; s[2] = 1f; s[3] = 5f;
        byte[] rgb = NormalBaePreprocessor.PostprocessToRgb24(t);
        Assert.Equal([0, 127, 255], rgb);
    }

    [Fact]
    public void HedPostprocess_QuantizesToU8Grid()
    {
        using Tensor edge = new(new TensorShape(1, 1, 1, 3), DType.F32);
        Span<float> s = edge.AsSpan<float>();
        s[0] = 0.25f; s[1] = 0.75f; s[2] = 1f;
        float[] unit = HedPreprocessor.PostprocessToUnit(edge);
        Assert.Equal(63f / 255f, unit[0], 6);
        Assert.Equal(191f / 255f, unit[1], 6);
        Assert.Equal(1f, unit[2]);

        float[] safe = HedPreprocessor.PostprocessToUnit(edge, safe: true);
        Assert.Equal(0f, safe[0]);                 // floor(0.75)/2 = 0
        Assert.Equal(255f / 255f, safe[2], 6);     // 1.5 clips to 255
    }

    [Fact]
    public void Ade20kPalette_MatchesControlnetAuxAnchors()
    {
        Assert.Equal(150, Ade20kPalette.ClassCount);
        Assert.Equal(0x787878u, Ade20kPalette.Color(0));   // wall [120,120,120]
        Assert.Equal(0x0066C8u, Ade20kPalette.Color(20));  // car [0,102,200]
        Assert.Equal(0x5C00FFu, Ade20kPalette.Color(149)); // last entry [92,0,255]
        Assert.Throws<ArgumentOutOfRangeException>(() => Ade20kPalette.Color(150));

        byte[] rgb = Ade20kPalette.Colorize([0, 149]);
        Assert.Equal([120, 120, 120, 92, 0, 255], rgb);
    }

    [Fact]
    public void UperNetSegPreprocess_NormalizesWithImageNetStats()
    {
        byte[] rgb = [255, 0, 128, 255, 0, 128, 255, 0, 128, 255, 0, 128];
        using Tensor t = UperNetSegPreprocessor.Preprocess(rgb, 2, 2);
        Assert.Equal(new TensorShape(1, 3, 2, 2), t.Shape);
        ReadOnlySpan<float> s = t.AsReadOnlySpan<float>();
        Assert.Equal((1f - 0.485f) / 0.229f, s[0], 5);
        Assert.Equal((0f - 0.456f) / 0.224f, s[4], 5);
        Assert.Equal((128f / 255f - 0.406f) / 0.225f, s[8], 5);
    }

    [Fact]
    public void UperNetSegArgmax_TakesLowestIndexOnTies()
    {
        using Tensor logits = new(new TensorShape(1, 3, 1, 2), DType.F32);
        Span<float> s = logits.AsSpan<float>();
        // Planar [K, H·W]: pixel 0 sees {1, 1, 0.5} — tie between classes 0 and 1 → 0.
        // Pixel 1 sees {0, 2, 3} → 2.
        s[0] = 1f; s[2] = 1f; s[4] = 0.5f;
        s[1] = 0f; s[3] = 2f; s[5] = 3f;
        byte[] classes = UperNetSegPreprocessor.Argmax(logits);
        Assert.Equal([0, 2], classes);
    }
}

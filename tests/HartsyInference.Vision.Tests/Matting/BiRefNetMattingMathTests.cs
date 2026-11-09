using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Matting.BiRefNet;

namespace HartsyInference.Vision.Tests.Matting;

public sealed unsafe class BiRefNetMattingMathTests
{
    private const float Tolerance = 1e-5f;
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    [Fact]
    public void Preprocess_NormalizesWhiteAndBlackToImageNetRange()
    {
        const int size = 7;
        byte[] white = Enumerable.Repeat((byte)255, size * size * 3).ToArray();
        byte[] black = new byte[size * size * 3];

        using Tensor whiteT = BiRefNetMatter.Preprocess(white, size, size);
        using Tensor blackT = BiRefNetMatter.Preprocess(black, size, size);

        Assert.Equal(new long[] { 1, 3, BiRefNetMatter.InputSize, BiRefNetMatter.InputSize }, Dims(whiteT));
        int plane = BiRefNetMatter.InputSize * BiRefNetMatter.InputSize;
        float* w = (float*)whiteT.DataPointer, b = (float*)blackT.DataPointer;
        for (int c = 0; c < 3; c++)
        {
            Assert.Equal((1f - Mean[c]) / Std[c], w[c * plane + 12345], 4);
            Assert.Equal((0f - Mean[c]) / Std[c], b[c * plane + 999], 4);
        }
    }

    [Fact]
    public void Preprocess_KeepsChannelOrderInterleavedRgbToPlanar()
    {
        const int size = 4;
        byte[] rgb = new byte[size * size * 3];
        for (int i = 0; i < size * size; i++)
        {
            rgb[i * 3 + 0] = 255;
            rgb[i * 3 + 1] = 0;
            rgb[i * 3 + 2] = 128;
        }

        using Tensor t = BiRefNetMatter.Preprocess(rgb, size, size);

        int plane = BiRefNetMatter.InputSize * BiRefNetMatter.InputSize;
        float* p = (float*)t.DataPointer;
        Assert.Equal((1f - Mean[0]) / Std[0], p[0], 4);
        Assert.Equal((0f - Mean[1]) / Std[1], p[plane], 4);
        Assert.Equal((128f / 255f - Mean[2]) / Std[2], p[2 * plane], 3);
    }

    [Fact]
    public void Postprocess_ZeroLogitsGiveHalfCoverageAtSourceSize()
    {
        using Tensor logits = Filled(new TensorShape(1, 1, BiRefNetMatter.InputSize, BiRefNetMatter.InputSize), 0f);

        float[] matte = BiRefNetMatter.Postprocess(logits, 5, 3);

        Assert.Equal(15, matte.Length);
        foreach (float v in matte)
        {
            Assert.InRange(v, 0f, 1f);
            Assert.Equal(127f / 255f, v, 6);
        }
    }

    [Fact]
    public void Postprocess_SaturatedLogitsGiveFullAndEmptyCoverage()
    {
        const int plane = BiRefNetMatter.InputSize * BiRefNetMatter.InputSize;
        using Tensor logits = Filled(new TensorShape(1, 1, BiRefNetMatter.InputSize, BiRefNetMatter.InputSize), 0f);
        float* lp = (float*)logits.DataPointer;
        for (int i = 0; i < plane; i++) lp[i] = i < plane / 2 ? 30f : -30f;

        float[] matte = BiRefNetMatter.Postprocess(logits, 8, 8);

        Assert.Equal(1f, matte[0], 6);
        Assert.Equal(0f, matte[63], 6);
    }

    private static Tensor Filled(TensorShape shape, float value)
    {
        Tensor t = new(shape, DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }

    private static long[] Dims(Tensor t)
    {
        long[] dims = new long[t.Shape.Rank];
        for (int i = 0; i < dims.Length; i++) dims[i] = t.Shape[i];
        return dims;
    }
}

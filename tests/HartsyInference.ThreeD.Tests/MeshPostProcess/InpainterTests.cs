using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic images: Telea fast-marching inpainting fills gutters smoothly and leaves known texels alone.</summary>
public sealed class InpainterTests
{
    private const int Size = 32;

    [Fact]
    public void Hole_InALinearGradientIsFilledNearTheTrueValues()
    {
        float[] image = new float[Size * Size];
        bool[] known = new bool[Size * Size];
        for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                image[y * Size + x] = (x + 2 * y) / 100f;
                known[y * Size + x] = !(x is >= 12 and < 20 && y is >= 12 and < 20);
            }

        float[] filled = Inpainter.Inpaint(image, Size, Size, 1, known, 3);

        for (int i = 0; i < image.Length; i++)
        {
            if (known[i]) Assert.Equal(image[i], filled[i]);
            else Assert.InRange(Math.Abs(filled[i] - image[i]), 0, 0.03);
        }
    }

    [Fact]
    public void Gutters_AroundAKnownDiscStayWithinTheKnownRangeAndAreSmooth()
    {
        float[] image = new float[Size * Size];
        bool[] known = new bool[Size * Size];
        for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                double d = Math.Sqrt((x - 15.5) * (x - 15.5) + (y - 15.5) * (y - 15.5));
                known[y * Size + x] = d < 9;
                image[y * Size + x] = known[y * Size + x] ? 0.3f + 0.4f * (float)(x / (double)Size) : 0f;
            }

        float[] filled = Inpainter.Inpaint(image, Size, Size, 1, known, 3);

        float lo = float.MaxValue, hi = float.MinValue;
        for (int i = 0; i < image.Length; i++)
            if (known[i]) { lo = Math.Min(lo, image[i]); hi = Math.Max(hi, image[i]); }
        double maxStep = 0;
        for (int y = 0; y < Size; y++)
            for (int x = 0; x < Size; x++)
            {
                int i = y * Size + x;
                Assert.False(float.IsNaN(filled[i]));
                Assert.InRange(filled[i], lo - 0.1f, hi + 0.1f);
                if (x + 1 < Size) maxStep = Math.Max(maxStep, Math.Abs(filled[i] - filled[i + 1]));
                if (y + 1 < Size) maxStep = Math.Max(maxStep, Math.Abs(filled[i] - filled[i + Size]));
            }
        Assert.True(maxStep < 0.15, $"Largest neighbor step {maxStep}.");
    }

    [Fact]
    public void ConstantImage_StaysConstantInEveryChannel()
    {
        float[] image = new float[Size * Size * 3];
        bool[] known = new bool[Size * Size];
        for (int i = 0; i < Size * Size; i++)
        {
            known[i] = i % Size < 10;
            image[i * 3] = known[i] ? 0.25f : 0f;
            image[i * 3 + 1] = known[i] ? 0.5f : 0f;
            image[i * 3 + 2] = known[i] ? 0.75f : 0f;
        }

        float[] filled = Inpainter.Inpaint(image, Size, Size, 3, known, 2);

        for (int i = 0; i < Size * Size; i++)
        {
            Assert.Equal(0.25f, filled[i * 3], 4);
            Assert.Equal(0.5f, filled[i * 3 + 1], 4);
            Assert.Equal(0.75f, filled[i * 3 + 2], 4);
        }
    }

    [Fact]
    public void InputIsNotModifiedAndResultIsDeterministic()
    {
        float[] image = new float[Size * Size];
        bool[] known = new bool[Size * Size];
        for (int i = 0; i < image.Length; i++)
        {
            known[i] = i % 5 != 0 && i % 7 != 0;
            image[i] = known[i] ? (i % 11) / 10f : -1f;
        }
        float[] copy = (float[])image.Clone();

        float[] a = Inpainter.Inpaint(image, Size, Size, 1, known, 2);
        float[] b = Inpainter.Inpaint(image, Size, Size, 1, known, 2);

        Assert.Equal(copy, image);
        Assert.Equal(a, b);
    }

    [Fact]
    public void NothingKnown_OrEverythingKnown_ReturnsTheImageUnchanged()
    {
        float[] image = [1f, 2f, 3f, 4f];

        Assert.Equal(image, Inpainter.Inpaint(image, 2, 2, 1, [false, false, false, false], 1));
        Assert.Equal(image, Inpainter.Inpaint(image, 2, 2, 1, [true, true, true, true], 1));
    }

    [Fact]
    public void RejectsMismatchedShapes()
    {
        Assert.Throws<ArgumentException>(() => Inpainter.Inpaint(new float[4], 2, 2, 1, new bool[3], 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Inpainter.Inpaint(new float[4], 2, 2, 1, new bool[4], 0));
    }

    [Fact]
    public void ClampToUnit_HoldsFilledValuesInRangeWhereTheGradientTermRunsAway()
    {
        // A steep ramp that ends at 1 against a wide unknown band: extrapolating its gradient overshoots without the clamp.
        const int Wide = 64;
        float[] image = new float[Wide * Wide];
        bool[] known = new bool[Wide * Wide];
        for (int y = 0; y < Wide; y++)
            for (int x = 0; x < Wide; x++)
            {
                image[y * Wide + x] = Math.Min(1f, x / 8f);
                known[y * Wide + x] = x < 9;
            }

        float[] free = Inpainter.Inpaint(image, Wide, Wide, 1, known, 3);
        float[] clamped = Inpainter.Inpaint(image, Wide, Wide, 1, known, 3, clampToUnit: true);

        Assert.Contains(free, v => v is > 1.001f or < -0.001f);
        Assert.All(clamped, v => Assert.InRange(v, 0f, 1f));
    }
}

using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Prompting;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 text bundles: the per-token RMS normalisation and the tap order the denoiser's bundle split relies on.</summary>
public sealed unsafe class Fibo2TextBundlesTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void Constants_MatchTheFiboTapTable()
    {
        Assert.Equal(new int[] { 5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35 }, Fibo2TextBundles.TapLayers);
        Assert.Equal(6, Fibo2TextBundles.BundleTaps.Count);
        Assert.Equal(new int[] { 9, 20, 31 }, Fibo2TextBundles.BundleTaps[0]);
        Assert.Equal(new int[] { 5, 16, 27 }, Fibo2TextBundles.BundleTaps[1]);
        Assert.Equal(new int[] { 7, 18, 29 }, Fibo2TextBundles.BundleTaps[2]);
        Assert.Equal(new int[] { 9, 20, 31 }, Fibo2TextBundles.BundleTaps[3]);
        Assert.Equal(new int[] { 11, 22, 33 }, Fibo2TextBundles.BundleTaps[4]);
        Assert.Equal(new int[] { 13, 24, 35 }, Fibo2TextBundles.BundleTaps[5]);
        Assert.Equal(Fibo2TextBundles.BundleTaps[0], Fibo2TextBundles.BundleTaps[3]);
    }

    /// <summary>Hidden 2, three taps, two bundles. Tap 0 is (3, 4): mean square 12.5, so it normalises to (3, 4) / sqrt(12.5).
    /// Tap 1 is (1, 1), which is already unit RMS. Tap 2 is (0, 2), which normalises to (0, sqrt 2). The bundle table
    /// [[2, 0], [1, 1]] therefore pins both the normalisation and the concatenation order.</summary>
    [Fact]
    public void Core_NormalisesEachTapPerToken_AndConcatenatesInBundleOrder()
    {
        using Tensor taps = FromFloats(new float[] { 3f, 4f, 1f, 1f, 0f, 2f }, new TensorShape(1, 1, 6));
        int[][] bundles = [[2, 0], [1, 1]];

        using Tensor output = Fibo2TextBundles.Build(taps, hidden: 2, tapCount: 3, bundles);

        Assert.Equal(new TensorShape(1, 1, 8), output.Shape);
        float s2 = MathF.Sqrt(2f);
        float a = 3f / MathF.Sqrt(12.5f);
        float b = 4f / MathF.Sqrt(12.5f);
        AssertFloats([0f, s2, a, b, 1f, 1f, 1f, 1f], output);
    }

    [Fact]
    public void Core_IsInvariantToTheScaleOfEachTap()
    {
        using Tensor scaled = FromFloats(new float[] { 6f, 8f, 7f, 7f, 0f, 14f }, new TensorShape(1, 1, 6));
        int[][] bundles = [[2, 0], [1, 1]];

        using Tensor output = Fibo2TextBundles.Build(scaled, hidden: 2, tapCount: 3, bundles);

        float s2 = MathF.Sqrt(2f);
        float a = 3f / MathF.Sqrt(12.5f);
        float b = 4f / MathF.Sqrt(12.5f);
        AssertFloats([0f, s2, a, b, 1f, 1f, 1f, 1f], output);
    }

    /// <summary>The Fibo wrapper on 15 taps and hidden 2. Even packed slots are (1, 0) and odd ones (0, 1), so each bundle's
    /// expected values follow from its slot parities. Layers map to slots by their position in the tap list: bundle 2 reads
    /// layers 7, 18, 29, which are slots 1, 6, 11.</summary>
    [Fact]
    public void FiboWrapper_MapsLayersToSlots_AndWritesSixBundlesPerToken()
    {
        const int hidden = 2;
        float[] taps = new float[2 * 15 * hidden];
        for (int token = 0; token < 2; token++)
        {
            for (int slot = 0; slot < 15; slot++)
            {
                float scale = token == 0 ? 1f : 3f;
                int offset = (token * 15 + slot) * hidden;
                taps[offset] = (slot % 2 == 0 ? 1f : 0f) * scale;
                taps[offset + 1] = (slot % 2 == 0 ? 0f : 1f) * scale;
            }
        }
        using Tensor input = FromFloats(taps, new TensorShape(1, 2, 15 * hidden));

        using Tensor output = Fibo2TextBundles.Build(input, hidden);

        Assert.Equal(new TensorShape(1, 2, 6 * 3 * hidden), output.Shape);
        // A one-hot pair (1, 0) has mean square 1/2, so it normalises to (sqrt 2, 0).
        float r = MathF.Sqrt(2f);
        float[] bundle0 = [r, 0f, 0f, r, r, 0f];
        float[] bundle1 = [r, 0f, 0f, r, r, 0f];
        float[] bundle2 = [0f, r, r, 0f, 0f, r];
        float[] bundle4 = [0f, r, r, 0f, 0f, r];
        float[] bundle5 = [r, 0f, 0f, r, r, 0f];
        for (int token = 0; token < 2; token++)
        {
            float[] row = Floats(output, token * 36, 36);
            Assert.True(Same(bundle0, row.AsSpan(0, 6)), "bundle 0 (layers 9, 20, 31)");
            Assert.True(Same(bundle1, row.AsSpan(6, 6)), "bundle 1 (layers 5, 16, 27)");
            Assert.True(Same(bundle2, row.AsSpan(12, 6)), "bundle 2 (layers 7, 18, 29)");
            Assert.True(Same(bundle0, row.AsSpan(18, 6)), "bundle 3 must equal bundle 0");
            Assert.True(Same(bundle4, row.AsSpan(24, 6)), "bundle 4 (layers 11, 22, 33)");
            Assert.True(Same(bundle5, row.AsSpan(30, 6)), "bundle 5 (layers 13, 24, 35)");
        }
    }

    [Fact]
    public void Build_RejectsTapsOfTheWrongWidthOrDType()
    {
        using Tensor narrow = FromFloats(new float[6], new TensorShape(1, 1, 6));
        Assert.Throws<ArgumentException>(() => Fibo2TextBundles.Build(narrow, hidden: 2));

        using Tensor f16 = new(new TensorShape(1, 1, 30), DType.F16);
        Assert.Throws<ArgumentException>(() => Fibo2TextBundles.Build(f16, hidden: 2));
    }

    private static bool Same(float[] expected, ReadOnlySpan<float> actual)
    {
        if (actual.Length != expected.Length)
            return false;
        for (int i = 0; i < expected.Length; i++)
        {
            if (MathF.Abs(expected[i] - actual[i]) > Tolerance)
                return false;
        }
        return true;
    }

    private static void AssertFloats(float[] expected, Tensor actual)
    {
        float[] values = Floats(actual, 0, expected.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - values[i]) <= Tolerance, $"index {i}: expected {expected[i]}, got {values[i]}");
    }

    private static float[] Floats(Tensor tensor, int offset, int count)
    {
        float[] values = new float[count];
        new ReadOnlySpan<float>((float*)tensor.DataPointer + offset, count).CopyTo(values);
        return values;
    }

    private static Tensor FromFloats(float[] values, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        values.AsSpan().CopyTo(new Span<float>((void*)tensor.DataPointer, values.Length));
        return tensor;
    }
}

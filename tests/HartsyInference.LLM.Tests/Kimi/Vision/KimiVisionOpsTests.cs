using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi.Vision;
using HartsyInference.LLM.Multimodal;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi.Vision;

/// <summary>Patch order, position resampling and the patch merger, each against a small case worked out by hand.</summary>
public sealed class KimiVisionOpsTests
{
    private const float Tol = 1e-5f;

    [Fact]
    public void Patchify_EmitsEachMergeBlockAsFourConsecutiveRows()
    {
        // Pixel (c, y, x) holds c·100 + y·10 + x, so each row names the pixel it came from.
        float[] pixels = new float[3 * 4 * 4];
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < 4; y++)
                for (int x = 0; x < 4; x++)
                    pixels[(c * 4 + y) * 4 + x] = c * 100 + y * 10 + x;
        using Tensor image = Make(pixels, 1, 3, 4, 4);
        using IBackend backend = new CpuBackend();

        using Tensor rows = VisionPatchOps.Patchify(backend, image, 4, 4, 2, 1);
        float[] v = rows.AsReadOnlySpan<float>().ToArray();

        Assert.Equal(new float[] { 1, 101, 201 }, v.AsSpan(1 * 3, 3).ToArray());
        Assert.Equal(new float[] { 10, 110, 210 }, v.AsSpan(2 * 3, 3).ToArray());
        Assert.Equal(new float[] { 2, 102, 202 }, v.AsSpan(4 * 3, 3).ToArray());
        Assert.Equal(new float[] { 20, 120, 220 }, v.AsSpan(8 * 3, 3).ToArray());
        Assert.Equal(new float[] { 22, 122, 222 }, v.AsSpan(12 * 3, 3).ToArray());
    }

    [Fact]
    public void Resample_AveragesTheFourCornersWhenTheGridHalves()
    {
        // A 2x2 table sampled at one token sits halfway between all four entries.
        float[] table = [1f, 2f, 3f, 4f];
        float[] dst = new float[1];
        KimiVisionPosEmbed.Resample(dst, table, 2, 1, 1, 1, 1);

        Assert.Equal(2.5f, dst[0], Tol);
    }

    [Fact]
    public void Resample_IsTheIdentityOnTheNativeGrid()
    {
        float[] table = [0.5f, -1f, 2f, 3f, 0.25f, -0.75f, 1f, 4f, -2f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f];
        float[] dst = new float[table.Length];
        KimiVisionPosEmbed.Resample(dst, table, 3, 2, 3, 3, 1);

        for (int i = 0; i < table.Length; i++) Assert.Equal(table[i], dst[i], Tol);
    }

    [Fact]
    public void Projector_MatchesHandComputedGoldenForOneMergedToken()
    {
        // Four tokens of width one merge into one 4-wide row [1, 2, 0, 0]. GELU gives [g(1), g(2), 0, 0] with g(1) = 0.8413447460685429
        // and g(2) = 1.9544997361036416. Identity up-projection, then the two outputs are g(1) and g(2), then RMS over the two.
        float g1 = 0.8413447460685429f, g2 = 1.9544997361036416f;
        float rms = MathF.Sqrt((g1 * g1 + g2 * g2) / 2f + 1e-5f);
        using Tensor tokens = Make([1f, 2f, 0f, 0f], 4, 1);
        using Tensor up = Make(Identity(4), 4, 4);
        using Tensor down = Make([1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f], 2, 4);
        using Tensor norm = Make([1f, 1f], 2);
        using IBackend backend = new CpuBackend();

        using Tensor merged = KimiVisionProjector.Apply(backend, tokens, up, down, norm, 2, 1e-5f);

        float[] actual = merged.AsReadOnlySpan<float>().ToArray();
        Assert.Equal(2, actual.Length);
        Assert.Equal(g1 / rms, actual[0], 1e-4f);
        Assert.Equal(g2 / rms, actual[1], 1e-4f);
    }

    private static float[] Identity(int n)
    {
        float[] m = new float[n * n];
        for (int i = 0; i < n; i++) m[i * n + i] = 1f;
        return m;
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}

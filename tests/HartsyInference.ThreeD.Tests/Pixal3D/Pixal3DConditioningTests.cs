using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Pipelines.Stages;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Pixal3D;

/// <summary>The token split and the negative condition of Pixal3D's conditioning pair, on synthetic DINOv3 tokens.</summary>
public sealed class Pixal3DConditioningTests
{
    private const int ImageSize = 64, Patches = 4, Globals = 5, Channels = 3;

    // Global rows hold their row number; patch (py, px) holds (px, py, 7), so a bilinear sample of the map reads back its own source position.
    private static Tensor Tokens()
    {
        Tensor tokens = new(new TensorShape(1, Globals + Patches * Patches, Channels), DType.F32);
        Span<float> v = tokens.AsSpan<float>();
        for (int g = 0; g < Globals; g++)
            v.Slice(g * Channels, Channels).Fill(100 + g);
        for (int py = 0; py < Patches; py++)
        {
            for (int px = 0; px < Patches; px++)
            {
                int o = (Globals + py * Patches + px) * Channels;
                v[o] = px;
                v[o + 1] = py;
                v[o + 2] = 7;
            }
        }

        return tokens;
    }

    [Fact]
    public void Split_KeepsTheGlobalTokensAndSamplesThePatchMapAtEachCell()
    {
        using Tensor tokens = Tokens();
        Pixal3DView view = Pixal3DView.Default;
        int[] coords = [0, 3, 7, 12, 0, 8, 8, 8, 0, 15, 0, 15];

        using Pixal3DCondition cond = Pixal3DConditionerStage.Split(tokens, ImageSize, 16, coords, view);

        Assert.Equal(new TensorShape(1, Globals, Channels), cond.Global.Shape);
        for (int g = 0; g < Globals; g++)
            Assert.Equal(100f + g, cond.Global.AsReadOnlySpan<float>()[g * Channels]);
        Assert.Equal(new TensorShape(1, 3, Channels), cond.Proj.Shape);
        float[] uv = Pixal3DProjection.CellCoordinates(coords, 16, view, ImageSize);
        ReadOnlySpan<float> proj = cond.Proj.AsReadOnlySpan<float>();
        for (int i = 0; i < 3; i++)
        {
            // grid_sample (align_corners false, border): source = clamp(((coord + 1) · size - 1) / 2, 0, size - 1), exact for a linear map.
            float expectedX = Math.Clamp(((uv[i * 2] + 1f) * Patches - 1f) / 2f, 0f, Patches - 1f);
            float expectedY = Math.Clamp(((uv[i * 2 + 1] + 1f) * Patches - 1f) / 2f, 0f, Patches - 1f);
            Assert.Equal(expectedX, proj[i * Channels], 1e-5);
            Assert.Equal(expectedY, proj[i * Channels + 1], 1e-5);
            Assert.Equal(7f, proj[i * Channels + 2], 1e-5);
        }
    }

    [Fact]
    public void Split_WithNoCoordinatesCoversTheDenseGridInTokenOrder()
    {
        using Tensor tokens = Tokens();

        using Pixal3DCondition cond = Pixal3DConditionerStage.Split(tokens, ImageSize, 4, [], Pixal3DView.Default);

        Assert.Equal(new TensorShape(1, 64, Channels), cond.Proj.Shape);
        // The first cell is grid (0, 0, 0) and the last (3, 3, 3); both are sampled somewhere inside the patch map.
        Assert.All(cond.Proj.AsReadOnlySpan<float>().ToArray(), x => Assert.True(float.IsFinite(x)));
    }

    [Fact]
    public void Split_WithANafQueryMapAppendsTheUpsampledSamplesAfterThePatchSamples()
    {
        // A 144 px image has 9x9 patches, the smallest map NAF's 9-cell window fits; the query map is 18x18 (dilation 2).
        const int size = 144, patches = 9, nafSize = 18, globals = 5, width = 8;
        Tensor tokens = new(new TensorShape(1, globals + patches * patches, width), DType.F32);
        Random rng = new(3);
        foreach (ref float v in tokens.AsSpan<float>())
            v = (float)(rng.NextDouble() * 2 - 1);
        Tensor query = new(new TensorShape(1, Pixal3DNaf.Dim, nafSize, nafSize), DType.F32);
        foreach (ref float v in query.AsSpan<float>())
            v = (float)(rng.NextDouble() * 2 - 1);
        int[] coords = [0, 1, 2, 3, 0, 14, 5, 9];
        using (tokens)
        using (query)
        {
            using Pixal3DCondition plain = Pixal3DConditionerStage.Split(tokens, size, 16, coords, Pixal3DView.Default);
            using Pixal3DCondition joined = Pixal3DConditionerStage.Split(tokens, size, 16, coords, Pixal3DView.Default, query, nafSize);

            Assert.Equal(new TensorShape(1, 2, 2 * width), joined.Proj.Shape);
            float[] uv = Pixal3DProjection.CellCoordinates(coords, 16, Pixal3DView.Default, size);
            using Tensor expectedNaf = Pixal3DNaf.SampleUpsampled(
                query.AsReadOnlySpan<float>(), nafSize, nafSize, tokens.AsReadOnlySpan<float>()[(globals * width)..], patches, patches, width, uv);
            ReadOnlySpan<float> a = plain.Proj.AsReadOnlySpan<float>(), b = expectedNaf.AsReadOnlySpan<float>(), actual = joined.Proj.AsReadOnlySpan<float>();
            for (int row = 0; row < 2; row++)
            {
                for (int c = 0; c < width; c++)
                {
                    Assert.Equal(a[row * width + c], actual[row * 2 * width + c]);
                    Assert.Equal(b[row * width + c], actual[row * 2 * width + width + c]);
                }
            }
        }
    }

    [Fact]
    public void Split_RejectsTokensThatLeaveNoGlobalRows()
    {
        using Tensor patchesOnly = new(new TensorShape(1, Patches * Patches, Channels), DType.F32);

        Assert.Throws<ArgumentException>(() => Pixal3DConditionerStage.Split(patchesOnly, ImageSize, 4, [], Pixal3DView.Default));
    }

    [Fact]
    public void ZerosLike_BuildsTheUnconditionalPairWithBothShapes()
    {
        using Tensor tokens = Tokens();
        using Pixal3DCondition cond = Pixal3DConditionerStage.Split(tokens, ImageSize, 4, [], Pixal3DView.Default);

        using Pixal3DCondition negative = cond.ZerosLike();

        Assert.Equal(cond.Global.Shape, negative.Global.Shape);
        Assert.Equal(cond.Proj.Shape, negative.Proj.Shape);
        Assert.All(negative.Global.AsReadOnlySpan<float>().ToArray(), x => Assert.Equal(0f, x));
        Assert.All(negative.Proj.AsReadOnlySpan<float>().ToArray(), x => Assert.Equal(0f, x));
    }
}

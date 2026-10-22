using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Double-condition CFG's text-only context zeroes the lyric rows, which the DiT's context appends last.
/// Counting them from the front took the text row count from the leading dim of the engine's <c>[1, T, 768]</c>
/// features.</summary>
public sealed class AceStepPipelineOnlyTextContextTests
{
    [Fact]
    public void BuildOnlyTextContext_ZeroesExactlyTheTrailingLyricRows()
    {
        const int textRows = 3, lyricRows = 2, dim = 4;
        int rows = 1 + textRows + lyricRows;
        using Tensor ctx = new Tensor(new TensorShape(rows, dim), DType.F32);
        ctx.AsSpan<float>().Fill(1f);

        using Tensor onlyText = AceStepPipeline.BuildOnlyTextContext(ctx, lyricRows);

        float[] expected = new float[rows * dim];
        expected.AsSpan(0, (1 + textRows) * dim).Fill(1f);
        Assert.Equal(expected, onlyText.AsSpan<float>().ToArray());
    }

    [Fact]
    public void BuildOnlyTextContext_RefusesMoreLyricRowsThanTheContextHolds()
    {
        using Tensor ctx = new Tensor(new TensorShape(3, 4), DType.F32);

        Assert.Throws<ArgumentOutOfRangeException>(() => AceStepPipeline.BuildOnlyTextContext(ctx, 3));
    }
}

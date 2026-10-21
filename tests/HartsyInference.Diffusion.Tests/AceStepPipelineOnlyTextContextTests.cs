using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Double-condition CFG's text-only context keeps the speaker and text rows and zeroes the lyric rows. The
/// engine passes the UMT5 features as <c>[1, T, 768]</c>, so the text rows are not the leading dim.</summary>
public sealed class AceStepPipelineOnlyTextContextTests
{
    [Fact]
    public void BuildOnlyTextContext_BatchOfOneText_ZeroesOnlyTheLyricRows()
    {
        const int textRows = 3, lyricRows = 2, dim = 4, textDim = 6;
        int rows = 1 + textRows + lyricRows;
        using Tensor ctx = new Tensor(new TensorShape(rows, dim), DType.F32);
        ctx.AsSpan<float>().Fill(1f);
        using Tensor text = new Tensor(new TensorShape(1, textRows, textDim), DType.F32);

        using Tensor onlyText = AceStepPipeline.BuildOnlyTextContext(ctx, text, textDim);

        float[] expected = new float[rows * dim];
        expected.AsSpan(0, (1 + textRows) * dim).Fill(1f);
        Assert.Equal(expected, onlyText.AsSpan<float>().ToArray());
    }
}

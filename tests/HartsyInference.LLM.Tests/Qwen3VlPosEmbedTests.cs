using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Multimodal;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The learned position table resizes align-corners, as HF (`interpolation_align_corners = True`) and llama.cpp
/// (`GGML_SCALE_FLAG_ALIGN_CORNERS`) do.</summary>
public sealed class Qwen3VlPosEmbedTests
{
    private const int Grid = 4, Hidden = 3;

    [Theory]
    [InlineData(6, 6, 2)]
    [InlineData(2, 8, 2)]
    [InlineData(4, 4, 2)]
    [InlineData(1, 3, 1)]
    public void Interpolation_IsAlignCornersBilinear(int gh, int gw, int m)
    {
        Random rng = new(5);
        using Tensor table = new(new TensorShape(Grid * Grid, Hidden), DType.F32);
        Span<float> t = table.AsSpan<float>();
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        using Tensor embed = new(new TensorShape(gh * gw, Hidden), DType.F32);
        Qwen3VlEncoder.AddInterpolatedPosEmbed(embed, table, Grid, Hidden, gh, gw, m);

        ReadOnlySpan<float> got = embed.AsReadOnlySpan<float>();
        int token = 0;
        for (int bh = 0; bh < gh / m; bh++)
            for (int bw = 0; bw < gw / m; bw++)
                for (int mh = 0; mh < m; mh++)
                    for (int mw = 0; mw < m; mw++, token++)
                    {
                        double fy = Src(bh * m + mh, gh), fx = Src(bw * m + mw, gw);
                        int y0 = (int)fy, x0 = (int)fx, y1 = Math.Min(y0 + 1, Grid - 1), x1 = Math.Min(x0 + 1, Grid - 1);
                        double dy = fy - y0, dx = fx - x0;
                        for (int c = 0; c < Hidden; c++)
                        {
                            double want = (1 - dy) * (1 - dx) * t[(y0 * Grid + x0) * Hidden + c] + (1 - dy) * dx * t[(y0 * Grid + x1) * Hidden + c]
                                + dy * (1 - dx) * t[(y1 * Grid + x0) * Hidden + c] + dy * dx * t[(y1 * Grid + x1) * Hidden + c];
                            Assert.True(Math.Abs(want - got[token * Hidden + c]) < 1e-5, $"token {token} ch {c}: {got[token * Hidden + c]} vs {want}");
                        }
                    }
    }

    private static double Src(int i, int n) => n == 1 ? 0 : i * (Grid - 1.0) / (n - 1);
}

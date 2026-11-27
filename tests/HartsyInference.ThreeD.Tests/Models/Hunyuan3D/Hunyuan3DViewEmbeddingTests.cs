using HartsyInference.ThreeD.Models.Hunyuan3D;

namespace HartsyInference.ThreeD.Tests.Models.Hunyuan3D;

public sealed class Hunyuan3DViewEmbeddingTests
{
    [Fact]
    public void Create_ForTheFirstViewIsZeroSinesThenOneCosines()
    {
        float[] embedding = Hunyuan3DViewEmbedding.Create(0, 1536);

        Assert.Equal(1536, embedding.Length);
        Assert.All(embedding.Take(768), v => Assert.Equal(0f, v));
        Assert.All(embedding.Skip(768), v => Assert.Equal(1f, v));
    }

    [Fact]
    public void Create_MatchesUpstreamGet1dSincosPosEmbedFromGrid()
    {
        // numpy get_1d_sincos_pos_embed_from_grid(1536, arange(4)).astype(float32), rows 1 and 2
        float[] one = Hunyuan3DViewEmbedding.Create(1, 1536);
        float[] two = Hunyuan3DViewEmbedding.Create(2, 1536);

        Assert.Equal(0.8414709568023682f, one[0], 6);
        Assert.Equal(0.8349704146385193f, one[1], 6);
        Assert.Equal(0.8284307718276978f, one[2], 6);
        Assert.Equal(0.00010120648221345618f, one[767], 6);
        Assert.Equal(0.5403022766113281f, one[768], 6);
        Assert.Equal(0.5502948760986328f, one[769], 6);
        Assert.Equal(1f, one[1535], 6);
        Assert.Equal(0.9092974066734314f, two[0], 6);
        Assert.Equal(0.9189598560333252f, two[1], 6);
        Assert.Equal(-0.416146844625473f, two[768], 6);
        Assert.Equal(-0.39435112476348877f, two[769], 6);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    public void Create_RejectsAnOddOrEmptyWidth(int width)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Hunyuan3DViewEmbedding.Create(0, width));
    }
}

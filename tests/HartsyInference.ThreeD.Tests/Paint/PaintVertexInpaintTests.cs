using HartsyInference.ThreeD.Paint;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Unit lane: upstream's mesh-vertex inpaint, on a strip of triangles split into two UV charts.</summary>
public sealed class PaintVertexInpaintTests
{
    // A strip of two quads along x. The left quad's chart sits at the texture's left, the right quad's chart at its right, and the shared edge (x = 1) is split: vertices 1 and 4,
    // 2 and 5 have the same positions but different UVs.
    private static readonly float[] Positions = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 1, 0];
    private static readonly float[] Uvs = [0.05f, 0.05f, 0.3f, 0.05f, 0.3f, 0.3f, 0.05f, 0.3f, 0.7f, 0.05f, 0.7f, 0.3f, 0.95f, 0.05f, 0.95f, 0.3f];
    private static readonly int[] Indices = [0, 1, 2, 0, 2, 3, 4, 6, 7, 4, 7, 5];

    private static int Texel(int vertex, int size) => (int)(Uvs[vertex * 2 + 1] * size) * size + (int)(Uvs[vertex * 2] * size);

    [Fact]
    public void Seed_CarriesColourAcrossASeamToAChartNoViewPainted()
    {
        int size = 16;
        float[] texture = new float[size * size * 3];
        bool[] known = new bool[size * size];
        foreach (int vertex in new[] { 0, 1, 2, 3 })
        {
            int texel = Texel(vertex, size);
            known[texel] = true;
            (texture[texel * 3], texture[texel * 3 + 1], texture[texel * 3 + 2]) = (0.9f, 0.6f, 0.5f);
        }

        int seeded = PaintVertexInpaint.Seed(texture, known, size, size, 3, Positions, Uvs, Indices);

        Assert.Equal(4, seeded);
        foreach (int vertex in new[] { 4, 5, 6, 7 })
        {
            int texel = Texel(vertex, size);
            Assert.True(known[texel]);
            Assert.Equal(0.9f, texture[texel * 3], 4);
            Assert.Equal(0.5f, texture[texel * 3 + 2], 4);
        }
    }

    [Fact]
    public void Seed_NeverOverwritesAPaintedTexel()
    {
        int size = 16;
        float[] texture = new float[size * size * 3];
        bool[] known = new bool[size * size];
        for (int vertex = 0; vertex < 8; vertex++)
        {
            int texel = Texel(vertex, size);
            known[texel] = true;
            texture[texel * 3] = vertex / 10f;
        }

        Assert.Equal(0, PaintVertexInpaint.Seed(texture, known, size, size, 3, Positions, Uvs, Indices));
        for (int vertex = 0; vertex < 8; vertex++)
            Assert.Equal(vertex / 10f, texture[Texel(vertex, size) * 3], 6);
    }
}

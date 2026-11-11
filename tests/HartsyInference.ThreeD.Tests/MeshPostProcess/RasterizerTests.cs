using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic triangles: UV-space rasterization coverage, barycentrics and overlap resolution.</summary>
public sealed class RasterizerTests
{
    [Fact]
    public void Triangle_CoversExactlyTheTexelsWhoseCentersAreInside()
    {
        // Lower-left half of the square: covers texel centers with x + y < 8 on an 8x8 grid (the diagonal ones sit on the edge).
        float[] uvs = [0, 0, 1, 0, 0, 1];

        RasterizedUvMap map = Rasterizer.Rasterize(uvs, [0, 1, 2], 8, 8);

        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
            {
                bool inside = (x + 0.5) + (y + 0.5) <= 8.0 + 1e-9;
                Assert.Equal(inside ? 0 : -1, map.TriangleIds[y * 8 + x]);
            }
    }

    [Fact]
    public void Barycentrics_ReproduceTheTexelCenterUv()
    {
        float[] uvs = [0.1f, 0.2f, 0.9f, 0.3f, 0.4f, 0.95f];
        int[] idx = [0, 1, 2];

        RasterizedUvMap map = Rasterizer.Rasterize(uvs, idx, 64, 64);

        int covered = 0;
        for (int y = 0; y < 64; y++)
            for (int x = 0; x < 64; x++)
            {
                int texel = y * 64 + x;
                if (map.TriangleIds[texel] < 0) continue;
                covered++;
                float b1 = map.Barycentric[texel * 2], b2 = map.Barycentric[texel * 2 + 1], b0 = 1 - b1 - b2;
                Assert.Equal((x + 0.5) / 64, b0 * uvs[0] + b1 * uvs[2] + b2 * uvs[4], 5);
                Assert.Equal((y + 0.5) / 64, b0 * uvs[1] + b1 * uvs[3] + b2 * uvs[5], 5);
            }
        Assert.InRange(covered, 700, 1500);
    }

    [Fact]
    public void Winding_DoesNotMatter()
    {
        float[] uvs = [0, 0, 1, 0, 0, 1];

        RasterizedUvMap ccw = Rasterizer.Rasterize(uvs, [0, 1, 2], 16, 16);
        RasterizedUvMap cw = Rasterizer.Rasterize(uvs, [0, 2, 1], 16, 16);

        Assert.Equal(ccw.TriangleIds, cw.TriangleIds);
    }

    [Fact]
    public void Overlap_IsWonByTheLowestTriangleIndex()
    {
        float[] uvs = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1];

        RasterizedUvMap map = Rasterizer.Rasterize(uvs, [3, 4, 5, 0, 1, 2], 8, 8);

        Assert.Equal(0, map.TriangleIds[0]);
    }

    [Fact]
    public void Output_IsIdenticalAcrossRunsDespiteThreading()
    {
        Random rng = new(5);
        float[] uvs = new float[600];
        for (int i = 0; i < uvs.Length; i++) uvs[i] = (float)rng.NextDouble();
        int[] idx = new int[300];
        for (int i = 0; i < idx.Length; i++) idx[i] = i;

        RasterizedUvMap a = Rasterizer.Rasterize(uvs, idx, 256, 256);
        RasterizedUvMap b = Rasterizer.Rasterize(uvs, idx, 256, 256);

        Assert.Equal(a.TriangleIds, b.TriangleIds);
        Assert.Equal(a.Barycentric, b.Barycentric);
    }

    [Fact]
    public void RejectsBadInput()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Rasterizer.Rasterize([0, 0, 1, 0, 0, 1], [0, 1, 2], 0, 4));
        Assert.Throws<ArgumentException>(() => Rasterizer.Rasterize([0, 0, 1, 0, 0, 1], [0, 1, 3], 4, 4));
    }
}

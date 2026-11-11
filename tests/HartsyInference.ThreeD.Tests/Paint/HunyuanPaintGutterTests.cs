using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;
using HartsyInference.ThreeD.Paint;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Unit lane: the far gutter of a baked texture takes the mean covered colour and nothing the mesh covers is touched.</summary>
public sealed class HunyuanPaintGutterTests
{
    private const int Size = 64;

    [Fact]
    public void FlattenFarGutter_ReplacesOnlyTexelsFarFromTheMesh()
    {
        Mesh mesh = new()
        {
            Vertices = [-0.5f, -0.5f, 0f, 0.5f, -0.5f, 0f, -0.5f, 0.5f, 0f],
            Indices = [0, 1, 2],
            Uvs = [0.1f, 0.1f, 0.3f, 0.1f, 0.1f, 0.3f],
        };
        PaintTexelSurface surface = PaintTexelSurface.Create(PaintMeshFrame.Create(mesh), mesh.Uvs, Size, Size);
        Assert.True(surface.Count > 50);
        float[] texture = new float[Size * Size * 3];
        for (int i = 0; i < surface.Texels.Length; i++)
            Array.Fill(texture, i % 2 == 0 ? 1f : 0.5f, surface.Texels[i] * 3, 3);
        double mean = 0;
        for (int i = 0; i < surface.Texels.Length; i++)
            mean += i % 2 == 0 ? 1.0 : 0.5;
        mean /= surface.Texels.Length;
        HashSet<int> covered = [.. surface.Texels];
        int inside = surface.Texels[0], far = 60 * Size + 60, near = inside;
        while (covered.Contains(near))
            near++;
        near += 2; // a few texels past the island's edge on the same row: within reach, not covered
        Assert.DoesNotContain(near, covered);
        Assert.DoesNotContain(far, covered);

        HunyuanPaintPipeline.FlattenFarGutter(texture, surface, Size);

        Assert.Equal(1f, texture[inside * 3]);
        Assert.Equal(0f, texture[near * 3]);
        Assert.Equal(mean, texture[far * 3], 4);
        Assert.Equal(mean, texture[far * 3 + 2], 4);
    }
}

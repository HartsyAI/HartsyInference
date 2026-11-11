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
        foreach (int texel in surface.Texels)
            Array.Fill(texture, 1f, texel * 3, 3);
        int covered = surface.Texels[0], near = 8 * Size + 26, far = 60 * Size + 60;

        HunyuanPaintPipeline.FlattenFarGutter(texture, surface, Size);

        Assert.Equal(1f, texture[covered * 3]);
        Assert.Equal(0f, texture[near * 3]);
        Assert.Equal(1f, texture[far * 3]);
        Assert.Equal(1f, texture[far * 3 + 2]);
    }
}

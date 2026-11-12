using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;
using HartsyInference.ThreeD.Paint;
using HartsyInference.ThreeD.Tests.MeshPostProcess;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Regression lane for the Paint input mesh: a mesh whose vertices are split along every face (a triangle soup, or a decimated mesh with duplicated seam vertices) unwraps into one
/// chart per face and fills the atlas with padding, so a 40,000-face test mesh gave 16,252 charts, 12 % atlas fill and 29 % of the atlas reached by the six views, against 4,910 charts, 22 %
/// and 78 % for the same mesh welded. <see cref="HunyuanPaintPipeline.WeldForUnwrap"/> merges the vertices before the pipeline unwraps.</summary>
public sealed class HunyuanPaintWeldTests
{
    private static Mesh Soup(Mesh mesh)
    {
        float[] vertices = new float[mesh.TriangleCount * 9];
        int[] indices = new int[mesh.TriangleCount * 3];
        for (int i = 0; i < mesh.Indices.Length; i++)
        {
            Array.Copy(mesh.Vertices, mesh.Indices[i] * 3, vertices, i * 3, 3);
            indices[i] = i;
        }

        return new Mesh { Vertices = vertices, Indices = indices };
    }

    private static double AtlasFill(Mesh unwrapped, int size)
    {
        PaintMeshFrame frame = PaintMeshFrame.Create(unwrapped);
        PaintTexelSurface surface = PaintTexelSurface.Create(frame, unwrapped.Uvs!, size, size);
        return (double)surface.Count / (size * size);
    }

    [Fact]
    public void WeldForUnwrap_MergesASoupBackIntoOneSharedSurface()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(3);
        Mesh soup = Soup(sphere);
        Assert.Equal(sphere.TriangleCount * 3, soup.VertexCount);

        Mesh welded = HunyuanPaintPipeline.WeldForUnwrap(soup);

        Assert.Equal(sphere.VertexCount, welded.VertexCount);
        Assert.Equal(sphere.TriangleCount, welded.TriangleCount);
    }

    [Fact]
    public void WeldedInput_UnwrapsIntoFewChartsAndAFilledAtlasWhereASoupDoesNot()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(3);
        Mesh soup = Soup(sphere);

        UvUnwrapResult split = UvUnwrapper.Unwrap(soup);
        UvUnwrapResult welded = UvUnwrapper.Unwrap(HunyuanPaintPipeline.WeldForUnwrap(soup));

        // One chart per face (or close) for the soup; a handful of large charts once it is welded.
        Assert.True(split.ChartCount > sphere.TriangleCount / 4, $"the soup gave only {split.ChartCount} charts");
        Assert.True(welded.ChartCount <= 40, $"the welded mesh gave {welded.ChartCount} charts");
        Assert.True(welded.ChartCount * 20 < split.ChartCount);
        double weldedFill = AtlasFill(welded.Mesh, 1024), splitFill = AtlasFill(split.Mesh, 1024);
        Assert.True(weldedFill > 0.3, $"welded atlas fill {weldedFill:P1}");
        Assert.True(weldedFill > 1.3 * splitFill, $"welded {weldedFill:P1} against soup {splitFill:P1}");
    }
}

using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Hunyuan3D.Part;
using HartsyInference.ThreeD.Tests.MeshPostProcess;

namespace HartsyInference.ThreeD.Tests.Part;

/// <summary>The geometry around the X-Part networks: normalisation, the faces and surface samples of a box, the part ids and the initial noise.</summary>
public sealed class XPartSamplingTests
{
    [Fact]
    public void Normalize_CentersTheBoxAndMakesTheLongestSideOnePointSix()
    {
        float[] vertices = [10, 0, 0, 14, 2, 1, 12, 1, 0.5f];
        (float[] normalized, double[] center, double scale) = XPartSampling.Normalize(vertices);

        Assert.Equal([12.0, 1.0, 0.5], center);
        Assert.Equal(2 / 0.8, scale, 9);
        Assert.Equal(-0.8f, normalized[0], 5);
        Assert.Equal(0.8f, normalized[3], 5);
        float[] back = XPartSampling.Denormalize(normalized, center, scale);
        for (int i = 0; i < vertices.Length; i++)
            Assert.Equal(vertices[i], back[i], 4);
        Assert.Throws<ArgumentException>(() => XPartSampling.Normalize([1, 1, 1, 1, 1, 1]));
        Assert.Throws<ArgumentException>(() => XPartSampling.Normalize([1, 2]));
    }

    [Fact]
    public void FacesInBox_TakesAFaceWithAnyVertexInside_AndSurfaceSamplesStayOnThoseFaces()
    {
        Mesh sphere = SyntheticMeshes.Icosphere(2);
        float[] box = [0, -2, -2, 2, 2, 2]; // the x >= 0 half
        int[] faces = XPartSampling.FacesInBox(sphere.Vertices, sphere.Indices, box);

        Assert.InRange(faces.Length, sphere.TriangleCount / 2, sphere.TriangleCount);
        Assert.All(faces, f => Assert.Contains(Enumerable.Range(0, 3), k => sphere.Vertices[sphere.Indices[f * 3 + k] * 3] >= 0));
        Assert.Empty(XPartSampling.FacesInBox(sphere.Vertices, sphere.Indices, [5, 5, 5, 6, 6, 6]));

        int[] sub = XPartSampling.SubIndices(sphere.Indices, faces);
        (float[] points, float[] features) = XPartSampling.SurfaceSamples(sphere.Vertices, sub, 500, 7);
        Assert.Equal(500 * 3, points.Length);
        Assert.Equal(500 * 4, features.Length);
        for (int i = 0; i < 500; i++)
        {
            float x = points[i * 3], y = points[i * 3 + 1], z = points[i * 3 + 2];
            Assert.InRange(MathF.Sqrt(x * x + y * y + z * z), 0.85f, 1.01f); // on the faceted unit sphere
            float nx = features[i * 4], ny = features[i * 4 + 1], nz = features[i * 4 + 2];
            Assert.Equal(1f, MathF.Sqrt(nx * nx + ny * ny + nz * nz), 3);
            Assert.Equal(0f, features[i * 4 + 3]); // the sharp-edge label of the release is 0
        }

        (float[] again, _) = XPartSampling.SurfaceSamples(sphere.Vertices, sub, 500, 7);
        Assert.Equal(points, again);
    }

    [Fact]
    public void KeptParts_SkipsEmptyBoxesAndKeepsTheLargestInOrder()
    {
        Assert.Equal([0, 1, 3], XPartSampling.KeptParts([5, 2, 0, 9], 50));
        Assert.Equal([0, 3], XPartSampling.KeptParts([5, 2, 0, 9], 2));
        Assert.Equal([1, 2], XPartSampling.KeptParts([4, 7, 7, 1], 2)); // 7 and 7 beat 4; ties go to the lower index
        Assert.Empty(XPartSampling.KeptParts([0, 0], 5));
    }

    [Fact]
    public void PartIds_AreAFixedSelectionOfDistinctIds()
    {
        int[] ids = XPartSampling.PartIds(12, 50, 3);
        Assert.Equal(12, ids.Distinct().Count());
        Assert.All(ids, id => Assert.InRange(id, 0, 49));
        Assert.Equal(ids, XPartSampling.PartIds(12, 50, 3));
        Assert.NotEqual(ids, XPartSampling.PartIds(12, 50, 4));
        Assert.Equal(50, XPartSampling.PartIds(50, 50, 1).Distinct().Count());
        Assert.Throws<ArgumentOutOfRangeException>(() => XPartSampling.PartIds(51, 50, 1));
    }

    [Fact]
    public void Gaussian_HasZeroMeanAndUnitSpreadAndIsSeeded()
    {
        float[] noise = XPartSampling.Gaussian(20000, 5);
        double mean = noise.Average(v => (double)v), std = Math.Sqrt(noise.Average(v => (v - mean) * (v - mean)));
        Assert.InRange(mean, -0.05, 0.05);
        Assert.InRange(std, 0.97, 1.03);
        Assert.Equal(noise, XPartSampling.Gaussian(20000, 5));
    }
}

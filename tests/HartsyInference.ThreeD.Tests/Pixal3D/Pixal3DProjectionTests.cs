using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.Pixal3D;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Pixal3D;

/// <summary>Pixal3D's projection and bilinear lookup against values produced by upstream's own <c>ProjGrid</c> and <c>sample_features</c> code
/// (<c>tests/python-reference/dump_pixal3d_projection.py</c>, torch float32), plus the geometry properties the doc states.</summary>
public sealed class Pixal3DProjectionTests
{
    private const double Tolerance = 3e-5;

    private static JsonDocument Fixture()
        => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Pixal3D", "Fixtures", "projection_reference.json")));

    [Fact]
    public void ProjectedCoordinates_MatchUpstreamsProjGrid()
    {
        using JsonDocument fixture = Fixture();
        int checkedCells = 0;
        foreach (JsonElement c in fixture.RootElement.GetProperty("projection").EnumerateArray())
        {
            int resolution = c.GetProperty("grid_resolution").GetInt32(), size = c.GetProperty("image_size").GetInt32();
            Pixal3DView view = new(c.GetProperty("fov").GetDouble(), c.GetProperty("mesh_scale").GetDouble());
            Assert.Equal(c.GetProperty("distance").GetDouble(), view.Distance, 1e-5);

            float[] uv = Pixal3DProjection.DenseCoordinates(resolution, view, size);

            int k = 0;
            foreach (JsonElement cell in c.GetProperty("cells").EnumerateArray())
            {
                JsonElement expected = c.GetProperty("uv")[k++];
                Assert.Equal(expected[0].GetDouble(), uv[cell.GetInt32() * 2], Tolerance);
                Assert.Equal(expected[1].GetDouble(), uv[cell.GetInt32() * 2 + 1], Tolerance);
                checkedCells++;
            }
        }

        Assert.True(checkedCells > 800);
    }

    [Fact]
    public void BilinearBorderSampling_MatchesUpstreamSampleFeatures()
    {
        using JsonDocument fixture = Fixture();
        foreach (JsonElement c in fixture.RootElement.GetProperty("sampling").EnumerateArray())
        {
            int h = c.GetProperty("height").GetInt32(), w = c.GetProperty("width").GetInt32(), channels = c.GetProperty("channels").GetInt32();
            float[] rows = c.GetProperty("rows").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
            float[] uv = c.GetProperty("uv").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
            double[] expected = c.GetProperty("expected").EnumerateArray().Select(v => v.GetDouble()).ToArray();

            using Tensor sampled = Pixal3DProjection.Sample(rows, channels, h, w, uv);

            Assert.Equal(new TensorShape(1, uv.Length / 2, channels), sampled.Shape);
            float[] actual = sampled.AsReadOnlySpan<float>().ToArray();
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], actual[i], 1e-5);
        }
    }

    [Fact]
    public void TheGridCornerFacingTheLeftEdgeLandsOnTheImagesLeftEdge()
    {
        // Cell (x = 0, y = mid, z = mid) of an odd grid sits at cube x = -0.5 / meshScale on the centre plane: pixel 0 of the image.
        foreach ((double fov, double scale, int size) in new[] { (0.8575560450553894, 1.0, 512), (0.4, 1.3, 1024) })
        {
            Pixal3DView view = new(fov, scale);
            float[] uv = Pixal3DProjection.CellCoordinates([0, 0, 8, 8], 17, view, size);

            Assert.Equal(1.0 / size - 1.0, uv[0], 1e-6);
            Assert.Equal(1.0 / size, uv[1], 1e-6);
        }
    }

    [Fact]
    public void ImageYGrowsDownwardAsTheGridsSecondAxisDescends()
    {
        // Grid y maps to the cube's z (up); a larger y is higher in the image, so a smaller image row.
        float[] uv = Pixal3DProjection.CellCoordinates([0, 8, 2, 8, 0, 8, 12, 8], 17, Pixal3DView.Default, 512);

        Assert.True(uv[1] > uv[3]);
        Assert.Equal(uv[0], uv[2], 1e-6);
    }

    [Fact]
    public void SparseCoordinatesEqualTheDenseOnesAtTheSameCells()
    {
        Pixal3DView view = new(0.7, 1.1);
        float[] dense = Pixal3DProjection.DenseCoordinates(6, view, 512);
        int[] coords = [0, 5, 0, 3, 0, 2, 4, 1, 0, 0, 0, 0];

        float[] sparse = Pixal3DProjection.CellCoordinates(coords, 6, view, 512);

        for (int i = 0; i < 3; i++)
        {
            int cell = (coords[i * 4 + 1] * 6 + coords[i * 4 + 2]) * 6 + coords[i * 4 + 3];
            Assert.Equal(dense[cell * 2], sparse[i * 2]);
            Assert.Equal(dense[cell * 2 + 1], sparse[i * 2 + 1]);
        }
    }

    [Fact]
    public void BorderSamplingClampsInsteadOfFadingToZero()
    {
        float[] rows = [1f, 2f, 3f, 4f];

        Span<float> corner = stackalloc float[1], far = stackalloc float[1];
        GridSampler.GridSampleRowsBorder(rows, 1, 2, 2, -3f, -3f, corner);
        GridSampler.GridSampleRowsBorder(rows, 1, 2, 2, 5f, 5f, far);

        Assert.Equal(1f, corner[0]);
        Assert.Equal(4f, far[0]);
    }

    [Theory]
    [InlineData(0.0, 1.0)]
    [InlineData(3.2, 1.0)]
    [InlineData(0.8, 0.0)]
    public void InvalidViewsAreRejected(double fov, double scale)
        => Assert.Throws<ArgumentOutOfRangeException>(() => Pixal3DProjection.DenseCoordinates(4, new Pixal3DView(fov, scale), 512));

    [Fact]
    public void CellsOutsideTheGridAreRejected()
        => Assert.Throws<ArgumentOutOfRangeException>(() => Pixal3DProjection.CellCoordinates([0, 4, 0, 0], 4, Pixal3DView.Default, 512));
}

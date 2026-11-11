using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Paint;
using HartsyInference.ThreeD.Tests.MeshPostProcess;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Unit lane, synthetic meshes: the Paint camera, screen-space rasterizer, geometry maps, view selection and back-projection bake.</summary>
public sealed class PaintGeometryTests
{
    private static Mesh Plane() => new()
    {
        // Original frame: the XY plane at z = 0 facing +z; PaintMeshFrame turns it into the plane y = 0 that the front view (elevation 0, azimuth 0) looks at.
        Vertices = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0],
        Indices = [0, 1, 2, 0, 2, 3],
        Uvs = [0, 0, 1, 0, 1, 1, 0, 1],
    };

    [Fact]
    public void WorldToCamera_PutsTheCameraAtTheOriginLookingDownMinusZ()
    {
        double[] m = PaintCamera.WorldToCamera(30, 40);
        double elev = -30 * Math.PI / 180, azim = 130 * Math.PI / 180;
        double d = PaintCamera.Distance;
        (double x, double y, double z, _) = PaintCamera.Transform(m, d * Math.Cos(elev) * Math.Cos(azim), d * Math.Cos(elev) * Math.Sin(azim), d * Math.Sin(elev));
        Assert.Equal(0, x, 9);
        Assert.Equal(0, y, 9);
        Assert.Equal(0, z, 9);
        (_, _, double zOrigin, _) = PaintCamera.Transform(m, 0, 0, 0);
        Assert.Equal(-d, zOrigin, 9);
    }

    [Fact]
    public void WorldToCamera_FrontViewHasWorldUpAsImageUp()
    {
        double[] m = PaintCamera.WorldToCamera(0, 0);
        (_, double y, _, _) = PaintCamera.Transform(m, 0, 0, 1);
        Assert.Equal(1, y, 9);
    }

    [Fact]
    public void Rasterizer_NearerTriangleWinsRegardlessOfWindingOrOrder()
    {
        // Two overlapping triangles: ndc z 0.5 listed first, -0.5 second (nearer, opposite winding).
        float[] clip = [-1, -1, 0.5f, 1, 1, -1, 0.5f, 1, -1, 1, 0.5f, 1, -1, -1, -0.5f, 1, -1, 1, -0.5f, 1, 1, -1, -0.5f, 1];
        ViewRaster raster = ViewRasterizer.Rasterize(clip, [0, 1, 2, 3, 4, 5], 16, 16);

        Assert.Equal(2, raster.FaceIds[3 * 16 + 3]);
        Assert.Equal(0, raster.FaceIds[15 * 16 + 15]);
        int p = 3 * 16 + 3;
        Assert.Equal(1f, raster.Barycentric[p * 3] + raster.Barycentric[p * 3 + 1] + raster.Barycentric[p * 3 + 2], 5);
    }

    [Fact]
    public void Rasterizer_IsDeterministicAcrossRuns()
    {
        PaintViewRenderer renderer = new(PaintMeshFrame.Create(SyntheticMeshes.Icosphere(3)));

        int[] a = renderer.Rasterize(20, 30, 128).FaceIds;
        int[] b = renderer.Rasterize(20, 30, 128).FaceIds;

        Assert.Equal(a, b);
        Assert.Contains(a, id => id > 0);
    }

    [Fact]
    public void MeshFrame_CentresScalesAndSwapsAxes()
    {
        Mesh mesh = new() { Vertices = [10, 0, 0, 12, 0, 0, 10, 2, 0], Indices = [0, 1, 2] };

        PaintMeshFrame frame = PaintMeshFrame.Create(mesh);

        double maxDistance = 0;
        for (int i = 0; i < 3; i++)
        {
            double x = frame.Vertices[i * 3], y = frame.Vertices[i * 3 + 1], z = frame.Vertices[i * 3 + 2];
            maxDistance = Math.Max(maxDistance, Math.Sqrt(x * x + y * y + z * z));
        }
        // The farthest vertex from the box centre sits at half the scale factor; original +x runs along frame -x.
        Assert.Equal(1.15 / 2, maxDistance, 5);
        Assert.True(frame.Vertices[3] < frame.Vertices[0]);
    }

    [Fact]
    public void NormalAndPositionMaps_OfAPlaneFacingTheCameraHaveTheExpectedColours()
    {
        PaintViewRenderer renderer = new(PaintMeshFrame.Create(Plane()));

        byte[] normal = renderer.RenderNormalMap(0, 0, 64);
        byte[] position = renderer.RenderPositionMap(0, 0, 64);

        int center = (32 * 64 + 32) * 3;
        // Plane normal (original +z) becomes frame -y, so the colour is (0.5, 0, 0.5); the plane passes through the origin, so position encodes 0.5.
        Assert.Equal(127, normal[center]);
        Assert.Equal(0, normal[center + 1]);
        Assert.Equal(127, normal[center + 2]);
        Assert.All(new[] { position[center], position[center + 1], position[center + 2] }, v => Assert.InRange((int)v, 122, 132));
        Assert.Equal(255, normal[0]);
        Assert.Equal(255, position[2]);
    }

    [Fact]
    public void ViewSelector_KeepsTheSixFixedViewsOnAConvexBodyAndValidatesTheLimit()
    {
        PaintViewRenderer renderer = new(PaintMeshFrame.Create(SyntheticMeshes.Icosphere(2)));

        IReadOnlyList<PaintViewCandidate> chosen = PaintViewSelector.Select(renderer, PaintViewCandidate.All, 12, 128);

        Assert.Equal(PaintViewCandidate.Fixed, chosen);
        Assert.Throws<ArgumentOutOfRangeException>(() => PaintViewSelector.Select(renderer, PaintViewCandidate.All, 5, 64));
    }

    [Fact]
    public void ViewSelector_NeverExceedsTheMaximumAndStartsWithTheFixedViews()
    {
        PaintViewRenderer renderer = new(PaintMeshFrame.Create(SyntheticMeshes.Torus(24, 12)));

        IReadOnlyList<PaintViewCandidate> chosen = PaintViewSelector.Select(renderer, PaintViewCandidate.All, 8, 128);

        Assert.InRange(chosen.Count, 6, 8);
        Assert.Equal(PaintViewCandidate.Fixed, chosen.Take(6).ToList());
    }

    [Fact]
    public void BackProjection_OfThePositionMapRecoversEachTexelsOwnPosition()
    {
        Mesh plane = Plane();
        PaintMeshFrame frame = PaintMeshFrame.Create(plane);
        PaintViewRenderer renderer = new(frame);
        const int resolution = 256, size = 32;
        byte[] map = renderer.RenderPositionMap(0, 0, resolution);
        float[] image = new float[map.Length];
        for (int i = 0; i < map.Length; i++)
            image[i] = map[i] / 255f;
        PaintTexelSurface surface = PaintTexelSurface.Create(frame, plane.Uvs!, size, size);

        PaintViewBake bake = PaintBackProjector.Project(renderer, surface, image, resolution, 3, 0, 0);

        int painted = 0;
        for (int i = 0; i < surface.Count; i++)
        {
            if (bake.Weights[i] <= 0f)
                continue;
            painted++;
            Assert.Equal(1f, bake.Weights[i], 4);
            for (int c = 0; c < 3; c++)
                Assert.InRange(Math.Abs(0.5 - surface.Positions[i * 3 + c] / PaintMeshFrame.DefaultScaleFactor - bake.Colors[i * 3 + c]), 0.0, 0.02);
        }
        Assert.True(painted > surface.Count / 2, $"only {painted} of {surface.Count} texels were painted");
    }

    [Fact]
    public void BackProjection_IgnoresAViewThatSeesTheSurfaceEdgeOn()
    {
        Mesh plane = Plane();
        PaintMeshFrame frame = PaintMeshFrame.Create(plane);
        PaintViewRenderer renderer = new(frame);
        PaintTexelSurface surface = PaintTexelSurface.Create(frame, plane.Uvs!, 16, 16);

        PaintViewBake bake = PaintBackProjector.Project(renderer, surface, new float[64 * 64 * 3], 64, 3, 0, 90);

        Assert.All(bake.Weights, w => Assert.Equal(0f, w));
    }

    [Fact]
    public void Merger_WeightsByCosinePowerAndSkipsRedundantViews()
    {
        Mesh plane = Plane();
        PaintTexelSurface surface = PaintTexelSurface.Create(PaintMeshFrame.Create(plane), plane.Uvs!, 4, 4);
        PaintViewBake a = new(surface.Count, 1), b = new(surface.Count, 1);
        for (int i = 0; i < surface.Count; i++)
        {
            a.Colors[i] = 1f;
            a.Weights[i] = 1f;
            b.Weights[i] = 0.5f;
        }

        PaintTextureMerger skipping = new(surface, 1, new PaintBakeOptions { CosineExponent = 2.0 });
        Assert.True(skipping.Add(a, 1.0));
        Assert.False(skipping.Add(b, 1.0));
        Assert.Equal(1f, skipping.Finish().Texture[surface.Texels[0]], 5);

        PaintTextureMerger blending = new(surface, 1, new PaintBakeOptions { CosineExponent = 2.0, SkipOverlap = 2.0 });
        Assert.True(blending.Add(a, 1.0));
        Assert.True(blending.Add(b, 1.0));
        (float[] mixed, bool[] painted) = blending.Finish();
        Assert.Equal(1.0 / (1.0 + 0.25), mixed[surface.Texels[0]], 5);
        Assert.True(painted[surface.Texels[0]]);
    }

    [Fact]
    public void ImageFilters_DetectAStepEdgeAndDilateBySquareRadius()
    {
        byte[] step = new byte[16 * 16];
        for (int y = 0; y < 16; y++)
            for (int x = 8; x < 16; x++)
                step[y * 16 + x] = 200;
        bool[] one = new bool[9 * 9];
        one[4 * 9 + 4] = true;

        bool[] edges = PaintImageFilters.Edges(step, 16, 16, 30, 80);
        bool[] grown = PaintImageFilters.Dilate(one, 9, 9, 2);

        Assert.True(edges[8 * 16 + 7] || edges[8 * 16 + 8]);
        Assert.False(edges[8 * 16 + 2]);
        Assert.Equal(25, grown.Count(v => v));
        Assert.True(grown[2 * 9 + 2] && grown[6 * 9 + 6] && !grown[1 * 9 + 4]);
    }
}

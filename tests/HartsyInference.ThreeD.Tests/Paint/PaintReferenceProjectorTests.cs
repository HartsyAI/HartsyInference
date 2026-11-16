using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Paint;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Unit lane, synthetic meshes and pictures: fitting the reference picture to the front view and laying it over the generated view.</summary>
public sealed class PaintReferenceProjectorTests
{
    // The XY square at z = 0, facing +z: its front view is the upright square, original +x to the right.
    private static Mesh Plane() => new()
    {
        Vertices = [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0],
        Indices = [0, 1, 2, 0, 2, 3],
    };

    // A picture of side `size` whose subject is the square [from, to) on both axes, coloured by quadrant: red top left, green top right, blue bottom left, white bottom right.
    private static (byte[] Rgb, byte[] Alpha) Quadrants(int size, int from, int to)
    {
        byte[] rgb = new byte[size * size * 3], alpha = new byte[size * size];
        int middle = (from + to) / 2;
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                int p = y * size + x;
                if (x < from || x >= to || y < from || y >= to)
                {
                    rgb[p * 3] = rgb[p * 3 + 1] = rgb[p * 3 + 2] = 255;
                    continue;
                }
                alpha[p] = 255;
                (byte r, byte g, byte b) = (y < middle, x < middle) switch
                {
                    (true, true) => ((byte)255, (byte)0, (byte)0),
                    (true, false) => ((byte)0, (byte)255, (byte)0),
                    (false, true) => ((byte)0, (byte)0, (byte)255),
                    _ => ((byte)255, (byte)255, (byte)255),
                };
                (rgb[p * 3], rgb[p * 3 + 1], rgb[p * 3 + 2]) = (r, g, b);
            }
        return (rgb, alpha);
    }

    private static (float R, float G, float B) At(PaintReferenceProjection projection, double u, double v)
    {
        int p = (int)(v * projection.Resolution) * projection.Resolution + (int)(u * projection.Resolution);
        Assert.True(projection.Covered[p], $"pixel at ({u}, {v}) is not covered");
        return (projection.Image[p * 3], projection.Image[p * 3 + 1], projection.Image[p * 3 + 2]);
    }

    [Fact]
    public void Project_FitsTheSubjectToTheSilhouetteUprightAndUnmirrored()
    {
        PaintViewRenderer renderer = new(PaintMeshFrame.Create(Plane()));
        (byte[] rgb, byte[] alpha) = Quadrants(300, 60, 240);

        PaintReferenceProjection projection = PaintReferenceProjector.Project(renderer, rgb, alpha, 300, 300, 256)!;

        Assert.True(projection.Fit > 0.95, $"fit {projection.Fit}");
        // The square fills the view's middle; probe a point well inside each quadrant.
        PaintViewRenderer probe = renderer;
        int[] faces = probe.Rasterize(0, 0, 256).FaceIds;
        int first = Array.FindIndex(faces, f => f > 0), last = Array.FindLastIndex(faces, f => f > 0);
        double top = first / 256 / 256.0, bottom = last / 256 / 256.0, left = first % 256 / 256.0, right = last % 256 / 256.0;
        double q1 = 0.25, q3 = 0.75;
        Assert.Equal((1f, 0f, 0f), At(projection, left + (right - left) * q1, top + (bottom - top) * q1));
        Assert.Equal((0f, 1f, 0f), At(projection, left + (right - left) * q3, top + (bottom - top) * q1));
        Assert.Equal((0f, 0f, 1f), At(projection, left + (right - left) * q1, top + (bottom - top) * q3));
    }

    [Fact]
    public void Project_ReportsAPoorFitWhenTheSubjectDoesNotMatchTheSilhouette()
    {
        PaintViewRenderer renderer = new(PaintMeshFrame.Create(Plane()));
        // A tall thin bar against a square mesh.
        int size = 300;
        byte[] rgb = new byte[size * size * 3], alpha = new byte[size * size];
        for (int y = 20; y < 280; y++)
            for (int x = 140; x < 160; x++)
                alpha[y * size + x] = 255;

        PaintReferenceProjection projection = PaintReferenceProjector.Project(renderer, rgb, alpha, size, size, 128)!;

        Assert.True(projection.Fit < PaintReferenceProjector.MinimumFit, $"fit {projection.Fit}");
    }

    [Fact]
    public void Project_ReturnsNullWithoutASubject()
    {
        PaintViewRenderer renderer = new(PaintMeshFrame.Create(Plane()));

        Assert.Null(PaintReferenceProjector.Project(renderer, new byte[64 * 64 * 3], new byte[64 * 64], 64, 64, 64));
    }

    // A quad from (x0, y0) to (x1, y1) at depth z, facing +z, with UVs over [u0, u1] x [0, 1], appended to the given lists.
    private static void Quad(List<float> vertices, List<int> indices, List<float> uvs, float x0, float y0, float x1, float y1, float z, float u0, float u1)
    {
        int b = vertices.Count / 3;
        vertices.AddRange([x0, y0, z, x1, y0, z, x1, y1, z, x0, y1, z]);
        uvs.AddRange([u0, 1, u1, 1, u1, 0, u0, 0]);
        indices.AddRange([b, b + 1, b + 2, b, b + 2, b + 3]);
    }

    private sealed record Scene(PaintViewRenderer Renderer, PaintTexelSurface Surface, PaintReferenceProjection Placement, byte[] Rgb, byte[] Alpha, int Size);

    // A scene whose front silhouette is the square [-1, 1]^2 (the subject's outline), so the placement fit is exact; the picture is Quadrants(300, 60, 240).
    private static Scene Build(Action<List<float>, List<int>, List<float>> extra)
    {
        List<float> vertices = [], uvs = [];
        List<int> indices = [];
        Quad(vertices, indices, uvs, -1, -1, 1, 1, 0, 0, 0.5f);
        extra(vertices, indices, uvs);
        Mesh mesh = new() { Vertices = [.. vertices], Indices = [.. indices], Uvs = [.. uvs] };
        PaintMeshFrame frame = PaintMeshFrame.Create(mesh);
        PaintViewRenderer renderer = new(frame);
        PaintTexelSurface surface = PaintTexelSurface.Create(frame, mesh.Uvs!, 128, 128);
        (byte[] rgb, byte[] alpha) = Quadrants(300, 60, 240);
        PaintReferenceProjection placement = PaintReferenceProjector.Project(renderer, rgb, alpha, 300, 300, 256)!;
        return new Scene(renderer, surface, placement, rgb, alpha, 128);
    }

    private static int Apply(Scene scene, float[] texture, bool[] known) =>
        PaintReferenceProjector.ApplyToTexels(scene.Placement, scene.Renderer, scene.Surface, texture, known, scene.Rgb, scene.Alpha, 300, 300, 3e-3);

    [Fact]
    public void ApplyToTexels_ReplacesThePaintedColourWithThePicturesOwnOnAFrontFacingSurface()
    {
        Scene scene = Build((_, _, _) => { });
        float[] texture = Enumerable.Repeat(0.5f, scene.Size * scene.Size * 3).ToArray();
        bool[] known = Enumerable.Repeat(true, scene.Size * scene.Size).ToArray();

        int written = Apply(scene, texture, known);

        Assert.True(written > scene.Surface.Count / 2, $"{written} of {scene.Surface.Count}");
        // Away from the quadrant borders every texel is one of the picture's four colours, with nothing of the painted grey mixed in.
        int[] pure = new int[4];
        (float, float, float)[] colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)];
        foreach (int texel in scene.Surface.Texels)
        {
            (float r, float g, float b) = (texture[texel * 3], texture[texel * 3 + 1], texture[texel * 3 + 2]);
            for (int k = 0; k < 4; k++)
                if (Math.Abs(r - colours[k].Item1) < 1e-3 && Math.Abs(g - colours[k].Item2) < 1e-3 && Math.Abs(b - colours[k].Item3) < 1e-3)
                    pure[k]++;
        }
        Assert.All(pure, count => Assert.True(count > scene.Surface.Count / 8, string.Join(", ", pure)));
    }

    [Fact]
    public void ApplyToTexels_LeavesSurfacesHiddenBehindANearerOneAlone()
    {
        // A smaller quad in front of the square's centre, on the other half of the atlas.
        Scene scene = Build((vertices, indices, uvs) => Quad(vertices, indices, uvs, -0.4f, -0.4f, 0.4f, 0.4f, 0.5f, 0.5f, 1));
        float[] texture = new float[scene.Size * scene.Size * 3];
        bool[] known = new bool[scene.Size * scene.Size];

        Apply(scene, texture, known);

        // The back square's texels behind the front quad (UV centre of its half) took nothing; its corners did.
        int behind = scene.Size / 2 * scene.Size + scene.Size / 4, corner = 4 * scene.Size + 4;
        Assert.False(known[behind]);
        Assert.True(known[corner]);
        Assert.True(known[scene.Size / 2 * scene.Size + scene.Size * 3 / 4]);
    }

    [Fact]
    public void ApplyToTexels_GivesANearerPartTheDrawnEdgeWhereTheMeshPutItLowerThanDrawn()
    {
        // A brim over a face: the picture's subject is the square [60, 240) with the brim colour above row 140 and the face colour below; the mesh's
        // brim, in front of the face, reaches down to row 144.
        const int size = 300, atlas = 256;
        byte[] rgb = new byte[size * size * 3], alpha = new byte[size * size];
        (byte R, byte G, byte B) brim = (60, 50, 110), face = (230, 180, 150);
        for (int y = 60; y < 240; y++)
            for (int x = 60; x < 240; x++)
            {
                int p = y * size + x;
                alpha[p] = 255;
                (rgb[p * 3], rgb[p * 3 + 1], rgb[p * 3 + 2]) = y < 140 ? brim : face;
            }
        List<float> vertices = [], uvs = [];
        List<int> indices = [];
        float edge = 1 - (144 - 60) / 180f * 2;
        Quad(vertices, indices, uvs, -1, -1, 1, 1, 0, 0, 0.5f);
        Quad(vertices, indices, uvs, -1, edge, 1, 1, 0.3f, 0.5f, 1);
        Mesh mesh = new() { Vertices = [.. vertices], Indices = [.. indices], Uvs = [.. uvs] };
        PaintMeshFrame frame = PaintMeshFrame.Create(mesh);
        PaintViewRenderer renderer = new(frame);
        PaintTexelSurface surface = PaintTexelSurface.Create(frame, mesh.Uvs!, atlas, atlas);
        PaintReferenceProjection placement = PaintReferenceProjector.Project(renderer, rgb, alpha, size, size, 256)!;
        float[] texture = new float[atlas * atlas * 3];
        bool[] known = new bool[atlas * atlas];

        PaintReferenceProjector.ApplyToTexels(placement, renderer, surface, texture, known, rgb, alpha, size, size, 3e-3);

        // The brim's texels just above its lower edge (drawn as face) take the brim colour; the face below the brim keeps the face colour.
        int rim = (int)(atlas * 0.985) * atlas + atlas * 3 / 4, chin = atlas * 3 / 4 * atlas + atlas / 4;
        Assert.True(known[rim] && known[chin]);
        Assert.Equal(brim.R / 255f, texture[rim * 3], 0.08f);
        Assert.Equal(brim.B / 255f, texture[rim * 3 + 2], 0.08f);
        Assert.Equal(face.R / 255f, texture[chin * 3], 0.02f);
    }

    [Fact]
    public void ApplyToTexels_SkipsSurfacesTurnedAwayFromTheFront()
    {
        // A strip turned 85 degrees about the vertical axis beside the square: the front picture would be a streak on it.
        Scene scene = Build((vertices, indices, uvs) =>
        {
            int b = vertices.Count / 3;
            float dx = MathF.Cos(85f * MathF.PI / 180f) * 0.8f, dz = MathF.Sin(85f * MathF.PI / 180f) * 0.8f;
            vertices.AddRange([1.05f, -1, 0, 1.05f + dx, -1, -dz, 1.05f + dx, 1, -dz, 1.05f, 1, 0]);
            uvs.AddRange([0.5f, 1, 1, 1, 1, 0, 0.5f, 0]);
            indices.AddRange([b, b + 1, b + 2, b, b + 2, b + 3]);
        });
        float[] texture = new float[scene.Size * scene.Size * 3];
        bool[] known = new bool[scene.Size * scene.Size];

        Apply(scene, texture, known);

        for (int y = 8; y < scene.Size - 8; y++)
            for (int x = scene.Size / 2 + 8; x < scene.Size - 8; x++)
                Assert.False(known[y * scene.Size + x], $"texel ({x}, {y}) of the turned strip took the picture");
    }
}

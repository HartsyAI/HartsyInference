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

    [Fact]
    public void Overlay_UsesThePictureWhereItAgreesAndKeepsTheViewWhereItDoesNot()
    {
        int size = 64;
        float[] image = new float[size * size * 3];
        bool[] covered = new bool[size * size];
        for (int p = 0; p < covered.Length; p++)
        {
            covered[p] = true;
            // Left half of the picture is light skin, right half too.
            (image[p * 3], image[p * 3 + 1], image[p * 3 + 2]) = (0.9f, 0.75f, 0.65f);
        }
        float[] view = new float[size * size * 3];
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                int p = y * size + x;
                // The generated view has slightly darker skin on the left and a blue brim on the right.
                (view[p * 3], view[p * 3 + 1], view[p * 3 + 2]) = x < size / 2 ? (0.8f, 0.65f, 0.55f) : (0.15f, 0.15f, 0.5f);
            }
        PaintReferenceProjection projection = new(image, covered, size, 1, 0, 1, 0, 0);

        PaintReferenceProjector.Overlay(projection, view);

        int left = 32 * size + 4, right = 32 * size + 60;
        Assert.Equal(0.9f, view[left * 3], 4);
        Assert.Equal(0.15f, view[right * 3], 4);
    }

    [Fact]
    public void Align_MovesThePictureOntoTheGeneratedViewByASmallShift()
    {
        // A picture with a dark band, and a generated view with the same band 4 px lower (a brim the mesh brings lower than drawn).
        int size = 128;
        float[] picture = new float[size * size * 3], view = new float[size * size * 3];
        bool[] covered = new bool[size * size];
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                int p = y * size + x;
                covered[p] = true;
                float pictureValue = y >= 50 && y < 60 ? 0.1f : 0.9f, viewValue = y >= 54 && y < 64 ? 0.1f : 0.9f;
                for (int c = 0; c < 3; c++)
                {
                    picture[p * 3 + c] = pictureValue;
                    view[p * 3 + c] = viewValue;
                }
            }
        PaintReferenceProjection projection = new(picture, covered, size, 1, 0, 1, 0, 0);

        (float[] aligned, bool[] alignedCovered) = PaintReferenceProjector.Align(projection, view);

        int inside = 60 * size + 64, above = 52 * size + 64;
        Assert.True(alignedCovered[inside]);
        Assert.Equal(0.1f, aligned[inside * 3], 2);
        Assert.Equal(0.9f, aligned[above * 3], 2);
    }

    [Fact]
    public void Overlay_LeavesUncoveredPixelsAlone()
    {
        int size = 16;
        PaintReferenceProjection projection = new(Enumerable.Repeat(1f, size * size * 3).ToArray(), new bool[size * size], size, 1, 0, 1, 0, 0);
        float[] view = Enumerable.Repeat(0.98f, size * size * 3).ToArray();

        Assert.Equal(0, PaintReferenceProjector.Overlay(projection, view));
        Assert.All(view, value => Assert.Equal(0.98f, value));
    }
}

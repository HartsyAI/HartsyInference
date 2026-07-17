using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Face;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Unit tests for the ArcFace 5-point/3-point similarity alignment math (pure geometry, no weights).</summary>
public sealed class FaceAlignmentTests
{
    [Fact]
    public void EstimateSimilarity_RecoversKnownTransform()
    {
        // Apply a known similarity (scale 2.5, rotation 30°, translation (12, -7)) to the template
        // points, then estimate the inverse mapping back — must recover the template exactly.
        float s = 2.5f;
        float cos = s * MathF.Cos(MathF.PI / 6f);
        float sin = s * MathF.Sin(MathF.PI / 6f);
        float[] src = new float[6];
        float[] dst = new float[6];
        for (int i = 0; i < 3; i++)
        {
            float tx = FaceAlignment.Template[i, 0];
            float ty = FaceAlignment.Template[i, 1];
            src[2 * i] = cos * tx - sin * ty + 12f;
            src[2 * i + 1] = sin * tx + cos * ty - 7f;
            dst[2 * i] = tx;
            dst[2 * i + 1] = ty;
        }

        FaceAlignment.Affine2x3 m = FaceAlignment.EstimateSimilarity(src, dst);
        for (int i = 0; i < 3; i++)
        {
            (float x, float y) = m.Apply(src[2 * i], src[2 * i + 1]);
            Assert.True(MathF.Abs(x - dst[2 * i]) < 1e-3f, $"point {i} x: {x} vs {dst[2 * i]}");
            Assert.True(MathF.Abs(y - dst[2 * i + 1]) < 1e-3f, $"point {i} y: {y} vs {dst[2 * i + 1]}");
        }

        // The recovered linear part must be the inverse rotation+scale: a = cos(-30°)/2.5, b = sin(-30°)/2.5.
        float invS = 1f / s;
        float expA = invS * MathF.Cos(-MathF.PI / 6f);
        float expC = invS * MathF.Sin(-MathF.PI / 6f);
        Assert.True(MathF.Abs(m.A - expA) < 1e-4f, $"A={m.A} exp={expA}");
        Assert.True(MathF.Abs(m.C - expC) < 1e-4f, $"C={m.C} exp={expC}");
        Assert.True(MathF.Abs(m.A - m.D) < 1e-5f, "similarity must have A == D");
        Assert.True(MathF.Abs(m.B + m.C) < 1e-5f, "similarity must have B == -C");
    }

    [Fact]
    public void EstimateSimilarity_LeastSquaresOverdetermined()
    {
        // 5 noisy correspondences of the identity: LSQ solution must stay near identity.
        float[] src = [10f, 10f, 90f, 10f, 50f, 50f, 30f, 90f, 70f, 90f];
        float[] dst = [10.4f, 9.7f, 89.6f, 10.2f, 50.1f, 50.3f, 29.8f, 89.9f, 70.2f, 90.1f];
        FaceAlignment.Affine2x3 m = FaceAlignment.EstimateSimilarity(src, dst);
        Assert.True(MathF.Abs(m.A - 1f) < 0.02f);
        Assert.True(MathF.Abs(m.B) < 0.02f);
        Assert.True(MathF.Abs(m.Tx) < 2f);
        Assert.True(MathF.Abs(m.Ty) < 2f);
    }

    [Fact]
    public void WarpAffine_IdentityPreservesPixels()
    {
        const int w = 8, h = 6;
        byte[] rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = (byte)(i * 7 % 256);
        FaceAlignment.Affine2x3 identity = new(1f, 0f, 0f, 0f, 1f, 0f);
        byte[] warped = FaceAlignment.WarpAffine(rgb, w, h, identity, w, h);
        Assert.Equal(rgb, warped);
    }

    [Fact]
    public void WarpAffine_TranslationShiftsAndPadsBlack()
    {
        const int w = 4, h = 4;
        byte[] rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = 200;
        // Shift content right by 2: dst(x,y) = src(x-2,y) → left 2 columns sample in-bounds only for x>=2.
        FaceAlignment.Affine2x3 shift = new(1f, 0f, 2f, 0f, 1f, 0f);
        byte[] warped = FaceAlignment.WarpAffine(rgb, w, h, shift, w, h);
        Assert.Equal(0, warped[0]);                    // x=0 samples src x=-2 → black
        Assert.Equal(200, warped[(0 * w + 3) * 3]);    // x=3 samples src x=1 → in bounds
    }

    [Fact]
    public void AlignToTemplate_MapsKeypointsOntoTemplate()
    {
        // Build a synthetic 200×200 image with distinct colors at three keypoint locations laid out like a
        // face (eyes above nose). After alignment the template positions must hold those colors.
        const int size = 200;
        byte[] rgb = new byte[size * size * 3];
        float[] pts = [60f, 80f, 140f, 80f, 100f, 120f];
        byte[][] colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]];
        for (int p = 0; p < 3; p++)
        {
            int cx = (int)pts[2 * p], cy = (int)pts[2 * p + 1];
            for (int dy = -6; dy <= 6; dy++)
                for (int dx = -6; dx <= 6; dx++)
                {
                    int off = ((cy + dy) * size + cx + dx) * 3;
                    rgb[off] = colors[p][0]; rgb[off + 1] = colors[p][1]; rgb[off + 2] = colors[p][2];
                }
        }

        byte[] aligned = FaceAlignment.AlignToTemplate(rgb, size, size, pts);
        Assert.Equal(FaceAlignment.CropSize * FaceAlignment.CropSize * 3, aligned.Length);
        for (int p = 0; p < 3; p++)
        {
            int tx = (int)MathF.Round(FaceAlignment.Template[p, 0]);
            int ty = (int)MathF.Round(FaceAlignment.Template[p, 1]);
            int off = (ty * FaceAlignment.CropSize + tx) * 3;
            Assert.True(aligned[off + p] > 200,
                $"template point {p} at ({tx},{ty}) expected channel {p} bright, got ({aligned[off]},{aligned[off + 1]},{aligned[off + 2]})");
        }
    }

    [Fact]
    public void TryGetAlignmentPoints_OrdersEyesByImageX()
    {
        // COCO order: 0 nose, 1 left-eye, 2 right-eye. Give "left eye" the LARGER x (mirrored image) —
        // the extractor must still emit the smaller-x eye first to match the template's image-left slot.
        List<Keypoint> kpts = [];
        for (int i = 0; i < 17; i++) kpts.Add(new Keypoint(0, 0, 0));
        kpts[0] = new Keypoint(100f, 120f, 0.9f);
        kpts[1] = new Keypoint(140f, 80f, 0.9f);
        kpts[2] = new Keypoint(60f, 80f, 0.9f);
        PoseDetection person = new(new YoloDetection(0, 0, 200, 200, 0.9f, 0), kpts);

        Assert.True(FaceAlignment.TryGetAlignmentPoints(person, 0.3f, out float[] pts));
        Assert.Equal(60f, pts[0]);
        Assert.Equal(140f, pts[2]);
        Assert.Equal(100f, pts[4]);

        // Below-threshold nose → no alignment points.
        kpts[0] = new Keypoint(100f, 120f, 0.1f);
        PoseDetection lowConf = new(new YoloDetection(0, 0, 200, 200, 0.9f, 0), kpts);
        Assert.False(FaceAlignment.TryGetAlignmentPoints(lowConf, 0.3f, out _));
    }
}

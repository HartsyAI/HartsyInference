using HartsyInference.Vision.Moge;

namespace HartsyInference.Vision.Tests.Moge;

public sealed class MogeFocalShiftTests
{
    // A planar-ish scene seen by a pinhole camera of known focal length: raw points are the true camera-space points with the z axis shifted back by the unknown shift.
    private static (float[] Points, bool[] Valid) Scene(int height, int width, double focal, double shift, Func<double, double, double> depthAt)
    {
        float[] u = new float[height * width], v = new float[height * width];
        MogeHead.ViewPlaneUv(u, v, height, width, (double)width / height);
        float[] points = new float[3 * height * width];
        bool[] valid = new bool[height * width];
        for (int i = 0; i < height * width; i++)
        {
            double z = depthAt(u[i], v[i]);
            // uv = focal * xy / z, so xy = uv * z / focal.
            points[i] = (float)(u[i] * z / focal);
            points[height * width + i] = (float)(v[i] * z / focal);
            points[2 * height * width + i] = (float)(z - shift);
            valid[i] = true;
        }

        return (points, valid);
    }

    [Fact]
    public void Recover_FindsTheShiftAndFocalOfAPinholeScene()
    {
        (float[] points, bool[] valid) = Scene(48, 64, focal: 1.3, shift: 0.7, (u, v) => 2.0 + 0.5 * u - 0.3 * v + 0.2 * Math.Sin(7 * u));

        (double focal, double shift) = MogeFocalShift.Recover(points, valid, 48, 64);

        Assert.Equal(1.3, focal, 3);
        Assert.Equal(0.7, shift, 3);
    }

    [Fact]
    public void Recover_WithAKnownFocalSolvesOnlyTheShift()
    {
        (float[] points, bool[] valid) = Scene(48, 64, focal: 1.3, shift: -0.4, (u, v) => 3.0 + u + 0.4 * v * v);

        (double focal, double shift) = MogeFocalShift.Recover(points, valid, 48, 64, focal: 1.3);

        Assert.Equal(1.3, focal);
        Assert.Equal(-0.4, shift, 3);
    }

    [Fact]
    public void Recover_IgnoresMaskedPointsAndFallsBackWhenTooFewAreValid()
    {
        (float[] points, bool[] valid) = Scene(48, 64, focal: 0.9, shift: 0.25, (u, v) => 2.5 + 0.3 * u);
        // Corrupt half of the image and mask it out: the answer must not move.
        for (int i = 0; i < valid.Length / 2; i++)
        {
            points[2 * valid.Length + i] = 99f;
            points[i] = -50f;
            valid[i] = false;
        }

        (double focal, double shift) = MogeFocalShift.Recover(points, valid, 48, 64);
        Assert.Equal(0.9, focal, 3);
        Assert.Equal(0.25, shift, 3);

        bool[] none = new bool[valid.Length];
        Assert.Equal((1.0, 0.0), MogeFocalShift.Recover(points, none, 48, 64));
    }
}

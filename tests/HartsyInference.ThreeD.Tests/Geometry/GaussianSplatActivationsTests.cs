using System.Text;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Geometry;

public sealed class GaussianSplatActivationsTests
{
    [Theory]
    [InlineData(0.004f)]
    [InlineData(0.5f)]
    [InlineData(3f)]
    public void SoftplusInverse_UndoesSoftplus(float x)
    {
        Assert.Equal(x, GaussianSplatActivations.Softplus(GaussianSplatActivations.SoftplusInverse(x)), 5);
    }

    [Fact]
    public void Softplus_ReturnsTheInputAboveTwenty_AsTorchDoes()
    {
        Assert.Equal(25f, GaussianSplatActivations.Softplus(25f));
        Assert.Equal(Math.Log(1.0 + Math.Exp(2.0)), GaussianSplatActivations.Softplus(2f), 5);
    }

    [Fact]
    public void LogScale_IsTheLogOfTheKernelFlooredScale()
    {
        const float bias = 0.004f, kernel = 0.0009f;
        float offset = GaussianSplatActivations.SoftplusInverse(bias);

        // A zero output gives the initial scale, floored by the kernel: sqrt(bias² + kernel²).
        Assert.Equal(Math.Log(Math.Sqrt(bias * (double)bias + kernel * (double)kernel)), GaussianSplatActivations.LogScale(0f, offset, kernel), 4);   // float32 SoftplusInverse of 0.004 is good to about 1e-5
        // A large raw value is the scale itself, as the floor stops mattering.
        Assert.Equal(Math.Log(Math.Sqrt(Math.Pow(Math.Log(1.0 + Math.Exp(1.5 + offset)), 2) + kernel * (double)kernel)),
            GaussianSplatActivations.LogScale(1.5f, offset, kernel), 4);
    }

    [Fact]
    public void TurnToYUp_IsTheLeftProductWithAQuarterTurnAboutX()
    {
        // (w, x, y, z) = identity turns to (sqrt 1/2, sqrt 1/2, 0, 0), the turn itself.
        (float w, float x, float y, float z) = GaussianSplatActivations.TurnToYUp(1f, 0f, 0f, 0f);
        Assert.Equal(Math.Sqrt(0.5), w, 6);
        Assert.Equal(Math.Sqrt(0.5), x, 6);
        Assert.Equal(0f, y);
        Assert.Equal(0f, z);

        // The left product against a hand-expanded Hamilton product, for an orientation that is not about x.
        (float qw, float qx, float qy, float qz) = (0.4f, 0.1f, 0.7f, -0.3f);
        double c = Math.Sqrt(0.5);
        (float rw, float rx, float ry, float rz) = GaussianSplatActivations.TurnToYUp(qw, qx, qy, qz);
        Assert.Equal(c * qw - c * qx, rw, 5);
        Assert.Equal(c * qx + c * qw, rx, 5);
        Assert.Equal(c * qy - c * qz, ry, 5);
        Assert.Equal(c * qz + c * qy, rz, 5);
    }

    /// <summary>The criterion that does not depend on any upstream's code: a splat is an ellipsoid with covariance <c>R·S²·Rᵀ</c>, and turning the scene with the positions <c>(x, y, z) → (x, −z, y)</c>
    /// must give the covariance <c>T·Σ·Tᵀ</c>. Only the left product <c>q_R ⊗ q</c> (matrices <c>T·R</c>) does; the right one (<c>R·T</c>) tilts the ellipsoid about the wrong axes unless <c>q</c> is about x.</summary>
    [Fact]
    public void TheTurnedRotationGivesTheCovarianceOfTheTurnedScene()
    {
        double[,] t = { { 1, 0, 0 }, { 0, 0, -1 }, { 0, 1, 0 } };
        Random rng = new(11);
        double worstLeft = 0, worstRight = 0;
        for (int trial = 0; trial < 50; trial++)
        {
            double[] q = [rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1, rng.NextDouble() * 2 - 1];
            double n = Math.Sqrt(q.Sum(v => v * v));
            q = [.. q.Select(v => v / n)];
            double[] scale = [0.2 + rng.NextDouble(), 0.2 + rng.NextDouble(), 0.2 + rng.NextDouble()];
            double[,] sigma = Covariance(q, scale);
            double[,] want = Multiply(Multiply(t, sigma), Transpose(t));

            (float w, float x, float y, float z) = GaussianSplatActivations.TurnToYUp((float)q[0], (float)q[1], (float)q[2], (float)q[3]);
            worstLeft = Math.Max(worstLeft, MaxDifference(Covariance([w, x, y, z], scale), want));

            // The other order, written out: R_q · T as a rotation.
            double[,] right = Multiply(RotationMatrix(q), t);
            double[,] rightSigma = new double[3, 3];
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 3; k++)
                        rightSigma[i, j] += right[i, k] * scale[k] * scale[k] * right[j, k];
            worstRight = Math.Max(worstRight, MaxDifference(rightSigma, want));
        }

        Assert.True(worstLeft < 1e-5, $"the left product differs from the turned covariance by {worstLeft:E2}");
        Assert.True(worstRight > 1e-2, $"the right product should not match, but is within {worstRight:E2}");
    }

    private static double[,] RotationMatrix(double[] q)
    {
        double w = q[0], x = q[1], y = q[2], z = q[3];
        return new[,]
        {
            { 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y) },
            { 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x) },
            { 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y) },
        };
    }

    private static double[,] Covariance(double[] q, double[] scale)
    {
        double[,] r = RotationMatrix(q), sigma = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    sigma[i, j] += r[i, k] * scale[k] * scale[k] * r[j, k];
        return sigma;
    }

    private static double[,] Transpose(double[,] m) => new[,] { { m[0, 0], m[1, 0], m[2, 0] }, { m[0, 1], m[1, 1], m[2, 1] }, { m[0, 2], m[1, 2], m[2, 2] } };

    private static double[,] Multiply(double[,] a, double[,] b)
    {
        double[,] c = new double[3, 3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++)
                    c[i, j] += a[i, k] * b[k, j];
        return c;
    }

    private static double MaxDifference(double[,] a, double[,] b)
    {
        double worst = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                worst = Math.Max(worst, Math.Abs(a[i, j] - b[i, j]));
        return worst;
    }
}

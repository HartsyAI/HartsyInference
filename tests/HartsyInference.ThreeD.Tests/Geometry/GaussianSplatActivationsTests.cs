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

        // The other order differs in the y and z mixing only, and the two agree for a rotation about x.
        (float aw, float ax, float ay, float az) = GaussianSplatActivations.TurnToYUpAfter(qw, qx, qy, qz);
        Assert.Equal(rw, aw);
        Assert.Equal(rx, ax);
        Assert.Equal(c * qy + c * qz, ay, 5);
        Assert.Equal(c * qz - c * qy, az, 5);
        Assert.Equal(GaussianSplatActivations.TurnToYUp(0.6f, 0.8f, 0f, 0f), GaussianSplatActivations.TurnToYUpAfter(0.6f, 0.8f, 0f, 0f));
    }

    [Fact]
    public void TurnToYUpAfter_ComposesTheQuarterTurnAfterTheSplatsOwnRotation()
    {
        // Identity turns to the turn itself.
        (float w, float x, float y, float z) = GaussianSplatActivations.TurnToYUpAfter(1f, 0f, 0f, 0f);
        Assert.Equal(Math.Sqrt(0.5), w, 6);
        Assert.Equal(Math.Sqrt(0.5), x, 6);
        Assert.Equal(0f, y);
        Assert.Equal(0f, z);

        // A half turn about y is where the two orders of the product part: q ⊗ q_R = (0, 0, c, -c), while q_R ⊗ q would be (0, 0, c, c).
        (float rw, float rx, float ry, float rz) = GaussianSplatActivations.TurnToYUpAfter(0f, 0f, 1f, 0f);
        Assert.Equal(0f, rw, 6);
        Assert.Equal(0f, rx, 6);
        Assert.Equal(Math.Sqrt(0.5), ry, 6);
        Assert.Equal(-Math.Sqrt(0.5), rz, 6);

        // The product keeps a unit quaternion unit.
        (float uw, float ux, float uy, float uz) = GaussianSplatActivations.TurnToYUpAfter(0.5f, 0.5f, 0.5f, 0.5f);
        Assert.Equal(1.0, uw * (double)uw + ux * (double)ux + uy * (double)uy + uz * (double)uz, 5);
    }
}

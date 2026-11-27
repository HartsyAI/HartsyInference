using HartsyInference.ThreeD.Models.Sam3dObjects;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamPoseDecoderTests
{
    private static float[] Normalize6d(double[] raw)
    {
        float[] result = new float[6];
        for (int i = 0; i < 6; i++)
            result[i] = (float)((raw[i] - SamPoseDecoder.Rotation6dMean[i]) / SamPoseDecoder.Rotation6dStd[i]);
        return result;
    }

    private static readonly float[] Ones = [1f, 1f, 1f];
    private static readonly float[] Zeros = [0f, 0f, 0f];

    [Fact]
    public void Decode_IdentityRotationGivesTheUnitQuaternion()
    {
        SamPose pose = SamPoseDecoder.Decode(Normalize6d([1, 0, 0, 0, 1, 0]), Zeros, Zeros, Ones, Zeros);

        Assert.Equal(1f, pose.QuatW, 5);
        Assert.Equal(0f, pose.QuatX, 5);
        Assert.Equal(0f, pose.QuatY, 5);
        Assert.Equal(0f, pose.QuatZ, 5);
        Assert.Equal(1f, pose.ScaleX, 5);
    }

    [Fact]
    public void Decode_NinetyDegreesAboutZ_UsesTheBasisVectorsAsColumns()
    {
        // b1 = +y, b2 = −x gives the matrix [[0, −1, 0], [1, 0, 0], [0, 0, 1]]: a +90° turn about z, q = (cos 45°, 0, 0, sin 45°).
        SamPose pose = SamPoseDecoder.Decode(Normalize6d([0, 1, 0, -1, 0, 0]), Zeros, Zeros, Ones, Zeros);

        Assert.Equal(MathF.Sqrt(0.5f), pose.QuatW, 5);
        Assert.Equal(0f, pose.QuatX, 5);
        Assert.Equal(0f, pose.QuatY, 5);
        Assert.Equal(MathF.Sqrt(0.5f), pose.QuatZ, 5);
    }

    [Fact]
    public void Decode_GramSchmidtMakesANonOrthogonalSecondVectorOrthogonal()
    {
        // a2 has a component along a1; the result must still be a unit quaternion of a proper rotation.
        SamPose pose = SamPoseDecoder.Decode(Normalize6d([1, 0, 0, 0.7, 1, 0]), Zeros, Zeros, Ones, Zeros);

        float norm = pose.QuatW * pose.QuatW + pose.QuatX * pose.QuatX + pose.QuatY * pose.QuatY + pose.QuatZ * pose.QuatZ;
        Assert.Equal(1f, norm, 5);
        Assert.Equal(1f, pose.QuatW, 5);
    }

    [Fact]
    public void Decode_QuaternionRealPartIsNonNegative()
    {
        // A −90° turn about z: the quaternion keeps a positive real part and the sign moves to z.
        SamPose pose = SamPoseDecoder.Decode(Normalize6d([0, -1, 0, 1, 0, 0]), Zeros, Zeros, Ones, Zeros);

        Assert.True(pose.QuatW >= 0f);
        Assert.Equal(MathF.Sqrt(0.5f), pose.QuatW, 5);
        Assert.Equal(-MathF.Sqrt(0.5f), pose.QuatZ, 5);
    }

    [Fact]
    public void Decode_ScaleIsExpOfTheLatentMeanedAndTimesTheSceneScaleAndDownsampleFactor()
    {
        float[] logScale = [MathF.Log(2f), MathF.Log(4f), MathF.Log(6f)];

        SamPose pose = SamPoseDecoder.Decode(Normalize6d([1, 0, 0, 0, 1, 0]), Zeros, logScale, [0.5f, 0.5f, 0.5f], Zeros, 2f);

        // mean(2, 4, 6) · 0.5 · 2 = 4, the same on all three axes.
        Assert.Equal(4f, pose.ScaleX, 4);
        Assert.Equal(4f, pose.ScaleY, 4);
        Assert.Equal(4f, pose.ScaleZ, 4);
    }

    [Fact]
    public void Decode_TranslationIsScaledBySceneScaleThenShifted()
    {
        SamPose pose = SamPoseDecoder.Decode(Normalize6d([1, 0, 0, 0, 1, 0]), [1f, 2f, 3f], Zeros, [2f, 2f, 2f], [10f, 20f, 30f]);

        Assert.Equal(12f, pose.TranslationX, 5);
        Assert.Equal(24f, pose.TranslationY, 5);
        Assert.Equal(36f, pose.TranslationZ, 5);
    }

    [Fact]
    public void Decode_RejectsANonUniformOrNonPositiveSceneScale()
    {
        float[] r = Normalize6d([1, 0, 0, 0, 1, 0]);

        Assert.Throws<ArgumentException>(() => SamPoseDecoder.Decode(r, Zeros, Zeros, [1f, 2f, 1f], Zeros));
        Assert.Throws<ArgumentException>(() => SamPoseDecoder.Decode(r, Zeros, Zeros, [0f, 0f, 0f], Zeros));
        Assert.Throws<ArgumentException>(() => SamPoseDecoder.Decode(r, Zeros, Zeros, [-1f, -1f, -1f], Zeros));
    }

    [Fact]
    public void Decode_RejectsWrongLengths()
    {
        Assert.Throws<ArgumentException>(() => SamPoseDecoder.Decode(new float[5], Zeros, Zeros, Ones, Zeros));
    }
}

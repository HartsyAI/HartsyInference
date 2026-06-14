using Xunit;
using HartsyInference.Interactive.Camera;

namespace HartsyInference.Interactive.Tests;

/// <summary>SE(3) invariants: rigid inverse round-trip, rotation orthonormality, slerp endpoints, and the
/// action-integrator's basic motion semantics.</summary>
public class Se3MathTests
{
    [Fact]
    public void Invert_TimesOriginal_IsIdentity()
    {
        float[] pose = new float[16];
        Se3Math.GetExtrinsics(rollDeg: 10f, pitchDeg: -20f, yawDeg: 35f, x: 100f, y: -50f, z: 30f, pose);
        float[] inv = new float[16];
        Se3Math.Invert(pose, inv);

        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
            {
                float v = 0;
                for (int k = 0; k < 4; k++) v += inv[r * 4 + k] * pose[k * 4 + c];
                Assert.True(MathF.Abs(v - (r == c ? 1f : 0f)) < 1e-4f, $"inv·pose [{r},{c}] = {v}");
            }
    }

    [Fact]
    public void EulerRotation_IsOrthonormal()
    {
        float[] r = new float[9];
        Se3Math.EulerDegreesToRotation(33f, -12f, 77f, r);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                float dot = 0;
                for (int k = 0; k < 3; k++) dot += r[i * 3 + k] * r[j * 3 + k];
                Assert.True(MathF.Abs(dot - (i == j ? 1f : 0f)) < 1e-5f, $"row dot [{i},{j}] = {dot}");
            }
    }

    [Fact]
    public void GetExtrinsics_ScalesTranslationByCentimeters()
    {
        float[] pose = new float[16];
        Se3Math.GetExtrinsics(0f, 0f, 0f, 100f, 200f, 300f, pose);
        Assert.Equal(1.0f, pose[3], 5);
        Assert.Equal(2.0f, pose[7], 5);
        Assert.Equal(3.0f, pose[11], 5);
    }

    [Fact]
    public void Slerp_HitsEndpointsAndStaysUnit()
    {
        float[] q0 = [0f, 0f, 0f, 1f];                              // identity
        float s = MathF.Sin(MathF.PI / 4), c = MathF.Cos(MathF.PI / 4);
        float[] q1 = [s, 0f, 0f, c];                                // 90° about X
        float[] r = new float[4];

        Se3Math.Slerp(q0, q1, 0f, r);
        for (int i = 0; i < 4; i++) Assert.Equal(q0[i], r[i], 4);
        Se3Math.Slerp(q0, q1, 1f, r);
        for (int i = 0; i < 4; i++) Assert.Equal(q1[i], r[i], 4);

        Se3Math.Slerp(q0, q1, 0.5f, r);
        float norm = MathF.Sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]);
        Assert.Equal(1f, norm, 4);
        Assert.Equal(MathF.Sin(MathF.PI / 8), r[0], 4);             // 45° about X
    }

    [Fact]
    public void IntegrateActions_ForwardKeyMoves_MouseRotates()
    {
        float[] start = new float[16];
        Se3Math.Identity(start);
        float[][] kbdForward = [[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]];
        float[][] mouseNone = [[0, 0], [0, 0]];
        float[][] poses = Se3Math.IntegrateActions(start, kbdForward, mouseNone, moveStep: 0.5f, rotateScaleDeg: 90f);

        Assert.Equal(2, poses.Length);
        float d0 = Distance(poses[0]), d1 = Distance(poses[1]);
        Assert.True(d1 > d0 && d0 > 0, $"holding W must translate: {d0} → {d1}");

        // Pure mouse: position fixed, rotation changes.
        float[][] kbdNone = [[0, 0, 0, 0, 0, 0]];
        float[][] mouseYaw = [[0.5f, 0]];
        float[][] rotated = Se3Math.IntegrateActions(start, kbdNone, mouseYaw, moveStep: 0.5f, rotateScaleDeg: 90f);
        Assert.True(MathF.Abs(Distance(rotated[0])) < 1e-5f, "mouse-only must not translate");
        bool rotationChanged = false;
        float[] identityPose = new float[16];
        Se3Math.GetExtrinsics(0, 0, 0, 0, 0, 0, identityPose);
        for (int i = 0; i < 12; i++)
            if (MathF.Abs(rotated[0][i] - identityPose[i]) > 1e-4f) rotationChanged = true;
        Assert.True(rotationChanged, "mouse delta must rotate the camera");
    }

    private static float Distance(float[] pose) =>
        MathF.Sqrt(pose[3] * pose[3] + pose[7] * pose[7] + pose[11] * pose[11]);
}

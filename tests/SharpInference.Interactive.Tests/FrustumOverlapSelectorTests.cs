using Xunit;
using SharpInference.Interactive.Camera;
using SharpInference.Interactive.Memory;

namespace SharpInference.Interactive.Tests;

/// <summary>Geometric memory retrieval: candidates sharing the current view rank above candidates facing away, and
/// zero-overlap slots fall back to the most recent candidates.</summary>
public class FrustumOverlapSelectorTests
{
    [Fact]
    public void Select_PrefersSameFacingCameraOverOppositeFacing()
    {
        FrustumOverlapSelector selector = new(fx: 64, fy: 64, cx: 64, cy: 64, width: 128, height: 128);
        float[] current = new float[16];
        Se3Math.Identity(current);

        float[] sameFacing = new float[16];
        Se3Math.Identity(sameFacing);
        sameFacing[3] = 0.5f;   // small lateral offset, same view direction

        float[] opposite = new float[16];
        Se3Math.GetExtrinsics(0f, 0f, 180f, 0f, 0f, 0f, opposite);   // turned around

        int[] picked = selector.Select([opposite, sameFacing], current, k: 1);
        Assert.Equal(1, picked[0]);
    }

    [Fact]
    public void Select_FallsBackToMostRecentOnZeroOverlap()
    {
        FrustumOverlapSelector selector = new(fx: 64, fy: 64, cx: 64, cy: 64, width: 128, height: 128, far: 1f);
        float[] current = new float[16];
        Se3Math.Identity(current);

        // All candidates miles away — no shared visible points → most recent unselected wins.
        float[] far0 = new float[16]; Se3Math.Identity(far0); far0[3] = 1e6f;
        float[] far1 = new float[16]; Se3Math.Identity(far1); far1[3] = 1e6f;
        float[] far2 = new float[16]; Se3Math.Identity(far2); far2[3] = 1e6f;

        int[] picked = selector.Select([far0, far1, far2], current, k: 2);
        Assert.Equal(new[] { 1, 2 }, picked);   // most recent two, in historical order
    }

    [Fact]
    public void GeneratePointsInSphere_IsDeterministicAndBounded()
    {
        float[] a = FrustumOverlapSelector.GeneratePointsInSphere(256, radius: 5f, seed: 7);
        float[] b = FrustumOverlapSelector.GeneratePointsInSphere(256, radius: 5f, seed: 7);
        Assert.Equal(a, b);
        for (int i = 0; i < 256; i++)
        {
            float r = MathF.Sqrt(a[i * 3] * a[i * 3] + a[i * 3 + 1] * a[i * 3 + 1] + a[i * 3 + 2] * a[i * 3 + 2]);
            Assert.True(r <= 5.0001f, $"point {i} outside sphere: {r}");
        }
    }
}

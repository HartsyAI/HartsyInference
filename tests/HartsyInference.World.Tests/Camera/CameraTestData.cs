using HartsyInference.World.Camera;
using Xunit;

namespace HartsyInference.World.Tests.Camera;

/// <summary>Shared random camera inputs and comparison helpers for the camera primitive tests.</summary>
internal static class CameraTestData
{
    /// <summary>Random camera-to-world pose built from the Matrix-Game extrinsics path (orthonormal rotation).</summary>
    public static float[] RandomPose(Random rng)
    {
        float[] pose = new float[16];
        Se3Math.GetExtrinsics(
            Uniform(rng, -180f, 180f), Uniform(rng, -80f, 80f), Uniform(rng, -180f, 180f),
            Uniform(rng, -300f, 300f), Uniform(rng, -300f, 300f), Uniform(rng, -300f, 300f), pose);
        return pose;
    }

    public static float Uniform(Random rng, float lo, float hi) => lo + (float)rng.NextDouble() * (hi - lo);

    public static bool SameBits(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        if (expected.Length != actual.Length) return false;
        return MemoryMarshal.AsBytes(expected).SequenceEqual(MemoryMarshal.AsBytes(actual));
    }

    public static void AssertClose(float expected, float actual, float tolerance)
    {
        Assert.True(MathF.Abs(expected - actual) <= tolerance, $"expected {expected}, actual {actual}, tolerance {tolerance}");
    }
}

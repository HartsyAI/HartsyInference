using HartsyInference.World.Camera;
using Xunit;
using static HartsyInference.World.Tests.Camera.CameraTestData;

namespace HartsyInference.World.Tests.Camera;

public sealed class CameraRayFrameTests
{
    [Fact]
    public void FromCameraToWorld_SplitsOriginAndRotation()
    {
        // Rotation 90 degrees about Z (camera +X -> world +Y, camera +Y -> world -X), translation (1, 2, 3).
        float[] pose = [0f, -1f, 0f, 1f, 1f, 0f, 0f, 2f, 0f, 0f, 1f, 3f, 0f, 0f, 0f, 1f];
        CameraRayFrame frame = CameraRayFrame.FromCameraToWorld(pose);
        Assert.Equal(1f, frame.OriginX);
        Assert.Equal(2f, frame.OriginY);
        Assert.Equal(3f, frame.OriginZ);

        (float x, float y, float z) = frame.RotateToWorld(1f, 0f, 0f);
        AssertClose(0f, x, 1e-6f); AssertClose(1f, y, 1e-6f); AssertClose(0f, z, 1e-6f);

        (x, y, z) = frame.RotateToWorld(0f, 1f, 0f);
        AssertClose(-1f, x, 1e-6f); AssertClose(0f, y, 1e-6f); AssertClose(0f, z, 1e-6f);
    }

    [Fact]
    public void FromCameraToWorld_RejectsShortPose()
    {
        Assert.Throws<ArgumentException>(() => CameraRayFrame.FromCameraToWorld(new float[15]));
    }
}

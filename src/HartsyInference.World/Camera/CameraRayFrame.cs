namespace HartsyInference.World.Camera;

/// <summary>Camera-to-world pose split for ray generation: the camera center in world space and the rotation that maps a
/// camera-space direction into the world frame. The rotation is applied without normalization.</summary>
public readonly record struct CameraRayFrame(
    float OriginX, float OriginY, float OriginZ,
    float R00, float R01, float R02,
    float R10, float R11, float R12,
    float R20, float R21, float R22)
{
    /// <summary>Splits a row-major camera-to-world 4×4 pose.</summary>
    public static CameraRayFrame FromCameraToWorld(ReadOnlySpan<float> cameraToWorld)
    {
        if (cameraToWorld.Length < 16) throw new ArgumentException("cameraToWorld must hold 16 floats.", nameof(cameraToWorld));
        return new CameraRayFrame(
            cameraToWorld[3], cameraToWorld[7], cameraToWorld[11],
            cameraToWorld[0], cameraToWorld[1], cameraToWorld[2],
            cameraToWorld[4], cameraToWorld[5], cameraToWorld[6],
            cameraToWorld[8], cameraToWorld[9], cameraToWorld[10]);
    }

    /// <summary>World-space direction of a camera-space direction: <c>R · d</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public (float X, float Y, float Z) RotateToWorld(float x, float y, float z) =>
        (R00 * x + R01 * y + R02 * z, R10 * x + R11 * y + R12 * z, R20 * x + R21 * y + R22 * z);
}

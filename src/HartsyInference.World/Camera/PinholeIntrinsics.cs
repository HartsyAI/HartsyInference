namespace HartsyInference.World.Camera;

/// <summary>Pinhole intrinsics <c>[fx, fy, cx, cy]</c> in pixels of the image they describe.</summary>
public readonly record struct PinholeIntrinsics(float Fx, float Fy, float Cx, float Cy);

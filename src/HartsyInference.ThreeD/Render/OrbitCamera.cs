using System.Numerics;

namespace HartsyInference.ThreeD.Render;

/// <summary>A perspective camera orbiting the centre of a bounding sphere, far enough back that the sphere fits the picture. Geometry is turned to Y-up first when it is stored Z-up.</summary>
internal sealed class OrbitCamera
{
    private readonly Matrix4x4 _viewProjection;
    private readonly float _width, _height;

    public OrbitCamera(Vector3 min, Vector3 max, MeshRenderOptions options, int width, int height)
    {
        _width = width;
        _height = height;
        Vector3 centre = (min + max) * 0.5f;
        float radius = Math.Max((max - min).Length() * 0.5f, 1e-6f);
        float fov = options.FovDegrees * MathF.PI / 180f;
        float aspect = width / (float)height;
        float halfAngle = MathF.Atan(MathF.Min(MathF.Tan(fov / 2), MathF.Tan(fov / 2) * aspect));
        float distance = radius / MathF.Sin(halfAngle) * 1.02f;

        float azimuth = options.Viewpoint.AzimuthDegrees * MathF.PI / 180f, elevation = options.Viewpoint.ElevationDegrees * MathF.PI / 180f;
        Vector3 direction = new(MathF.Sin(azimuth) * MathF.Cos(elevation), MathF.Sin(elevation), MathF.Cos(azimuth) * MathF.Cos(elevation));
        Eye = centre + direction * distance;
        // Straight above or below, the up vector of the horizon is degenerate; the model's front direction takes its place.
        Vector3 up = MathF.Abs(MathF.Abs(options.Viewpoint.ElevationDegrees) - 90f) < 0.01f ? new Vector3(-MathF.Sin(azimuth), 0, -MathF.Cos(azimuth)) * MathF.Sign(options.Viewpoint.ElevationDegrees) : Vector3.UnitY;
        Matrix4x4 view = Matrix4x4.CreateLookAt(Eye, centre, up);
        float near = MathF.Max(distance - radius * 1.05f, distance * 0.01f), far = distance + radius * 1.05f;
        _viewProjection = view * Matrix4x4.CreatePerspectiveFieldOfView(fov, aspect, near, far);

        Vector3 forward = Vector3.Normalize(centre - Eye);
        Vector3 right = Vector3.Normalize(Vector3.Cross(forward, up));
        Vector3 cameraUp = Vector3.Cross(right, forward);
        // A headlight raised and moved to the left reads shape better than one straight down the view.
        Light = Vector3.Normalize(-forward + 0.6f * cameraUp - 0.4f * right);
    }

    /// <summary>The camera position.</summary>
    public Vector3 Eye { get; }

    /// <summary>The direction towards the light.</summary>
    public Vector3 Light { get; }

    /// <summary>The picture position of <paramref name="point"/> in samples, and its distance along the view (the clip w); w is not positive for a point behind the camera.</summary>
    public (float X, float Y, float W) Project(Vector3 point)
    {
        Vector4 clip = Vector4.Transform(new Vector4(point, 1f), _viewProjection);
        if (clip.W <= 1e-9f)
            return (0, 0, 0);
        return ((clip.X / clip.W * 0.5f + 0.5f) * _width, (1f - (clip.Y / clip.W * 0.5f + 0.5f)) * _height, clip.W);
    }

    /// <summary>The turn that makes a Z-up model Y-up: (x, y, z) becomes (x, z, -y).</summary>
    public static Vector3 ToYUp(Vector3 p, MeshUpAxis up) => up == MeshUpAxis.Z ? new Vector3(p.X, p.Z, -p.Y) : p;
}

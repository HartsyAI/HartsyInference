namespace HartsyInference.ThreeD.Models.Pixal3D;

/// <summary>The camera Pixal3D conditions on: a pinhole on the <c>-y</c> axis of TRELLIS.2's Z-up cube <c>[-0.5, 0.5]³</c>, looking along
/// <c>+y</c> with <c>+z</c> up. The cube's width fills the image width at its centre plane, so the field of view fixes the distance.</summary>
/// <param name="FovRadians">Horizontal field of view (upstream <c>camera_angle_x</c>).</param>
/// <param name="MeshScale">Upstream <c>mesh_scale</c>: how much larger than the unit grid the object's frame is assumed to be.</param>
public readonly record struct Pixal3DView(double FovRadians, double MeshScale)
{
    /// <summary>Upstream's default horizontal field of view, used when MoGe is not consulted.</summary>
    public const double DefaultFovRadians = 0.8575560450553894;

    /// <summary>The default view: <see cref="DefaultFovRadians"/> at mesh scale 1.</summary>
    public static Pixal3DView Default => new(DefaultFovRadians, 1.0);

    /// <summary>Camera distance from the origin: <c>1 / (2 · MeshScale · tan(Fov / 2))</c>, upstream's <c>distance_from_fov</c> with no
    /// extended pixels, for the grid corner <c>(-1, 0, 0)</c> landing on the image's left edge.</summary>
    public double Distance => 1.0 / (2.0 * MeshScale * Math.Tan(FovRadians / 2.0));

    /// <summary>Throws when the field of view is not in (0, π) radians or the mesh scale is not positive.</summary>
    public void Validate()
    {
        if (!(FovRadians > 0.0 && FovRadians < Math.PI))
            throw new ArgumentOutOfRangeException(nameof(FovRadians), FovRadians, "The field of view must be in (0, pi) radians.");
        if (!(MeshScale > 0.0))
            throw new ArgumentOutOfRangeException(nameof(MeshScale), MeshScale, "The mesh scale must be positive.");
    }
}

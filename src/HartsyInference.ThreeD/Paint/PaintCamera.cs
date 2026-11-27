namespace HartsyInference.ThreeD.Paint;

/// <summary>The orthographic camera Hunyuan3D Paint renders and bakes with (<c>camera_utils.get_mv_matrix</c> and
/// <c>get_orthographic_projection_matrix</c>). Matrices are row-major 4x4 acting on column vectors.</summary>
public static class PaintCamera
{
    /// <summary>Camera distance from the origin.</summary>
    public const double Distance = 1.45;

    /// <summary>Side of the orthographic window.</summary>
    public const double OrthoScale = 1.2;

    /// <summary>Near plane.</summary>
    public const double Near = 0.1;

    /// <summary>Far plane.</summary>
    public const double Far = 100.0;

    /// <summary>World-to-camera matrix for an elevation and azimuth in degrees; the camera looks at the origin along -z with +z world up.</summary>
    public static double[] WorldToCamera(double elevationDegrees, double azimuthDegrees, double distance = Distance)
    {
        double elev = -elevationDegrees * Math.PI / 180.0;
        double azim = (azimuthDegrees + 90.0) * Math.PI / 180.0;
        double[] position = [distance * Math.Cos(elev) * Math.Cos(azim), distance * Math.Cos(elev) * Math.Sin(azim), distance * Math.Sin(elev)];
        double[] look = Normalize([-position[0], -position[1], -position[2]]);
        double[] right = Normalize(Cross(look, [0, 0, 1.0]));
        double[] up = Normalize(Cross(right, look));
        double[] forward = [-look[0], -look[1], -look[2]];
        double[] m = new double[16];
        for (int c = 0; c < 3; c++)
        {
            m[c] = right[c];
            m[4 + c] = up[c];
            m[8 + c] = forward[c];
        }
        for (int r = 0; r < 3; r++)
            m[r * 4 + 3] = -(m[r * 4] * position[0] + m[r * 4 + 1] * position[1] + m[r * 4 + 2] * position[2]);
        m[15] = 1.0;
        return m;
    }

    /// <summary>Orthographic projection for the symmetric window of side <paramref name="scale"/>.</summary>
    public static double[] Orthographic(double scale = OrthoScale, double near = Near, double far = Far)
    {
        double half = scale * 0.5;
        double[] m = new double[16];
        m[0] = 1.0 / half;
        m[5] = 1.0 / half;
        m[10] = -2.0 / (far - near);
        m[11] = -(far + near) / (far - near);
        m[15] = 1.0;
        return m;
    }

    /// <summary>Transforms a point by a row-major 4x4 matrix (w assumed 1).</summary>
    public static (double X, double Y, double Z, double W) Transform(ReadOnlySpan<double> m, double x, double y, double z) =>
        (m[0] * x + m[1] * y + m[2] * z + m[3], m[4] * x + m[5] * y + m[6] * z + m[7],
         m[8] * x + m[9] * y + m[10] * z + m[11], m[12] * x + m[13] * y + m[14] * z + m[15]);

    private static double[] Cross(double[] a, double[] b) =>
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

    private static double[] Normalize(double[] v)
    {
        double len = Math.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return [v[0] / len, v[1] / len, v[2] / len];
    }
}

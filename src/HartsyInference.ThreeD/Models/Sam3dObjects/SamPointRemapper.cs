namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>Coordinate remaps of the upstream <c>PointRemapper</c>; the released config uses <see cref="Linear"/>.</summary>
public enum SamPointRemapType
{
    Linear,
    Sinh,
    Exp,
    SinhExp,
    ExpDisparity,
}

/// <summary>The remap of <c>point_remapper.py</c> on interleaved <c>x, y, z</c> points, in place, and its inverse. As upstream, <c>exp_disparity</c>
/// gives infinity or NaN for <c>z &lt;= 0</c> and <c>exp</c> for <c>z &lt;= -1</c>; invalid points are expected to be zeroed before the remap.</summary>
public static class SamPointRemapper
{
    /// <summary>Parses the config name (<c>linear</c>, <c>sinh</c>, <c>exp</c>, <c>sinh_exp</c>, <c>exp_disparity</c>).</summary>
    public static SamPointRemapType Parse(string name) => name switch
    {
        "linear" => SamPointRemapType.Linear,
        "sinh" => SamPointRemapType.Sinh,
        "exp" => SamPointRemapType.Exp,
        "sinh_exp" => SamPointRemapType.SinhExp,
        "exp_disparity" => SamPointRemapType.ExpDisparity,
        _ => throw new ArgumentException($"Invalid remap type '{name}'; expected linear, sinh, exp, sinh_exp or exp_disparity.", nameof(name)),
    };

    public static void Apply(Span<float> points, SamPointRemapType type)
    {
        RequireTriples(points);
        for (int i = 0; i < points.Length; i += 3)
        {
            float x = points[i], y = points[i + 1], z = points[i + 2];
            switch (type)
            {
                case SamPointRemapType.Linear:
                    break;
                case SamPointRemapType.Sinh:
                    points[i] = float.Asinh(x);
                    points[i + 1] = float.Asinh(y);
                    points[i + 2] = float.Asinh(z);
                    break;
                case SamPointRemapType.Exp:
                    points[i] = x / (1f + z);
                    points[i + 1] = y / (1f + z);
                    points[i + 2] = float.LogP1(z);
                    break;
                case SamPointRemapType.ExpDisparity:
                    points[i] = x / z;
                    points[i + 1] = y / z;
                    points[i + 2] = MathF.Log(z);
                    break;
                case SamPointRemapType.SinhExp:
                    points[i] = float.Asinh(x);
                    points[i + 1] = float.Asinh(y);
                    points[i + 2] = MathF.Log(MathF.Max(z, 1e-8f));
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown remap type.");
            }
        }
    }

    public static void Invert(Span<float> points, SamPointRemapType type)
    {
        RequireTriples(points);
        for (int i = 0; i < points.Length; i += 3)
        {
            float x = points[i], y = points[i + 1], z = points[i + 2];
            switch (type)
            {
                case SamPointRemapType.Linear:
                    break;
                case SamPointRemapType.Sinh:
                    points[i] = MathF.Sinh(x);
                    points[i + 1] = MathF.Sinh(y);
                    points[i + 2] = MathF.Sinh(z);
                    break;
                case SamPointRemapType.Exp:
                    float zExp = float.ExpM1(z);
                    points[i] = x * (1f + zExp);
                    points[i + 1] = y * (1f + zExp);
                    points[i + 2] = zExp;
                    break;
                case SamPointRemapType.ExpDisparity:
                    float e = MathF.Exp(z);
                    points[i] = x * e;
                    points[i + 1] = y * e;
                    points[i + 2] = e;
                    break;
                case SamPointRemapType.SinhExp:
                    points[i] = MathF.Sinh(x);
                    points[i + 1] = MathF.Sinh(y);
                    points[i + 2] = MathF.Exp(z);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown remap type.");
            }
        }
    }

    private static void RequireTriples(Span<float> points)
    {
        if (points.Length % 3 != 0)
            throw new ArgumentException($"Points are x, y, z triples; got {points.Length} values.", nameof(points));
    }
}

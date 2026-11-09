namespace HartsyInference.ThreeD.Geometry;

/// <summary>Double-precision 3-vector for the card mesher; operations follow the evaluation order of the reference NumPy code.</summary>
internal readonly record struct Vector3d(double X, double Y, double Z)
{
    public static Vector3d operator +(Vector3d a, Vector3d b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

    public static Vector3d operator -(Vector3d a, Vector3d b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);

    public static Vector3d operator *(Vector3d a, double s) => new(a.X * s, a.Y * s, a.Z * s);

    public static Vector3d operator *(double s, Vector3d a) => new(a.X * s, a.Y * s, a.Z * s);

    public static Vector3d operator /(Vector3d a, double s) => new(a.X / s, a.Y / s, a.Z / s);

    public double Length => Math.Sqrt(Dot(this, this));

    public static double Dot(Vector3d a, Vector3d b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;

    public static Vector3d Cross(Vector3d a, Vector3d b) =>
        new(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);

    /// <summary>v / (|v| + 1e-12), the reference normalisation.</summary>
    public static Vector3d Unit(Vector3d v) => v / (v.Length + 1e-12);
}

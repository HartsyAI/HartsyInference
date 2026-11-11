namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Closest point on a triangle mesh surface found by <see cref="TriangleBvh"/>.</summary>
/// <param name="Triangle">Index of the triangle holding the point.</param>
/// <param name="X">Closest point, x.</param>
/// <param name="Y">Closest point, y.</param>
/// <param name="Z">Closest point, z.</param>
/// <param name="DistanceSquared">Squared distance from the query to the closest point.</param>
/// <param name="U">Barycentric weight of the triangle's second vertex.</param>
/// <param name="V">Barycentric weight of the triangle's third vertex; the first vertex weighs <c>1 - U - V</c>.</param>
public readonly record struct BvhHit(int Triangle, double X, double Y, double Z, double DistanceSquared, double U, double V)
{
    /// <summary>Unsigned distance from the query to the surface.</summary>
    public double Distance => Math.Sqrt(DistanceSquared);
}

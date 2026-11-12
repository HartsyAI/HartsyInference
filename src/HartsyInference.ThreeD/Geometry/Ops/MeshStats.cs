namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>The measurements <see cref="MeshStatistics.Measure"/> takes of a triangle mesh.</summary>
/// <param name="VertexCount">Vertices as the mesh stores them (a seam duplicates a vertex).</param>
/// <param name="TriangleCount">Triangles.</param>
/// <param name="Min">Smallest <c>[x, y, z]</c>.</param>
/// <param name="Max">Largest <c>[x, y, z]</c>.</param>
/// <param name="Components">Vertex-connected pieces of the welded surface.</param>
/// <param name="BoundaryEdges">Edges of the welded surface with one face.</param>
/// <param name="NonManifoldEdges">Edges of the welded surface with more than two faces.</param>
/// <param name="Watertight">The surface has triangles and neither boundary nor non-manifold edges.</param>
public sealed record MeshStats(
    int VertexCount, int TriangleCount, float[] Min, float[] Max, int Components, long BoundaryEdges, long NonManifoldEdges, bool Watertight);

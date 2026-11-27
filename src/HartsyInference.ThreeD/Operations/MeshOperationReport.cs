namespace HartsyInference.ThreeD.Operations;

/// <summary>What one step did.</summary>
/// <param name="Operation">The operation id.</param>
/// <param name="VerticesBefore">Vertices before.</param>
/// <param name="VerticesAfter">Vertices after.</param>
/// <param name="TrianglesBefore">Triangles before.</param>
/// <param name="TrianglesAfter">Triangles after.</param>
/// <param name="Notes">What it could not keep or did not do.</param>
public sealed record MeshOperationReport(string Operation, long VerticesBefore, long VerticesAfter, long TrianglesBefore, long TrianglesAfter, IReadOnlyList<string> Notes);

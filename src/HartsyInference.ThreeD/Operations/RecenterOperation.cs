using System.Numerics;
using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Operations;

/// <summary>Moves the whole model (every primitive by the same amount, so parts stay in place) so its bounding box is centred on the origin, or, with <c>ground</c>, centred over the origin with its lowest point on the floor. Only positions change; everything else is kept.</summary>
public sealed class RecenterOperation : IMeshOperation
{
    /// <inheritdoc/>
    public string Id => "recenter";

    /// <inheritdoc/>
    public string Description => "Centre the model's bounding box on the origin (ground = 1 stands it on the floor instead). Keeps everything else.";

    /// <inheritdoc/>
    public IReadOnlyList<MeshOperationParameter> Parameters { get; } =
    [
        new("ground", "0 centres on all three axes; 1 centres across the floor and puts the lowest point at 0 along the up axis.", 0, 0, 1, Integer: true),
        new("up", "The up axis for ground: 0 for Y, 1 for Z.", 0, 0, 1, Integer: true),
    ];

    /// <inheritdoc/>
    public MeshOperationOutcome Apply(MeshDocument document, MeshOperationArguments arguments, CancellationToken cancel)
    {
        (Vector3 min, Vector3 max) = Bounds(document);
        Vector3 centre = (min + max) * 0.5f;
        Vector3 shift = -centre;
        if (arguments.GetInt("ground") == 1)
        {
            int up = arguments.GetInt("up") == 1 ? 2 : 1;
            float lowest = up == 2 ? min.Z : min.Y;
            shift = up == 2 ? new Vector3(-centre.X, -centre.Y, -lowest) : new Vector3(-centre.X, -lowest, -centre.Z);
        }
        return PrimitiveEdit.Map(document, Id, (_, primitive) =>
        {
            float[] moved = new float[primitive.Geometry.Vertices.Length];
            for (int i = 0; i < moved.Length; i += 3)
            {
                moved[i] = primitive.Geometry.Vertices[i] + shift.X;
                moved[i + 1] = primitive.Geometry.Vertices[i + 1] + shift.Y;
                moved[i + 2] = primitive.Geometry.Vertices[i + 2] + shift.Z;
            }
            return (new MeshPrimitive { Geometry = PrimitiveEdit.WithPositions(primitive.Geometry, moved), Material = primitive.Material }, null);
        }, cancel);
    }

    /// <summary>The bounding box of every vertex of every primitive.</summary>
    internal static (Vector3 Min, Vector3 Max) Bounds(MeshDocument document)
    {
        Vector3 min = new(float.PositiveInfinity), max = new(float.NegativeInfinity);
        foreach (MeshPrimitive primitive in document.Primitives)
        {
            float[] v = primitive.Geometry.Vertices;
            for (int i = 0; i < v.Length; i += 3)
            {
                Vector3 p = new(v[i], v[i + 1], v[i + 2]);
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
            }
        }
        return (min, max);
    }
}

using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Operations;

/// <summary>Scales the whole model uniformly about the origin: by a factor, or so its longest bounding-box side has a given length (to make a model that came out in arbitrary units a metre long, say). Normals, UVs, colours and textures are kept.</summary>
public sealed class RescaleOperation : IMeshOperation
{
    /// <inheritdoc/>
    public string Id => "rescale";

    /// <inheritdoc/>
    public string Description => "Scale the model uniformly about the origin: give factor, or longestSide to make the longest bounding-box side that long. Keeps everything else.";

    /// <inheritdoc/>
    public IReadOnlyList<MeshOperationParameter> Parameters { get; } =
    [
        new("factor", "The scale factor.", null, 1e-6, 1e6),
        new("longestSide", "The length the longest bounding-box side should have; an alternative to factor.", null, 1e-6, 1e6),
    ];

    /// <inheritdoc/>
    public void Check(MeshOperationArguments arguments)
    {
        if (arguments.Has("factor") == arguments.Has("longestSide"))
            throw new ArgumentException("rescale takes exactly one of factor or longestSide.");
    }

    /// <inheritdoc/>
    public MeshOperationOutcome Apply(MeshDocument document, MeshOperationArguments arguments, CancellationToken cancel)
    {
        Check(arguments);
        float factor;
        if (arguments.Has("factor"))
        {
            factor = (float)arguments.Get("factor");
        }
        else
        {
            (System.Numerics.Vector3 min, System.Numerics.Vector3 max) = RecenterOperation.Bounds(document);
            System.Numerics.Vector3 size = max - min;
            float longest = MathF.Max(size.X, MathF.Max(size.Y, size.Z));
            if (!(longest > 0))
                throw new ArgumentException("The model has no extent to scale to a length.");
            factor = (float)arguments.Get("longestSide") / longest;
        }
        return PrimitiveEdit.Map(document, Id, (_, primitive) =>
        {
            float[] scaled = new float[primitive.Geometry.Vertices.Length];
            for (int i = 0; i < scaled.Length; i++)
                scaled[i] = primitive.Geometry.Vertices[i] * factor;
            return (new MeshPrimitive { Geometry = PrimitiveEdit.WithPositions(primitive.Geometry, scaled), Material = primitive.Material }, null);
        }, cancel);
    }
}

using HartsyInference.ThreeD.Document;

namespace HartsyInference.ThreeD.Operations;

/// <summary>An edit of a <see cref="MeshDocument"/>: decimate, fill holes, smooth and so on. It works on the document, so it applies to the output of every 3D model, and a new operation is one class and one line in <see cref="MeshOperations.All"/>; nothing about the model that made the mesh is known or needed.</summary>
public interface IMeshOperation
{
    /// <summary>The id a request names it by (<c>decimate</c>, <c>fill-holes</c>, ...).</summary>
    string Id { get; }

    /// <summary>What it does, and what it keeps or loses (textures, UVs, colours).</summary>
    string Description { get; }

    /// <summary>The numbers it takes.</summary>
    IReadOnlyList<MeshOperationParameter> Parameters { get; }

    /// <summary>Applies it. The arguments are already validated against <see cref="Parameters"/>; an operation that needs one of several optional numbers checks that itself and throws <see cref="ArgumentException"/>.</summary>
    /// <exception cref="ArgumentException">The combination of arguments is not valid for this operation.</exception>
    MeshOperationOutcome Apply(MeshDocument document, MeshOperationArguments arguments, CancellationToken cancel);
}

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

    /// <summary>Checks the combination of arguments, which the per-parameter ranges cannot say (decimate takes triangles or ratio, not both and not neither). <see cref="MeshOperations.Run"/> calls it for every step before any step runs, so a bad last step costs no work; the default accepts everything.</summary>
    /// <exception cref="ArgumentException">The combination of arguments is not valid for this operation.</exception>
    void Check(MeshOperationArguments arguments)
    {
    }

    /// <summary>Applies it. The arguments are already validated against <see cref="Parameters"/> and <see cref="Check"/>.</summary>
    MeshOperationOutcome Apply(MeshDocument document, MeshOperationArguments arguments, CancellationToken cancel);
}

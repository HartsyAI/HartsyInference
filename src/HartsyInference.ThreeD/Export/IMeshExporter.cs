using HartsyInference.ThreeD.Document;

namespace HartsyInference.ThreeD.Export;

/// <summary>Writes a <see cref="MeshDocument"/> in one file format. It knows the format and nothing about the model that made the mesh, so a new format is one class and one line in <see cref="MeshExporters"/>.</summary>
public interface IMeshExporter
{
    /// <summary>The lower-case format id a caller names it by (<c>obj</c>, <c>ply</c>, <c>stl</c>, <c>glb</c>).</summary>
    string Id { get; }

    /// <summary>Writes <paramref name="document"/>. Everything the format cannot carry is listed in <see cref="MeshExport.Notes"/>.</summary>
    /// <param name="document">The model.</param>
    /// <param name="name">The base file name, without an extension.</param>
    MeshExport Export(MeshDocument document, string name);
}

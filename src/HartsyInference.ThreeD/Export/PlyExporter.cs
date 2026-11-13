using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.ThreeD.Export;

/// <summary>Binary little-endian PLY of the merged mesh: positions, normals and vertex colours when every primitive has them. Textures and UVs have no PLY property here and are dropped with a note. A Gaussian-splat PLY is a different file and is not produced from a mesh.</summary>
public sealed class PlyExporter : IMeshExporter
{
    /// <inheritdoc/>
    public string Id => "ply";

    /// <inheritdoc/>
    public MeshExport Export(MeshDocument document, string name)
    {
        ArgumentNullException.ThrowIfNull(document);
        List<string> notes = MeshExporters.CommonNotes(document, carriesMotion: false);
        if (document.HasTextures)
            notes.Add("Textures were dropped; PLY carries geometry, normals and vertex colours only. Export obj or glb to keep them.");
        else if (document.Primitives.Any(p => p.Geometry.Uvs is not null))
            notes.Add("UV coordinates were dropped; PLY carries geometry, normals and vertex colours only.");
        if (document.Primitives.Any(p => p.Geometry.VertexColors is null) && document.Primitives.Any(p => p.Geometry.VertexColors is not null))
            notes.Add("Only some primitives had vertex colours, so none were written.");
        byte[] bytes = PlyWriter.WriteMeshBinary(document.Merge());
        return new MeshExport([new MeshExportFile(MeshFileNames.Stem(name) + ".ply", "application/octet-stream", bytes)], notes);
    }
}

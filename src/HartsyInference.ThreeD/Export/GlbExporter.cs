using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.ThreeD.Export;

/// <summary>Binary glTF with everything a <see cref="MeshDocument"/> holds: geometry, normals, UVs, vertex colours, materials with their images and the source's <c>asset.extras</c>. Used to turn another format into glb and to write an edited model back.</summary>
public sealed class GlbExporter : IMeshExporter
{
    /// <inheritdoc/>
    public string Id => "glb";

    /// <inheritdoc/>
    public MeshExport Export(MeshDocument document, string name)
    {
        ArgumentNullException.ThrowIfNull(document);
        return new MeshExport(
            [new MeshExportFile(MeshFileNames.Stem(name) + ".glb", "model/gltf-binary", MeshDocumentGlbWriter.Write(document))],
            MeshExporters.CommonNotes(document, carriesMotion: false));
    }
}

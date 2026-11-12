using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Export;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.Engine.Services;

/// <summary>The conversion service: reads with <see cref="MeshDocumentReader"/>, writes with a <see cref="MeshExporters"/> entry, and reports what the target could not carry.</summary>
internal sealed class MeshConversionService : IMeshConversionService
{
    /// <inheritdoc/>
    public IReadOnlyList<string> Sources => MeshDocumentReader.Formats;

    /// <inheritdoc/>
    public IReadOnlyList<string> Targets { get; } = [.. MeshExporters.All.Select(e => e.Id)];

    /// <inheritdoc/>
    public MeshConversion Convert(byte[] data, string source, string target, string? name = null)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentException.ThrowIfNullOrWhiteSpace(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(target);
        IMeshExporter exporter = MeshExporters.Get(target);
        if (string.Equals(source, "ply", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException("A PLY file is not read: the engine's PLY output is a Gaussian-splat cloud, which holds splats rather than a mesh. Convert a mesh from glb or obj.");
        MeshDocument document = MeshDocumentReader.Read(data, source);
        MeshExport export = exporter.Export(document, name ?? "model");
        return new MeshConversion
        {
            Target = exporter.Id,
            Files = export.Files,
            Notes = export.Notes,
            Vertices = document.VertexCount,
            Triangles = document.TriangleCount,
            StaticOnly = document.HadMotion,
        };
    }
}

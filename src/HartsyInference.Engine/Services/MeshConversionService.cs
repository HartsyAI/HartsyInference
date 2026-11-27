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

    /// <summary>Whether the PLY header (the first 64 KiB) lists the properties only a Gaussian-splat file has.</summary>
    private static bool IsSplatHeader(byte[] ply)
    {
        string header = System.Text.Encoding.ASCII.GetString(ply, 0, Math.Min(ply.Length, 1 << 16));
        int end = header.IndexOf("end_header", StringComparison.Ordinal);
        return (end < 0 ? header : header[..end]).Contains("f_dc_0", StringComparison.Ordinal);
    }

    /// <inheritdoc/>
    public MeshConversion Convert(byte[] data, string source, string target, string? name = null)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentException.ThrowIfNullOrWhiteSpace(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(target);
        IMeshExporter exporter = MeshExporters.Get(target);
        if (string.Equals(source, "ply", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException(IsSplatHeader(data)
                ? "A Gaussian-splat PLY holds splats, not a mesh, so it cannot be converted to a mesh format."
                : "A mesh PLY is not read as a source yet; send the model as glb or obj.");
        }
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

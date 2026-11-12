using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Export;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Operations;

namespace HartsyInference.Engine.Services;

/// <summary>The operation service: reads with <see cref="MeshDocumentReader"/>, runs <see cref="MeshOperations"/>, writes with a <see cref="MeshExporters"/> entry.</summary>
internal sealed class MeshOperationService : IMeshOperationService
{
    /// <inheritdoc/>
    public IReadOnlyList<IMeshOperation> Operations => ThreeD.Operations.MeshOperations.All;

    /// <inheritdoc/>
    public MeshOperationResult Apply(byte[] data, string source, IReadOnlyList<MeshOperationStep> steps, string target = "glb", string? name = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(steps);
        ArgumentException.ThrowIfNullOrWhiteSpace(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(target);
        IMeshExporter exporter = MeshExporters.Get(target);
        MeshDocument document = MeshDocumentReader.Read(data, source);
        MeshOperationRun run = ThreeD.Operations.MeshOperations.Run(document, steps, cancel);
        MeshExport export = exporter.Export(run.Document, name ?? "model");
        return new MeshOperationResult
        {
            Output = new MeshConversion
            {
                Target = exporter.Id,
                Files = export.Files,
                Notes = export.Notes,
                Vertices = run.Document.VertexCount,
                Triangles = run.Document.TriangleCount,
                StaticOnly = run.Document.HadMotion,
            },
            Steps = run.Steps,
        };
    }
}

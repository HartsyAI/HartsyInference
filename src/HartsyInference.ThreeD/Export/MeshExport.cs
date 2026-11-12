namespace HartsyInference.ThreeD.Export;

/// <summary>What an exporter wrote and what it could not carry.</summary>
/// <param name="Files">The files, the model file first.</param>
/// <param name="Notes">What the target format dropped or could not express, in words a host can show; empty when nothing was lost.</param>
public sealed record MeshExport(IReadOnlyList<MeshExportFile> Files, IReadOnlyList<string> Notes);

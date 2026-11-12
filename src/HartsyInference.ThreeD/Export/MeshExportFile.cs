namespace HartsyInference.ThreeD.Export;

/// <summary>One file of an export.</summary>
/// <param name="Name">The file name, without a folder; files that refer to each other (an OBJ and its MTL and images) are named so they work side by side.</param>
/// <param name="ContentType">The media type.</param>
/// <param name="Data">The bytes.</param>
public sealed record MeshExportFile(string Name, string ContentType, byte[] Data);

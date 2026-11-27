using System.IO.Compression;
using HartsyInference.ThreeD.Export;

namespace HartsyInference.Engine.Requests;

/// <summary>The result of converting a model file to another format.</summary>
public sealed record MeshConversion
{
    /// <summary>The format the files are in.</summary>
    public required string Target { get; init; }

    /// <summary>The files, the model file first; an OBJ comes with its MTL and its images, which only work together.</summary>
    public required IReadOnlyList<MeshExportFile> Files { get; init; }

    /// <summary>What the target format dropped or could not express (a skin, animations, textures, colours, primitives in another mode), in words a host can show; empty when nothing was lost.</summary>
    public IReadOnlyList<string> Notes { get; init; } = [];

    /// <summary>Vertices written.</summary>
    public long Vertices { get; init; }

    /// <summary>Triangles written.</summary>
    public long Triangles { get; init; }

    /// <summary>The source had a skin or animations, and the output is static geometry.</summary>
    public bool StaticOnly { get; init; }

    /// <summary>The files as one zip archive, which is how a multi-file export (an OBJ with its MTL and images) travels as a single download.</summary>
    public byte[] Archive()
    {
        using MemoryStream stream = new();
        using (ZipArchive zip = new(stream, ZipArchiveMode.Create, leaveOpen: true))
        {
            foreach (MeshExportFile file in Files)
            {
                ZipArchiveEntry entry = zip.CreateEntry(file.Name, CompressionLevel.Optimal);
                using Stream output = entry.Open();
                output.Write(file.Data);
            }
        }
        return stream.ToArray();
    }
}

using System.Text;
using HartsyInference.ThreeD.Document;

namespace HartsyInference.ThreeD.Io;

/// <summary>Reads a model file into a <see cref="MeshDocument"/> by its format id (<c>glb</c> or <c>obj</c>). To read another format, add a case here; nothing downstream depends on which format a document came from.</summary>
public static class MeshDocumentReader
{
    /// <summary>The formats <see cref="Read"/> takes.</summary>
    public static IReadOnlyList<string> Formats { get; } = ["glb", "obj"];

    /// <summary>Reads <paramref name="data"/> as <paramref name="format"/> (case-insensitive).</summary>
    /// <exception cref="NotSupportedException">The format is not one of <see cref="Formats"/>.</exception>
    public static MeshDocument Read(byte[] data, string format)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(format);
        return format.ToLowerInvariant() switch
        {
            "glb" => GlbReader.ParseDocument(data),
            "obj" => ObjDocumentReader.Parse(Encoding.UTF8.GetString(data)),
            _ => throw new NotSupportedException($"Mesh format '{format}' cannot be read; expected one of {string.Join(", ", Formats)}."),
        };
    }
}

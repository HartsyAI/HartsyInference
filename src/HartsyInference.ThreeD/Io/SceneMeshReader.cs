using System.Text;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>Reads a scene mesh by its container format: <c>obj</c> (Wavefront text) or <c>glb</c> (binary glTF).</summary>
public static class SceneMeshReader
{
    /// <summary>Parses <paramref name="data"/> as the named format, which is matched case-insensitively.</summary>
    public static Mesh Read(byte[] data, string format)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(format);
        return format.ToLowerInvariant() switch
        {
            "obj" => ObjReader.Parse(Encoding.UTF8.GetString(data)),
            "glb" => GlbReader.Parse(data),
            _ => throw new NotSupportedException($"Scene mesh format '{format}' is not supported; expected obj or glb."),
        };
    }

    /// <summary>The format named by a file's extension (<c>obj</c> or <c>glb</c>).</summary>
    public static string FormatOf(string path)
    {
        string extension = Path.GetExtension(path).TrimStart('.').ToLowerInvariant();
        return extension is "obj" or "glb"
            ? extension
            : throw new NotSupportedException($"Scene mesh '{path}' must be a .obj or .glb file.");
    }
}

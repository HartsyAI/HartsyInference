using System.Text;
using System.Text.Json.Nodes;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Io;

/// <summary>Embeds a JSON description of how a file was made: in a GLB under <c>asset.extras</c> (a field the glTF spec reserves for application data), in a PLY as <c>comment</c> header lines, so a file kept outside the application can still say how it was made.</summary>
public static class MeshMetadata
{
    /// <summary>The key the description is stored under in <c>asset.extras</c>.</summary>
    public const string Key = "hartsyinference";

    /// <summary>The prefix of the PLY comment line that carries the description.</summary>
    public const string PlyPrefix = "comment hartsyinference ";

    /// <summary>Returns <paramref name="glb"/> with <paramref name="description"/> under <c>asset.extras.hartsyinference</c>, keeping any extras the file already has.</summary>
    /// <exception cref="HartsyInferenceException">The bytes are not a GLB with a JSON chunk.</exception>
    public static byte[] SetGlb(byte[] glb, JsonObject description)
    {
        ArgumentNullException.ThrowIfNull(description);
        JsonObject root = JsonNode.Parse(GlbContainer.Split(glb).Json)!.AsObject();
        JsonObject asset = root["asset"] as JsonObject ?? new JsonObject { ["version"] = "2.0" };
        JsonObject extras = asset["extras"] as JsonObject ?? new JsonObject();
        extras[Key] = description.DeepClone();
        asset["extras"] = extras;
        root["asset"] = asset;
        return GlbContainer.ReplaceJson(glb, Encoding.UTF8.GetBytes(root.ToJsonString()));
    }

    /// <summary>The description stored by <see cref="SetGlb"/>, or null when the file has none.</summary>
    public static JsonObject? ReadGlb(byte[] glb)
    {
        JsonObject root = JsonNode.Parse(GlbContainer.Split(glb).Json)!.AsObject();
        return root["asset"]?["extras"]?[Key] as JsonObject;
    }

    /// <summary>Returns <paramref name="ply"/> with <paramref name="description"/> as the one <see cref="PlyPrefix"/> comment line of its header, after the <c>format</c> line; a line an earlier call wrote is replaced.</summary>
    /// <exception cref="HartsyInferenceException">The bytes have no PLY header.</exception>
    public static byte[] SetPly(byte[] ply, JsonObject description)
    {
        ArgumentNullException.ThrowIfNull(description);
        (int start, int end) = HeaderBounds(ply);
        List<string> lines = [.. Encoding.UTF8.GetString(ply, start, end - start).Split('\n')];
        lines.RemoveAll(line => line.StartsWith(PlyPrefix, StringComparison.Ordinal));
        int format = lines.FindIndex(line => line.StartsWith("format ", StringComparison.Ordinal));
        if (format < 0)
            throw new HartsyInferenceException("The PLY header has no format line.");
        lines.Insert(format + 1, PlyPrefix + description.ToJsonString().Replace('\n', ' '));
        byte[] header = Encoding.UTF8.GetBytes(string.Join('\n', lines));
        byte[] result = new byte[header.Length + ply.Length - end];
        Buffer.BlockCopy(header, 0, result, 0, header.Length);
        Buffer.BlockCopy(ply, end, result, header.Length, ply.Length - end);
        return result;
    }

    /// <summary>The description stored by <see cref="SetPly"/>, or null when the header has none.</summary>
    public static JsonObject? ReadPly(byte[] ply)
    {
        (int start, int end) = HeaderBounds(ply);
        foreach (string line in Encoding.UTF8.GetString(ply, start, end - start).Split('\n'))
        {
            if (line.StartsWith(PlyPrefix, StringComparison.Ordinal))
                return JsonNode.Parse(line[PlyPrefix.Length..]) as JsonObject;
        }
        return null;
    }

    /// <summary>The header's text span: from the first byte to just before the newline that ends the <c>end_header</c> line, so a join of its lines needs one newline to close it.</summary>
    private static (int Start, int End) HeaderBounds(byte[] ply)
    {
        byte[] marker = Encoding.ASCII.GetBytes("end_header");
        int limit = Math.Min(ply.Length, 1 << 16) - marker.Length;
        for (int i = 0; i <= limit; i++)
        {
            if (data(ply, i, marker))
            {
                int end = i + marker.Length;
                return end < ply.Length && ply[end] == (byte)'\n' ? (0, end) : throw new HartsyInferenceException("The PLY header has no newline after end_header.");
            }
        }
        throw new HartsyInferenceException("The bytes have no PLY header.");

        static bool data(byte[] bytes, int at, byte[] pattern) => bytes.AsSpan(at, pattern.Length).SequenceEqual(pattern);
    }
}

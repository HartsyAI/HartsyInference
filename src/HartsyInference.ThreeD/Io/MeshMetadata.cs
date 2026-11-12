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

    /// <summary>Returns <paramref name="ply"/> with <paramref name="description"/> as one comment line after the <c>format</c> line of its header.</summary>
    /// <exception cref="HartsyInferenceException">The bytes have no PLY header.</exception>
    public static byte[] SetPly(byte[] ply, JsonObject description)
    {
        ArgumentNullException.ThrowIfNull(description);
        string marker = "\nformat ";
        int format = IndexOf(ply, Encoding.ASCII.GetBytes(marker));
        if (format < 0)
            throw new HartsyInferenceException("The bytes have no PLY header.");
        int lineEnd = Array.IndexOf(ply, (byte)'\n', format + 1);
        byte[] line = Encoding.UTF8.GetBytes(PlyPrefix + description.ToJsonString().Replace('\n', ' ') + "\n");
        byte[] result = new byte[ply.Length + line.Length];
        Buffer.BlockCopy(ply, 0, result, 0, lineEnd + 1);
        Buffer.BlockCopy(line, 0, result, lineEnd + 1, line.Length);
        Buffer.BlockCopy(ply, lineEnd + 1, result, lineEnd + 1 + line.Length, ply.Length - lineEnd - 1);
        return result;
    }

    private static int IndexOf(byte[] data, byte[] pattern)
    {
        int limit = Math.Min(data.Length, 1 << 16) - pattern.Length;
        for (int i = 0; i <= limit; i++)
        {
            if (data.AsSpan(i, pattern.Length).SequenceEqual(pattern))
                return i;
        }
        return -1;
    }
}

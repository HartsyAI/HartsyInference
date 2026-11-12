using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>Writes a segmented mesh to binary glTF 2.0: one node, mesh and flat-coloured material per part (<c>part_{id}</c>), faces with no part in a grey
/// <c>unassigned</c> node. A node's <c>extras</c> carry the part id, its face count and its axis-aligned box <c>[minX, minY, minZ, maxX, maxY, maxZ]</c>.
/// Each part's vertices are copied out of the source mesh, so the parts do not share accessors.</summary>
public static class PartGlbWriter
{
    private const uint Magic = 0x46546C67, JsonChunk = 0x4E4F534A, BinChunk = 0x004E4942;

    /// <summary>The colour of the part at <paramref name="index"/>: hues spaced by the golden angle so neighbouring indices differ, at a fixed saturation and value.</summary>
    public static float[] Color(int index)
    {
        double hue = index * 0.61803398875 % 1.0, h = hue * 6, f = h - Math.Floor(h);
        const double v = 0.85, s = 0.65;
        double p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
        (double r, double g, double b) = (int)Math.Floor(h) switch
        {
            0 => (v, t, p),
            1 => (q, v, p),
            2 => (p, v, t),
            3 => (p, q, v),
            4 => (t, p, v),
            _ => (v, p, q),
        };
        return [(float)r, (float)g, (float)b, 1f];
    }

    /// <summary>Serializes <paramref name="mesh"/>, split by <paramref name="faceIds"/> (one label per triangle; labels in <paramref name="partIds"/> are parts, anything else is
    /// unassigned). <paramref name="boxes"/> holds one box per entry of <paramref name="partIds"/>.</summary>
    public static byte[] Write(Mesh mesh, int[] faceIds, int[] partIds, float[][] boxes)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        ArgumentNullException.ThrowIfNull(faceIds);
        ArgumentNullException.ThrowIfNull(partIds);
        ArgumentNullException.ThrowIfNull(boxes);
        if (faceIds.Length != mesh.TriangleCount)
            throw new ArgumentException("One face label per triangle is required.", nameof(faceIds));
        if (boxes.Length != partIds.Length)
            throw new ArgumentException("One box per part is required.", nameof(boxes));

        // One pass over the faces: the faces of each part, and those whose label is no part.
        Dictionary<int, List<int>> byPart = partIds.Distinct().ToDictionary(id => id, _ => new List<int>());
        List<int> loose = [];
        for (int f = 0; f < faceIds.Length; f++)
            (byPart.TryGetValue(faceIds[f], out List<int>? faces) ? faces : loose).Add(f);
        List<Node> nodes = [];
        for (int i = 0; i < partIds.Length; i++)
        {
            int[] partFaces = [.. byPart[partIds[i]]];
            if (partFaces.Length > 0)
                nodes.Add(Split(mesh, partFaces, $"part_{partIds[i]}", Color(i), partIds[i], boxes[i]));
        }

        if (loose.Count > 0)
            nodes.Add(Split(mesh, [.. loose], "unassigned", [0.45f, 0.45f, 0.45f, 1f], -1, null));
        if (nodes.Count == 0)
            throw new ArgumentException("The mesh has no faces to export.", nameof(mesh));

        int binLength = 0;
        foreach (Node node in nodes)
        {
            node.PositionOffset = binLength;
            binLength += node.Vertices.Length * 4;
            node.IndexOffset = binLength;
            binLength += node.Indices.Length * 4;
        }

        byte[] bin = new byte[binLength];
        foreach (Node node in nodes)
        {
            for (int i = 0; i < node.Vertices.Length; i++)
                BinaryPrimitives.WriteSingleLittleEndian(bin.AsSpan(node.PositionOffset + i * 4), node.Vertices[i]);
            for (int i = 0; i < node.Indices.Length; i++)
                BinaryPrimitives.WriteUInt32LittleEndian(bin.AsSpan(node.IndexOffset + i * 4), (uint)node.Indices[i]);
        }

        byte[] json = Pad(BuildJson(nodes, binLength), 0x20), binPadded = Pad(bin, 0x00);
        int total = 12 + 8 + json.Length + 8 + binPadded.Length;
        byte[] glb = new byte[total];
        int p = 0;
        WriteU32(glb, ref p, Magic);
        WriteU32(glb, ref p, 2);
        WriteU32(glb, ref p, (uint)total);
        WriteU32(glb, ref p, (uint)json.Length);
        WriteU32(glb, ref p, JsonChunk);
        json.CopyTo(glb, p);
        p += json.Length;
        WriteU32(glb, ref p, (uint)binPadded.Length);
        WriteU32(glb, ref p, BinChunk);
        binPadded.CopyTo(glb, p);
        return glb;
    }

    private sealed class Node
    {
        public required string Name { get; init; }
        public required float[] Color { get; init; }
        public required int PartId { get; init; }
        public required float[] Vertices { get; init; }
        public required int[] Indices { get; init; }
        public required float[] Box { get; init; }
        public int PositionOffset, IndexOffset;
    }

    // The sub-mesh of `faces`: its vertices copied and renumbered in first-use order, and its box (the caller's, else measured).
    private static Node Split(Mesh mesh, int[] faces, string name, float[] color, int partId, float[]? box)
    {
        Dictionary<int, int> map = [];
        List<float> vertices = [];
        int[] indices = new int[faces.Length * 3];
        float[] min = [float.MaxValue, float.MaxValue, float.MaxValue], max = [float.MinValue, float.MinValue, float.MinValue];
        for (int f = 0; f < faces.Length; f++)
            for (int c = 0; c < 3; c++)
            {
                int source = mesh.Indices[faces[f] * 3 + c];
                if (!map.TryGetValue(source, out int local))
                {
                    local = map.Count;
                    map[source] = local;
                    for (int a = 0; a < 3; a++)
                    {
                        float value = mesh.Vertices[source * 3 + a];
                        vertices.Add(value);
                        min[a] = Math.Min(min[a], value);
                        max[a] = Math.Max(max[a], value);
                    }
                }

                indices[f * 3 + c] = local;
            }

        return new Node { Name = name, Color = color, PartId = partId, Vertices = [.. vertices], Indices = indices, Box = box ?? [.. min, .. max] };
    }

    private static byte[] BuildJson(List<Node> nodes, int binLength)
    {
        using MemoryStream stream = new();
        using (Utf8JsonWriter w = new(stream))
        {
            w.WriteStartObject();
            w.WriteStartObject("asset");
            w.WriteString("version", "2.0");
            w.WriteString("generator", "HartsyInference.ThreeD");
            w.WriteEndObject();
            w.WriteNumber("scene", 0);
            w.WriteStartArray("scenes");
            w.WriteStartObject();
            w.WriteStartArray("nodes");
            for (int i = 0; i < nodes.Count; i++)
                w.WriteNumberValue(i);
            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("nodes");
            for (int i = 0; i < nodes.Count; i++)
            {
                Node node = nodes[i];
                w.WriteStartObject();
                w.WriteString("name", node.Name);
                w.WriteNumber("mesh", i);
                w.WriteStartObject("extras");
                w.WriteNumber("partId", node.PartId);
                w.WriteNumber("faceCount", node.Indices.Length / 3);
                w.WriteStartArray("aabb");
                foreach (float value in node.Box)
                    w.WriteNumberValue(value);
                w.WriteEndArray();
                w.WriteEndObject();
                w.WriteEndObject();
            }

            w.WriteEndArray();

            w.WriteStartArray("materials");
            foreach (Node node in nodes)
            {
                w.WriteStartObject();
                w.WriteString("name", node.Name);
                w.WriteStartObject("pbrMetallicRoughness");
                w.WriteStartArray("baseColorFactor");
                foreach (float value in node.Color)
                    w.WriteNumberValue(value);
                w.WriteEndArray();
                w.WriteNumber("metallicFactor", 0);
                w.WriteNumber("roughnessFactor", 0.8);
                w.WriteEndObject();
                w.WriteEndObject();
            }

            w.WriteEndArray();

            w.WriteStartArray("meshes");
            for (int i = 0; i < nodes.Count; i++)
            {
                w.WriteStartObject();
                w.WriteStartArray("primitives");
                w.WriteStartObject();
                w.WriteStartObject("attributes");
                w.WriteNumber("POSITION", i * 2);
                w.WriteEndObject();
                w.WriteNumber("indices", i * 2 + 1);
                w.WriteNumber("material", i);
                w.WriteNumber("mode", 4);
                w.WriteEndObject();
                w.WriteEndArray();
                w.WriteEndObject();
            }

            w.WriteEndArray();

            w.WriteStartArray("bufferViews");
            foreach (Node node in nodes)
            {
                WriteView(w, node.PositionOffset, node.Vertices.Length * 4, 34962);
                WriteView(w, node.IndexOffset, node.Indices.Length * 4, 34963);
            }

            w.WriteEndArray();

            w.WriteStartArray("accessors");
            for (int i = 0; i < nodes.Count; i++)
            {
                Node node = nodes[i];
                w.WriteStartObject();
                w.WriteNumber("bufferView", i * 2);
                w.WriteNumber("componentType", 5126);
                w.WriteNumber("count", node.Vertices.Length / 3);
                w.WriteString("type", "VEC3");
                w.WriteStartArray("min");
                for (int a = 0; a < 3; a++)
                    w.WriteNumberValue(Bound(node.Vertices, a, false));
                w.WriteEndArray();
                w.WriteStartArray("max");
                for (int a = 0; a < 3; a++)
                    w.WriteNumberValue(Bound(node.Vertices, a, true));
                w.WriteEndArray();
                w.WriteEndObject();
                w.WriteStartObject();
                w.WriteNumber("bufferView", i * 2 + 1);
                w.WriteNumber("componentType", 5125);
                w.WriteNumber("count", node.Indices.Length);
                w.WriteString("type", "SCALAR");
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteStartArray("buffers");
            w.WriteStartObject();
            w.WriteNumber("byteLength", binLength);
            w.WriteEndObject();
            w.WriteEndArray();
            w.WriteEndObject();
        }

        return stream.ToArray();
    }

    // The accessor bounds are those of the vertices actually written (glTF requires them to be exact), not the part's box.
    private static float Bound(float[] vertices, int axis, bool upper)
    {
        float value = upper ? float.MinValue : float.MaxValue;
        for (int i = axis; i < vertices.Length; i += 3)
            value = upper ? Math.Max(value, vertices[i]) : Math.Min(value, vertices[i]);
        return value;
    }

    private static void WriteView(Utf8JsonWriter w, int offset, int length, int target)
    {
        w.WriteStartObject();
        w.WriteNumber("buffer", 0);
        w.WriteNumber("byteOffset", offset);
        w.WriteNumber("byteLength", length);
        w.WriteNumber("target", target);
        w.WriteEndObject();
    }

    private static byte[] Pad(byte[] data, byte fill)
    {
        int padded = (data.Length + 3) & ~3;
        if (padded == data.Length)
            return data;
        byte[] result = new byte[padded];
        data.CopyTo(result, 0);
        Array.Fill(result, fill, data.Length, padded - data.Length);
        return result;
    }

    private static void WriteU32(byte[] target, ref int position, uint value)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(target.AsSpan(position), value);
        position += 4;
    }
}

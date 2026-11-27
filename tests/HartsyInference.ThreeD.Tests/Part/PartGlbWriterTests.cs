using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.ThreeD.Tests.Part;

public sealed class PartGlbWriterTests
{
    // Two triangles apart in space plus a third that belongs to no part.
    private static Mesh ThreeTriangles() => new()
    {
        Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 5, 0, 0, 6, 0, 0, 5, 1, 2, 9, 9, 9, 10, 9, 9, 9, 10, 9],
        Indices = [0, 1, 2, 3, 4, 5, 6, 7, 8],
    };

    private static JsonElement ReadJson(byte[] glb, out int binLength)
    {
        Assert.Equal(0x46546C67u, BinaryPrimitives.ReadUInt32LittleEndian(glb));
        Assert.Equal((uint)glb.Length, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(8)));
        int jsonLength = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(12));
        JsonElement root = JsonDocument.Parse(glb.AsMemory(20, jsonLength)).RootElement.Clone();
        binLength = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(20 + jsonLength));
        Assert.Equal(glb.Length, 20 + jsonLength + 8 + binLength);
        return root;
    }

    [Fact]
    public void Write_SplitsTheMeshIntoOneColouredNodePerPartAndAnUnassignedNode()
    {
        float[][] boxes = [[5, 0, 0, 6, 1, 2], [0, 0, 0, 1, 1, 0]];

        byte[] glb = PartGlbWriter.Write(ThreeTriangles(), [7, 3, -1], [3, 7], boxes);

        JsonElement root = ReadJson(glb, out int binLength);
        JsonElement nodes = root.GetProperty("nodes");
        Assert.Equal(new[] { "part_3", "part_7", "unassigned" }, nodes.EnumerateArray().Select(n => n.GetProperty("name").GetString()!).ToArray());
        Assert.Equal(3, root.GetProperty("meshes").GetArrayLength());
        Assert.Equal(3, root.GetProperty("materials").GetArrayLength());
        JsonElement second = nodes[1].GetProperty("extras");
        Assert.Equal(7, second.GetProperty("partId").GetInt32());
        Assert.Equal(1, second.GetProperty("faceCount").GetInt32());
        Assert.Equal(boxes[1], second.GetProperty("aabb").EnumerateArray().Select(v => v.GetSingle()).ToArray());
        Assert.Equal(-1, nodes[2].GetProperty("extras").GetProperty("partId").GetInt32());

        // Each part owns its three vertices and its accessor bounds are those of the written vertices.
        JsonElement accessors = root.GetProperty("accessors");
        Assert.Equal(6, accessors.GetArrayLength());
        Assert.Equal(3, accessors[0].GetProperty("count").GetInt32());
        Assert.Equal([5f, 0f, 0f], accessors[0].GetProperty("min").EnumerateArray().Select(v => v.GetSingle()).ToArray());
        Assert.Equal([6f, 1f, 2f], accessors[0].GetProperty("max").EnumerateArray().Select(v => v.GetSingle()).ToArray());
        Assert.Equal(3 * 3 * 4 * 3 + 3 * 3 * 4, binLength);
    }

    [Fact]
    public void Write_ColoursDifferPerPartAndAreInUnitRange()
    {
        float[][] colours = [.. Enumerable.Range(0, 12).Select(PartGlbWriter.Color)];

        Assert.All(colours.SelectMany(c => c), v => Assert.InRange(v, 0f, 1f));
        Assert.Equal(12, colours.Select(c => string.Join(",", c)).Distinct().Count());
    }

    [Fact]
    public void Write_RejectsALabelCountThatDoesNotMatchTheFaces()
    {
        Assert.Throws<ArgumentException>(() => PartGlbWriter.Write(ThreeTriangles(), [0, 0], [0], [[0, 0, 0, 1, 1, 1]]));
        Assert.Throws<ArgumentException>(() => PartGlbWriter.Write(ThreeTriangles(), [0, 0, 0], [0], []));
    }

    [Fact]
    public void Write_SkipsAPartWithNoFaces()
    {
        byte[] glb = PartGlbWriter.Write(ThreeTriangles(), [0, 0, 0], [0, 99], [[0, 0, 0, 10, 10, 9], [0, 0, 0, 0, 0, 0]]);

        JsonElement root = ReadJson(glb, out _);
        Assert.Equal(new[] { "part_0" }, root.GetProperty("nodes").EnumerateArray().Select(n => n.GetProperty("name").GetString()!).ToArray());
    }
}

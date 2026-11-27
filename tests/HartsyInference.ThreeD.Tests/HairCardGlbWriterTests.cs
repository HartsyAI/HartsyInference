using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.ThreeD.Tests;

public sealed class HairCardGlbWriterTests
{
    private static readonly float[] HairSrgb = [0.4f, 0.3f, 0.25f];

    [Fact]
    public void Writes_hair_and_chain_nodes_with_reference_materials_and_attributes()
    {
        HairCardMesh cards = HairCardMesher.Build([StraightCard(5, 0), StraightCard(4, 1)], HairSrgb)!;
        Mesh chains = ChainTubeMesher.Build([[0f, 0f, 0f, 0f, 0f, 0.1f]])!;
        byte[] glb = HairCardGlbWriter.Write(
        [
            new HairGlbNode("hair_cards", cards.Mesh,
                new HairGlbMaterial("hair", cards.BaseColorLinear, 0f, 0.75f, true)),
            new HairGlbNode("spring_chains", chains,
                new HairGlbMaterial("spring_chains", [0.95f, 0.30f, 0.03f, 1f], 0f, 0.6f, false)),
        ]);

        (JsonDocument json, byte[] bin) = Parse(glb);
        using (json)
        {
            JsonElement root = json.RootElement;
            Assert.Equal("2.0", root.GetProperty("asset").GetProperty("version").GetString());
            Assert.Equal("hair_cards", root.GetProperty("nodes")[0].GetProperty("name").GetString());
            Assert.Equal("spring_chains", root.GetProperty("nodes")[1].GetProperty("name").GetString());

            JsonElement hair = root.GetProperty("materials")[0];
            Assert.Equal("hair", hair.GetProperty("name").GetString());
            Assert.Equal(0.0, hair.GetProperty("pbrMetallicRoughness").GetProperty("metallicFactor").GetDouble());
            Assert.Equal(0.75, hair.GetProperty("pbrMetallicRoughness").GetProperty("roughnessFactor").GetDouble(), 6);
            Assert.True(hair.GetProperty("doubleSided").GetBoolean());
            Assert.False(root.GetProperty("materials")[1].GetProperty("doubleSided").GetBoolean());

            JsonElement attributes = root.GetProperty("meshes")[0].GetProperty("primitives")[0].GetProperty("attributes");
            JsonElement accessors = root.GetProperty("accessors");
            int vertexCount = cards.Mesh.VertexCount;
            JsonElement color = accessors[attributes.GetProperty("COLOR_0").GetInt32()];
            Assert.Equal(5121, color.GetProperty("componentType").GetInt32());
            Assert.True(color.GetProperty("normalized").GetBoolean());
            Assert.Equal("VEC4", color.GetProperty("type").GetString());
            Assert.Equal(vertexCount, color.GetProperty("count").GetInt32());
            JsonElement uv = accessors[attributes.GetProperty("TEXCOORD_0").GetInt32()];
            Assert.Equal(5126, uv.GetProperty("componentType").GetInt32());
            Assert.Equal("VEC2", uv.GetProperty("type").GetString());
            Assert.Equal(vertexCount, uv.GetProperty("count").GetInt32());
            Assert.False(attributes.TryGetProperty("NORMAL", out _));

            JsonElement indices = accessors[root.GetProperty("meshes")[0].GetProperty("primitives")[0].GetProperty("indices").GetInt32()];
            Assert.Equal(5125, indices.GetProperty("componentType").GetInt32());
            Assert.Equal(cards.Mesh.Indices.Length, indices.GetProperty("count").GetInt32());
            Assert.Equal(bin.Length, root.GetProperty("buffers")[0].GetProperty("byteLength").GetInt32());
            foreach (JsonElement view in root.GetProperty("bufferViews").EnumerateArray())
                Assert.Equal(0, view.GetProperty("byteOffset").GetInt32() % 4);
        }
    }

    [Fact]
    public void Binary_payload_round_trips_positions_indices_and_colours()
    {
        HairCardMesh cards = HairCardMesher.Build([StraightCard(5, 0)], HairSrgb)!;
        byte[] glb = HairCardGlbWriter.Write([new HairGlbNode("hair_cards", cards.Mesh,
            new HairGlbMaterial("hair", cards.BaseColorLinear, 0f, 0.75f, true))]);

        (JsonDocument json, byte[] bin) = Parse(glb);
        using (json)
        {
            JsonElement root = json.RootElement;
            JsonElement primitive = root.GetProperty("meshes")[0].GetProperty("primitives")[0];
            JsonElement attributes = primitive.GetProperty("attributes");
            byte[] positions = ReadAccessor(root, bin, attributes.GetProperty("POSITION").GetInt32());
            for (int i = 0; i < cards.Mesh.Vertices.Length; i++)
                Assert.Equal(cards.Mesh.Vertices[i], BitConverter.ToSingle(positions, 4 * i));

            JsonElement positionAccessor = root.GetProperty("accessors")[attributes.GetProperty("POSITION").GetInt32()];
            float[] min = positionAccessor.GetProperty("min").EnumerateArray().Select(e => e.GetSingle()).ToArray();
            float[] max = positionAccessor.GetProperty("max").EnumerateArray().Select(e => e.GetSingle()).ToArray();
            for (int c = 0; c < 3; c++)
            {
                float[] axis = Enumerable.Range(0, cards.Mesh.VertexCount).Select(v => cards.Mesh.Vertices[3 * v + c]).ToArray();
                Assert.Equal(axis.Min(), min[c]);
                Assert.Equal(axis.Max(), max[c]);
            }

            byte[] indices = ReadAccessor(root, bin, primitive.GetProperty("indices").GetInt32());
            for (int i = 0; i < cards.Mesh.Indices.Length; i++)
                Assert.Equal((uint)cards.Mesh.Indices[i], BinaryPrimitives.ReadUInt32LittleEndian(indices.AsSpan(4 * i)));

            byte[] colors = ReadAccessor(root, bin, attributes.GetProperty("COLOR_0").GetInt32());
            for (int v = 0; v < cards.Mesh.VertexCount; v++)
            {
                for (int c = 0; c < 3; c++)
                {
                    byte expected = (byte)(cards.Mesh.VertexColors![3 * v + c] * 255f + 0.5f);
                    Assert.Equal(expected, colors[4 * v + c]);
                }
                Assert.Equal(255, colors[4 * v + 3]);
            }
        }
    }

    [Fact]
    public void Rejects_uv_count_that_does_not_match_the_vertices()
    {
        Mesh mesh = new()
        {
            Vertices = [0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f],
            Indices = [0, 1, 2],
            Uvs = [0f, 0f],
        };

        Assert.Throws<ArgumentException>(() => HairCardGlbWriter.Write(
            [new HairGlbNode("m", mesh, new HairGlbMaterial("m", [1f, 1f, 1f, 1f], 0f, 1f, false))]));
    }

    [Fact]
    public void Rejects_indices_outside_the_vertices()
    {
        Mesh mesh = new() { Vertices = [0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f], Indices = [0, 1, 3] };

        Assert.Throws<ArgumentException>(() => HairCardGlbWriter.Write(
            [new HairGlbNode("m", mesh, new HairGlbMaterial("m", [1f, 1f, 1f, 1f], 0f, 1f, false))]));
    }

    [Fact]
    public void Plain_mesh_writer_still_emits_positions_and_indices_only()
    {
        Mesh mesh = new() { Vertices = [0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f], Indices = [0, 1, 2] };

        byte[] glb = GlbWriter.Write(mesh);

        (JsonDocument json, _) = Parse(glb);
        using (json)
        {
            JsonElement attributes = json.RootElement.GetProperty("meshes")[0].GetProperty("primitives")[0]
                .GetProperty("attributes");
            Assert.True(attributes.TryGetProperty("POSITION", out _));
            Assert.False(attributes.TryGetProperty("COLOR_0", out _));
            Assert.False(attributes.TryGetProperty("TEXCOORD_0", out _));
        }
    }

    private static HairCard StraightCard(int points, int shape)
    {
        double[] seq = new double[points * 8];
        for (int i = 0; i < points; i++)
        {
            seq[8 * i] = 0.01 * i;
            seq[8 * i + 3] = 0.02;
            seq[8 * i + 4] = 0.004;
            seq[8 * i + 6] = 1.0;
        }
        return new HairCard(seq, shape, null);
    }

    private static (JsonDocument Json, byte[] Bin) Parse(byte[] glb)
    {
        Assert.Equal(0x46546C67u, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(0)));
        Assert.Equal(2u, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(4)));
        Assert.Equal((uint)glb.Length, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(8)));
        int jsonLength = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(12));
        Assert.Equal(0x4E4F534Au, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(16)));
        JsonDocument json = JsonDocument.Parse(glb.AsMemory(20, jsonLength));
        int binStart = 20 + jsonLength + 8;
        int binLength = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(20 + jsonLength));
        Assert.Equal(0x004E4942u, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(20 + jsonLength + 4)));
        return (json, glb.AsSpan(binStart, binLength).ToArray());
    }

    private static byte[] ReadAccessor(JsonElement root, byte[] bin, int accessorIndex)
    {
        JsonElement accessor = root.GetProperty("accessors")[accessorIndex];
        JsonElement view = root.GetProperty("bufferViews")[accessor.GetProperty("bufferView").GetInt32()];
        int offset = view.GetProperty("byteOffset").GetInt32();
        int length = view.GetProperty("byteLength").GetInt32();
        return bin.AsSpan(offset, length).ToArray();
    }
}

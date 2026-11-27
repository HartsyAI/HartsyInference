using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.ThreeD.Tests.Io;

public sealed class SkinnedGlbWriterTests
{
    private static Mesh Triangle() => new()
    {
        Vertices = [0, 0, 0, 1, 0, 0, 0, 2, 0],
        Indices = [0, 1, 2],
        Normals = [0, 0, 1, 0, 0, 1, 0, 0, 1],
        Uvs = [0, 0, 1, 0, 0, 1],
    };

    private static SkinnedBone[] Bones() =>
    [
        new("bone_0", -1, (0f, 0f, 0f), (0f, 1f, 0f)),
        new("bone_1", 0, (0f, 1f, 0f), (0f, 2f, 0f)),
        new("bone_2", 1, (0f, 2f, 0f), (0f, 3f, 0f)),
    ];

    private static (JsonDocument Json, byte[] Bin) Parse(byte[] glb)
    {
        Assert.Equal(0x46546C67u, BinaryPrimitives.ReadUInt32LittleEndian(glb));
        Assert.Equal(2u, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(4)));
        Assert.Equal((uint)glb.Length, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(8)));
        int jsonLength = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(12));
        Assert.Equal(0x4E4F534Au, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(16)));
        JsonDocument json = JsonDocument.Parse(glb.AsMemory(20, jsonLength));
        int binStart = 20 + jsonLength;
        int binLength = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(binStart));
        Assert.Equal(0x004E4942u, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(binStart + 4)));
        Assert.Equal(glb.Length, binStart + 8 + binLength);
        return (json, glb.AsSpan(binStart + 8, binLength).ToArray());
    }

    [Fact]
    public void Write_ProducesASkinWithJointsInverseBindMatricesAndWeights()
    {
        int[] joints = [0, 1, 0, 0, 1, 2, 0, 0, 2, 0, 0, 0];
        float[] weights = [0.75f, 0.25f, 0, 0, 0.5f, 0.5f, 0, 0, 1, 0, 0, 0];
        byte[] glb = SkinnedGlbWriter.Write(Triangle(), Bones(), joints, weights);
        (JsonDocument doc, byte[] bin) = Parse(glb);
        JsonElement root = doc.RootElement;

        JsonElement nodes = root.GetProperty("nodes");
        Assert.Equal(4, nodes.GetArrayLength());
        Assert.Equal("mesh", nodes[0].GetProperty("name").GetString());
        Assert.Equal(0, nodes[0].GetProperty("skin").GetInt32());
        Assert.Equal("bone_1", nodes[2].GetProperty("name").GetString());
        Assert.Equal(1f, nodes[2].GetProperty("translation")[1].GetSingle());
        Assert.Equal(3, nodes[1].GetProperty("children")[0].GetInt32() + 1);

        JsonElement skin = root.GetProperty("skins")[0];
        Assert.Equal(3, skin.GetProperty("joints").GetArrayLength());
        Assert.Equal(1, skin.GetProperty("skeleton").GetInt32());

        JsonElement accessors = root.GetProperty("accessors");
        JsonElement views = root.GetProperty("bufferViews");
        JsonElement attributes = root.GetProperty("meshes")[0].GetProperty("primitives")[0].GetProperty("attributes");
        JsonElement jointsAccessor = accessors[attributes.GetProperty("JOINTS_0").GetInt32()];
        Assert.Equal(5123, jointsAccessor.GetProperty("componentType").GetInt32());
        Assert.Equal("VEC4", jointsAccessor.GetProperty("type").GetString());
        JsonElement weightsAccessor = accessors[attributes.GetProperty("WEIGHTS_0").GetInt32()];
        int weightsOffset = views[weightsAccessor.GetProperty("bufferView").GetInt32()].GetProperty("byteOffset").GetInt32();
        float rowSum = 0f;
        for (int k = 0; k < 4; k++)
        {
            rowSum += BinaryPrimitives.ReadSingleLittleEndian(bin.AsSpan(weightsOffset + 4 * k));
        }

        Assert.Equal(1f, rowSum, 5);
        int jointsOffset = views[jointsAccessor.GetProperty("bufferView").GetInt32()].GetProperty("byteOffset").GetInt32();
        Assert.Equal(1, BinaryPrimitives.ReadUInt16LittleEndian(bin.AsSpan(jointsOffset + 2)));

        JsonElement bind = accessors[skin.GetProperty("inverseBindMatrices").GetInt32()];
        Assert.Equal("MAT4", bind.GetProperty("type").GetString());
        Assert.Equal(3, bind.GetProperty("count").GetInt32());
        int bindOffset = views[bind.GetProperty("bufferView").GetInt32()].GetProperty("byteOffset").GetInt32();
        // Bone 2's matrix is identity with translation (0, -2, 0) in column-major order.
        ReadOnlySpan<byte> m = bin.AsSpan(bindOffset + 2 * 64, 64);
        Assert.Equal(1f, BinaryPrimitives.ReadSingleLittleEndian(m[0..]));
        Assert.Equal(-2f, BinaryPrimitives.ReadSingleLittleEndian(m[52..]));
        Assert.Equal(1f, BinaryPrimitives.ReadSingleLittleEndian(m[60..]));
    }

    [Fact]
    public void Write_AccessorsStayInsideTheBinaryChunkAndAreAligned()
    {
        byte[] glb = SkinnedGlbWriter.Write(Triangle(), Bones(), new int[12], [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
        (JsonDocument doc, byte[] bin) = Parse(glb);
        foreach (JsonElement view in doc.RootElement.GetProperty("bufferViews").EnumerateArray())
        {
            int offset = view.GetProperty("byteOffset").GetInt32();
            Assert.Equal(0, offset % 4);
            Assert.True(offset + view.GetProperty("byteLength").GetInt32() <= bin.Length);
        }

        JsonElement position = doc.RootElement.GetProperty("accessors")[0];
        Assert.Equal(2f, position.GetProperty("max")[1].GetSingle());
        Assert.Equal(0f, position.GetProperty("min")[1].GetSingle());
    }

    [Fact]
    public void Write_RejectsInconsistentInput()
    {
        Assert.Throws<HartsyInferenceException>(() => SkinnedGlbWriter.Write(Triangle(), Bones(), new int[4], new float[12]));
        Assert.Throws<HartsyInferenceException>(() => SkinnedGlbWriter.Write(Triangle(), [], new int[12], new float[12]));
        Assert.Throws<HartsyInferenceException>(() => SkinnedGlbWriter.Write(Triangle(), Bones(), [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0], new float[12]));
        SkinnedBone[] bad = [new("a", 1, (0, 0, 0), (0, 0, 0)), new("b", -1, (0, 0, 0), (0, 0, 0))];
        Assert.Throws<HartsyInferenceException>(() => SkinnedGlbWriter.Write(Triangle(), bad, new int[12], new float[12]));
    }
}

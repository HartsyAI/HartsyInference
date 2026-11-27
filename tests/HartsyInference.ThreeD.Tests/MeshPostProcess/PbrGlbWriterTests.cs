using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.Vision.Codec;

namespace HartsyInference.ThreeD.Tests.MeshPostProcess;

/// <summary>Unit lane, synthetic mesh and textures: the produced GLB is parsed back chunk by chunk and checked against glTF 2.0 rules.</summary>
public sealed class PbrGlbWriterTests
{
    private const int Tex = 4;

    private static Mesh TexturedTriangle() => new()
    {
        Vertices = [0, 0, 0, 2, 0, 0, 0, 3, 1],
        Indices = [0, 1, 2],
        Normals = [0, 0, 1, 0, 0, 1, 0, 0, 1],
        Uvs = [0.1f, 0.2f, 0.9f, 0.2f, 0.1f, 0.8f],
    };

    private static PbrMaterial Material(bool withAlpha)
    {
        byte[] color = new byte[Tex * Tex * 3], metal = new byte[Tex * Tex], rough = new byte[Tex * Tex], alpha = new byte[Tex * Tex];
        for (int i = 0; i < Tex * Tex; i++)
        {
            color[i * 3] = (byte)(i * 10); color[i * 3 + 1] = (byte)(255 - i * 10); color[i * 3 + 2] = (byte)(i * 3);
            metal[i] = (byte)(i * 7); rough[i] = (byte)(200 - i * 5); alpha[i] = (byte)(i * 16);
        }
        return new PbrMaterial
        {
            BaseColor = new TextureImage(Tex, Tex, 3, color),
            Metallic = new TextureImage(Tex, Tex, 1, metal),
            Roughness = new TextureImage(Tex, Tex, 1, rough),
            Alpha = withAlpha ? new TextureImage(Tex, Tex, 1, alpha) : null,
        };
    }

    private sealed record Parsed(JsonElement Root, byte[] Bin, int ContainerLength);

    private static Parsed Parse(byte[] glb)
    {
        Assert.Equal(0x46546C67u, BinaryPrimitives.ReadUInt32LittleEndian(glb));
        Assert.Equal(2u, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(4)));
        Assert.Equal((uint)glb.Length, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(8)));
        int jsonLength = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(12));
        Assert.Equal(0x4E4F534Au, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(16)));
        Assert.Equal(0, jsonLength % 4);
        JsonElement root = JsonDocument.Parse(glb.AsMemory(20, jsonLength)).RootElement.Clone();
        int binHeader = 20 + jsonLength;
        int binLength = (int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(binHeader));
        Assert.Equal(0x004E4942u, BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(binHeader + 4)));
        Assert.Equal(0, binLength % 4);
        Assert.Equal(binHeader + 8 + binLength, glb.Length);
        return new Parsed(root, glb.AsSpan(binHeader + 8, binLength).ToArray(), glb.Length);
    }

    private static float[] ReadFloats(Parsed p, int accessor)
    {
        JsonElement acc = p.Root.GetProperty("accessors")[accessor];
        JsonElement view = p.Root.GetProperty("bufferViews")[acc.GetProperty("bufferView").GetInt32()];
        int components = acc.GetProperty("type").GetString() switch { "VEC3" => 3, "VEC2" => 2, _ => 1 };
        int count = acc.GetProperty("count").GetInt32() * components;
        int offset = view.GetProperty("byteOffset").GetInt32();
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = BinaryPrimitives.ReadSingleLittleEndian(p.Bin.AsSpan(offset + i * 4));
        return values;
    }

    private static byte[] ReadImage(Parsed p, int image)
    {
        JsonElement view = p.Root.GetProperty("bufferViews")[p.Root.GetProperty("images")[image].GetProperty("bufferView").GetInt32()];
        return p.Bin.AsSpan(view.GetProperty("byteOffset").GetInt32(), view.GetProperty("byteLength").GetInt32()).ToArray();
    }

    [Fact]
    public void Container_HasConsistentAccessorsViewsAndAlignment()
    {
        Parsed p = Parse(PbrGlbWriter.Write(TexturedTriangle(), Material(false)));

        JsonElement root = p.Root;
        Assert.Equal("2.0", root.GetProperty("asset").GetProperty("version").GetString());
        JsonElement prim = root.GetProperty("meshes")[0].GetProperty("primitives")[0];
        Assert.Equal(4, prim.GetProperty("mode").GetInt32());
        Assert.Equal(0, prim.GetProperty("material").GetInt32());
        JsonElement attrs = prim.GetProperty("attributes");
        // POSITION, NORMAL, TEXCOORD_0 and indices each get an accessor on their own view.
        Assert.Equal(0, attrs.GetProperty("POSITION").GetInt32());
        Assert.Equal(1, attrs.GetProperty("NORMAL").GetInt32());
        Assert.Equal(2, attrs.GetProperty("TEXCOORD_0").GetInt32());
        Assert.Equal(3, prim.GetProperty("indices").GetInt32());
        Assert.Equal(4, root.GetProperty("accessors").GetArrayLength());
        Assert.Equal(6, root.GetProperty("bufferViews").GetArrayLength());
        Assert.Equal(p.Bin.Length, root.GetProperty("buffers")[0].GetProperty("byteLength").GetInt32());

        foreach (JsonElement view in root.GetProperty("bufferViews").EnumerateArray())
        {
            int offset = view.GetProperty("byteOffset").GetInt32(), length = view.GetProperty("byteLength").GetInt32();
            Assert.Equal(0, offset % 4);
            Assert.InRange(offset + length, 0, p.Bin.Length);
        }
        foreach (JsonElement acc in root.GetProperty("accessors").EnumerateArray())
        {
            JsonElement view = root.GetProperty("bufferViews")[acc.GetProperty("bufferView").GetInt32()];
            int components = acc.GetProperty("type").GetString() switch { "VEC3" => 3, "VEC2" => 2, _ => 1 };
            int bytes = acc.GetProperty("count").GetInt32() * components * 4;
            Assert.Equal(bytes, view.GetProperty("byteLength").GetInt32());
        }
        JsonElement position = root.GetProperty("accessors")[0];
        Assert.Equal(new[] { 0.0, 0.0, 0.0 }, position.GetProperty("min").EnumerateArray().Select(e => e.GetDouble()).ToArray());
        Assert.Equal(new[] { 2.0, 3.0, 1.0 }, position.GetProperty("max").EnumerateArray().Select(e => e.GetDouble()).ToArray());
    }

    [Fact]
    public void Geometry_RoundTripsThroughGlbReader()
    {
        Mesh mesh = TexturedTriangle();

        Mesh read = GlbReader.Parse(PbrGlbWriter.Write(mesh, Material(false)));

        Assert.Equal(mesh.Vertices, read.Vertices);
        Assert.Equal(mesh.Indices, read.Indices);
    }

    [Fact]
    public void Attributes_AreWrittenAsGiven()
    {
        Mesh mesh = TexturedTriangle();

        Parsed p = Parse(PbrGlbWriter.Write(mesh, Material(false)));

        Assert.Equal(mesh.Vertices, ReadFloats(p, 0));
        Assert.Equal(mesh.Normals, ReadFloats(p, 1));
        Assert.Equal(mesh.Uvs, ReadFloats(p, 2));
    }

    [Fact]
    public void Material_UsesOpaqueModeAndTwoTexturesWithPackedMetallicRoughness()
    {
        PbrMaterial material = Material(false);

        Parsed p = Parse(PbrGlbWriter.Write(TexturedTriangle(), material));

        JsonElement mat = p.Root.GetProperty("materials")[0];
        Assert.Equal(material.Name, mat.GetProperty("name").GetString());
        Assert.Equal("OPAQUE", mat.GetProperty("alphaMode").GetString());
        Assert.False(mat.GetProperty("doubleSided").GetBoolean());
        JsonElement pbr = mat.GetProperty("pbrMetallicRoughness");
        Assert.Equal(1.0, pbr.GetProperty("metallicFactor").GetDouble());
        Assert.Equal(1.0, pbr.GetProperty("roughnessFactor").GetDouble());
        int colorTexture = pbr.GetProperty("baseColorTexture").GetProperty("index").GetInt32();
        int mrTexture = pbr.GetProperty("metallicRoughnessTexture").GetProperty("index").GetInt32();
        JsonElement textures = p.Root.GetProperty("textures");
        Assert.Equal(2, textures.GetArrayLength());
        Assert.Equal("image/png", p.Root.GetProperty("images")[textures[colorTexture].GetProperty("source").GetInt32()]
            .GetProperty("mimeType").GetString());

        DecodedPng color = PngDecoder.DecodeWithAlpha(ReadImage(p, textures[colorTexture].GetProperty("source").GetInt32()));
        Assert.Equal((Tex, Tex), (color.Width, color.Height));
        Assert.Equal(material.BaseColor.Pixels, color.Rgb);
        Assert.Null(color.Alpha);

        DecodedPng mr = PngDecoder.DecodeWithAlpha(ReadImage(p, textures[mrTexture].GetProperty("source").GetInt32()));
        for (int i = 0; i < Tex * Tex; i++)
        {
            Assert.Equal(0, mr.Rgb[i * 3]);
            Assert.Equal(material.Roughness.Pixels[i], mr.Rgb[i * 3 + 1]);
            Assert.Equal(material.Metallic.Pixels[i], mr.Rgb[i * 3 + 2]);
        }
    }

    [Fact]
    public void Alpha_GoesIntoTheBaseColorTextureAndTheModeIsConfigurable()
    {
        PbrMaterial material = Material(true);

        Parsed p = Parse(PbrGlbWriter.Write(TexturedTriangle(), material,
            new PbrGlbOptions { AlphaMode = PbrAlphaMode.Mask, AlphaCutoff = 0.25f, DoubleSided = true }));

        JsonElement mat = p.Root.GetProperty("materials")[0];
        Assert.Equal("MASK", mat.GetProperty("alphaMode").GetString());
        Assert.Equal(0.25, mat.GetProperty("alphaCutoff").GetDouble(), 6);
        Assert.True(mat.GetProperty("doubleSided").GetBoolean());
        DecodedPng color = PngDecoder.DecodeWithAlpha(ReadImage(p, 0));
        Assert.Equal(material.BaseColor.Pixels, color.Rgb);
        Assert.Equal(material.Alpha!.Pixels, color.Alpha);
    }

    [Fact]
    public void ZUpConversion_MapsXyzToXZMinusYOnPositionsAndNormalsAndKeepsWinding()
    {
        Mesh mesh = TexturedTriangle();

        Parsed p = Parse(PbrGlbWriter.Write(mesh, Material(false), new PbrGlbOptions { ConvertZUpToYUp = true }));

        float[] positions = ReadFloats(p, 0), normals = ReadFloats(p, 1);
        for (int v = 0; v < 3; v++)
        {
            Assert.Equal(mesh.Vertices[v * 3], positions[v * 3]);
            Assert.Equal(mesh.Vertices[v * 3 + 2], positions[v * 3 + 1]);
            Assert.Equal(-mesh.Vertices[v * 3 + 1], positions[v * 3 + 2]);
            Assert.Equal(mesh.Normals![v * 3 + 2], normals[v * 3 + 1]);
            Assert.Equal(-mesh.Normals[v * 3 + 1], normals[v * 3 + 2]);
        }
        JsonElement max = p.Root.GetProperty("accessors")[0].GetProperty("max");
        Assert.Equal(1.0, max[1].GetDouble());
        Assert.Equal(0.0, max[2].GetDouble());
        Mesh read = GlbReader.Parse(PbrGlbWriter.Write(mesh, Material(false), new PbrGlbOptions { ConvertZUpToYUp = true }));
        Assert.Equal(mesh.Indices, read.Indices);
    }

    [Fact]
    public void FlipV_WritesOneMinusV()
    {
        Mesh mesh = TexturedTriangle();

        Parsed p = Parse(PbrGlbWriter.Write(mesh, Material(false), new PbrGlbOptions { FlipV = true }));

        float[] uvs = ReadFloats(p, 2);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(mesh.Uvs![i * 2], uvs[i * 2]);
            Assert.Equal(1f - mesh.Uvs[i * 2 + 1], uvs[i * 2 + 1]);
        }
    }

    [Fact]
    public void MissingNormals_AreComputed()
    {
        Mesh mesh = new() { Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0], Indices = [0, 1, 2], Uvs = [0, 0, 1, 0, 0, 1] };

        Parsed p = Parse(PbrGlbWriter.Write(mesh, Material(false)));

        Assert.Equal(new float[] { 0, 0, 1, 0, 0, 1, 0, 0, 1 }, ReadFloats(p, 1));
        Assert.Null(mesh.Normals);
    }

    [Fact]
    public void RejectsMeshesAndMaterialsThatCannotBeWritten()
    {
        Mesh noUv = new() { Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0], Indices = [0, 1, 2] };
        Assert.Throws<ArgumentException>(() => PbrGlbWriter.Write(noUv, Material(false)));

        PbrMaterial mismatched = Material(false) with { Roughness = new TextureImage(2, 2, 1, new byte[4]) };
        Assert.Throws<ArgumentException>(() => PbrGlbWriter.Write(TexturedTriangle(), mismatched));

        Mesh badIndex = TexturedTriangle();
        badIndex.Indices[2] = 9;
        Assert.Throws<ArgumentException>(() => PbrGlbWriter.Write(badIndex, Material(false)));
    }
}

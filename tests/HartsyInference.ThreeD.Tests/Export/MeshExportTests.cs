using System.Buffers.Binary;
using System.Text;
using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Export;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.ThreeD.Tests.Export;

/// <summary>The document model and its exporters: what each format carries, what it reports dropping, and that a glb survives a write and a read.</summary>
public sealed class MeshExportTests
{
    private static MeshImage Png(byte seed) =>
        new("image/png", PngTextureEncoder.Encode(new TextureImage(2, 2, 3, [.. Enumerable.Range(0, 12).Select(i => (byte)(seed + i))])));

    private static MeshDocument Sample(bool hadMotion = false)
    {
        MeshImage baseColor = Png(10), normal = Png(40), metalRough = Png(80);
        Mesh quad = new()
        {
            Vertices = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
            Indices = [0, 1, 2, 0, 2, 3],
            Normals = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            Uvs = [0, 0, 1, 0, 1, 1, 0, 1],
            VertexColors = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        };
        Mesh triangle = new()
        {
            Vertices = [2, 0, 0, 3, 0, 0, 2, 1, 0],
            Indices = [0, 1, 2],
            Normals = [0, 0, 1, 0, 0, 1, 0, 0, 1],
            Uvs = [0, 0, 1, 0, 0, 1],
            VertexColors = [0, 0, 0, 1, 1, 1, 0.5f, 0.5f, 0.5f],
        };
        return new MeshDocument
        {
            Primitives = [new MeshPrimitive { Geometry = quad, Material = 0 }, new MeshPrimitive { Geometry = triangle, Material = 1 }],
            Materials =
            [
                new MeshMaterial { Name = "skin", BaseColor = baseColor, Normal = normal, MetallicRoughness = metalRough, Metallic = 0.2f, Roughness = 0.7f },
                new MeshMaterial { BaseColorFactor = [1f, 0f, 0f, 0.5f], AlphaMode = MeshAlphaMode.Blend, DoubleSided = true },
            ],
            Animations = hadMotion ? ["walk"] : [],
            HasSkin = hadMotion,
            AssetExtrasJson = """{"hartsyinference":{"seed":7}}""",
        };
    }

    [Fact]
    public void AGlbKeepsGeometryAttributesMaterialsImagesAndItsExtrasThroughAWriteAndARead()
    {
        MeshDocument source = Sample();

        MeshDocument read = GlbReader.ParseDocument(MeshDocumentGlbWriter.Write(source));

        Assert.Equal(2, read.Primitives.Count);
        Assert.Equal(source.Primitives[0].Geometry.Vertices, read.Primitives[0].Geometry.Vertices);
        Assert.Equal(source.Primitives[0].Geometry.Indices, read.Primitives[0].Geometry.Indices);
        Assert.Equal(source.Primitives[0].Geometry.Uvs, read.Primitives[0].Geometry.Uvs);
        Assert.Equal(source.Primitives[1].Geometry.VertexColors, read.Primitives[1].Geometry.VertexColors);
        Assert.Equal(2, read.Materials.Count);
        Assert.Equal(source.Materials[0].BaseColor!.Data, read.Materials[0].BaseColor!.Data);
        Assert.Equal(source.Materials[0].Normal!.Data, read.Materials[0].Normal!.Data);
        Assert.Equal(source.Materials[0].MetallicRoughness!.Data, read.Materials[0].MetallicRoughness!.Data);
        Assert.Equal((0.2f, 0.7f, "skin"), (read.Materials[0].Metallic, read.Materials[0].Roughness, read.Materials[0].Name));
        Assert.Equal(MeshAlphaMode.Blend, read.Materials[1].AlphaMode);
        Assert.True(read.Materials[1].DoubleSided);
        Assert.Equal(1, read.Primitives[1].Material);
        Assert.Contains("\"seed\":7", read.AssetExtrasJson);
    }

    [Fact]
    public void ADocumentMergeKeepsAnAttributeOnlyWhenEveryPrimitiveHasIt()
    {
        MeshDocument source = Sample();
        source.Primitives[1].Geometry.Uvs = null;

        Mesh merged = source.Merge();

        Assert.Null(merged.Uvs);
        Assert.NotNull(merged.Normals);
        Assert.Equal(7, merged.VertexCount);
        Assert.Equal([0, 1, 2, 0, 2, 3, 4, 5, 6], merged.Indices);
    }

    private static byte[] RawGlb(string extraRoot, string primitives)
    {
        string json = $$"""
            {"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
             "meshes":[{"primitives":[{{primitives}}]}],
             "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}],
             "bufferViews":[{"buffer":0,"byteLength":36}],"buffers":[{"byteLength":36}]{{extraRoot}}}
            """;
        byte[] bin = new byte[36];
        BinaryPrimitives.WriteSingleLittleEndian(bin.AsSpan(12), 1f);
        BinaryPrimitives.WriteSingleLittleEndian(bin.AsSpan(28), 1f);
        return GlbContainer.Assemble(Encoding.UTF8.GetBytes(json), bin);
    }

    [Fact]
    public void APrimitiveWithoutIndicesIsReadAsATriangleSoup()
    {
        byte[] glb = RawGlb("", """{"attributes":{"POSITION":0}}""");

        MeshDocument document = GlbReader.ParseDocument(glb);

        Assert.Equal([0, 1, 2], document.Primitives.Single().Geometry.Indices);
        Assert.Equal([0, 1, 2], GlbReader.Parse(glb).Indices);
    }

    [Fact]
    public void AnAnimatedSkinnedFileIsReadAsItsRestPoseAndSaysWhatWasLeftOut()
    {
        byte[] glb = RawGlb(""","skins":[{"joints":[0]}],"animations":[{"name":"walk"},{}]""", """{"attributes":{"POSITION":0}},{"attributes":{"POSITION":0},"mode":5}""");

        MeshDocument document = GlbReader.ParseDocument(glb);
        MeshExport export = new StlExporter().Export(document, "m");

        Assert.True(document.HasSkin);
        Assert.Equal(["walk", "animation_1"], document.Animations);
        Assert.Contains(document.Notes, n => n.Contains("strip, fan, line or point mode"));
        Assert.Contains(export.Notes, n => n.StartsWith("Static geometry only") && n.Contains("skin") && n.Contains("2 animation"));
    }

    [Fact]
    public void ObjExportWritesTheObjItsMtlAndTheImagesAndReadsBack()
    {
        MeshDocument source = Sample();

        MeshExport export = new ObjExporter().Export(source, "My Model!");

        Assert.Equal(["My_Model_.obj", "My_Model_.mtl"], export.Files.Take(2).Select(f => f.Name));
        string obj = Encoding.UTF8.GetString(export.Files[0].Data);
        string mtl = Encoding.UTF8.GetString(export.Files[1].Data);
        Assert.Contains("mtllib My_Model_.mtl", obj);
        Assert.Contains("usemtl material_1", obj);
        Assert.Contains("vt 0 1", obj);
        Assert.Contains("f 1/1/1 2/2/2 3/3/3", obj);
        Assert.Contains("f 5/5/5 6/6/6 7/7/7", obj);
        Assert.Contains("map_Kd My_Model__0_basecolor.png", mtl);
        Assert.Contains("norm My_Model__0_normal.png", mtl);
        Assert.Contains("d 0.5", mtl);
        Assert.Equal(source.Materials[0].BaseColor!.Data, export.Files.Single(f => f.Name == "My_Model__0_basecolor.png").Data);
        Assert.Contains(export.Notes, n => n.Contains("Metallic-roughness"));
        Assert.Contains(export.Notes, n => n.Contains("Vertex colours were dropped"));

        MeshDocument again = ObjDocumentReader.Parse(obj);
        Assert.Equal(source.Merge().Vertices, again.Merge().Vertices);
        Assert.Equal(source.Merge().Uvs, again.Merge().Uvs);
    }

    [Fact]
    public void PlyExportIsBinaryWithNormalsAndColoursAndSaysTexturesWereDropped()
    {
        MeshExport export = new PlyExporter().Export(Sample(), "m");

        byte[] data = export.Files.Single().Data;
        string header = Encoding.ASCII.GetString(data, 0, Encoding.ASCII.GetString(data).IndexOf("end_header\n", StringComparison.Ordinal) + 11);
        Assert.Contains("format binary_little_endian 1.0", header);
        Assert.Contains("element vertex 7", header);
        Assert.Contains("property float nx", header);
        Assert.Contains("property uchar red", header);
        Assert.Contains("element face 3", header);
        Assert.Equal(header.Length + 7 * (12 + 12 + 3) + 3 * 13, data.Length);
        Assert.Contains(export.Notes, n => n.StartsWith("Textures were dropped"));
    }

    [Fact]
    public void StlExportHasOneRecordPerTriangleWithAFacetNormal()
    {
        MeshExport export = new StlExporter().Export(Sample(), "m");

        byte[] data = export.Files.Single().Data;
        Assert.Equal(84 + 3 * 50, data.Length);
        Assert.Equal(3u, BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(80)));
        Assert.Equal(1f, BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(84 + 8)), 4);
        Assert.Contains(export.Notes, n => n.Contains("geometry only"));
    }

    [Fact]
    public void ASourceThatMovedExportsStaticGeometryAndSaysSoForEveryStaticFormat()
    {
        MeshDocument moving = Sample(hadMotion: true);

        foreach (IMeshExporter exporter in MeshExporters.All)
            Assert.Contains(exporter.Export(moving, "m").Notes, n => n.StartsWith("Static geometry only"));
    }

    [Fact]
    public void TheRegistryFindsAFormatByIdAndNamesTheSupportedOnesForAnUnknownId()
    {
        Assert.Equal(["obj", "ply", "stl", "glb"], MeshExporters.All.Select(e => e.Id));
        Assert.Equal("stl", MeshExporters.Get("STL").Id);

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => MeshExporters.Get("fbx"));

        Assert.Contains("obj, ply, stl, glb", error.Message);
    }

    [Fact]
    public void ADocumentFromAnObjWithoutNormalsExportsToGlbWithoutThem()
    {
        MeshDocument obj = ObjDocumentReader.Parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");

        MeshDocument glb = GlbReader.ParseDocument(new GlbExporter().Export(obj, "m").Files.Single().Data);

        Assert.Null(glb.Primitives.Single().Geometry.Normals);
        Assert.Equal(1, glb.TriangleCount);
    }
}

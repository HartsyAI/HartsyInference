using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Io;

/// <summary>
/// Writes one textured mesh as binary glTF 2.0: POSITION, NORMAL, TEXCOORD_0 and indices, plus a metallic-roughness material whose two
/// textures are embedded PNGs. Base color carries RGB (and alpha in the fourth channel when the material has one); the second texture
/// packs <c>R = 0, G = roughness, B = metallic</c>. Additive to <see cref="GlbWriter"/> and <see cref="HairCardGlbWriter"/>, which are untouched.
/// </summary>
public static class PbrGlbWriter
{
    private const int ArrayBuffer = 34962;
    private const int ElementArrayBuffer = 34963;
    private const int ComponentFloat = 5126;
    private const int ComponentUnsignedInt = 5125;

    /// <summary>Serializes <paramref name="mesh"/> (needs <see cref="Mesh.Uvs"/>) with <paramref name="material"/> to a GLB byte array.</summary>
    public static byte[] Write(Mesh mesh, PbrMaterial material, PbrGlbOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        ArgumentNullException.ThrowIfNull(material);
        options ??= new PbrGlbOptions();
        Validate(mesh, material);

        int vertexCount = mesh.VertexCount;
        float[] positions = (float[])mesh.Vertices.Clone();
        float[] normals = mesh.Normals is { } n && n.Length == positions.Length
            ? (float[])n.Clone()
            : MeshOps.ComputeVertexNormals(new Mesh { Vertices = mesh.Vertices, Indices = mesh.Indices }).Normals!;
        float[] uvs = (float[])mesh.Uvs!.Clone();
        if (options.ConvertZUpToYUp)
        {
            ZUpToYUp(positions);
            ZUpToYUp(normals);
        }
        if (options.FlipV)
            for (int i = 1; i < uvs.Length; i += 2) uvs[i] = 1f - uvs[i];

        byte[] baseColor = PngTextureEncoder.Encode(CombineBaseColor(material));
        byte[] metallicRoughness = PngTextureEncoder.Encode(PackMetallicRoughness(material));

        List<(int Offset, int Length)> views = [];
        using MemoryStream bin = new();
        AddView(bin, views, FloatBytes(positions));
        AddView(bin, views, FloatBytes(normals));
        AddView(bin, views, FloatBytes(uvs));
        AddView(bin, views, IndexBytes(mesh.Indices));
        AddView(bin, views, baseColor);
        AddView(bin, views, metallicRoughness);

        byte[] json = BuildJson(options, views, vertexCount, mesh.Indices.Length, positions);
        return GlbContainer.Assemble(json, bin.ToArray());
    }

    /// <summary>Writes the GLB to <paramref name="path"/>.</summary>
    public static void Save(string path, Mesh mesh, PbrMaterial material, PbrGlbOptions? options = null) =>
        File.WriteAllBytes(path, Write(mesh, material, options));

    private static void Validate(Mesh mesh, PbrMaterial material)
    {
        int vertexCount = mesh.VertexCount;
        if (vertexCount == 0 || mesh.TriangleCount == 0 || mesh.Vertices.Length % 3 != 0 || mesh.Indices.Length % 3 != 0)
            throw new ArgumentException("The mesh needs whole vertices and at least one triangle.", nameof(mesh));
        if (mesh.Uvs is null || mesh.Uvs.Length != vertexCount * 2)
            throw new ArgumentException("The mesh needs one UV per vertex.", nameof(mesh));
        foreach (float v in mesh.Vertices)
            if (!float.IsFinite(v)) throw new ArgumentException("Vertex positions must be finite.", nameof(mesh));
        foreach (float v in mesh.Uvs)
            if (!float.IsFinite(v)) throw new ArgumentException("UVs must be finite.", nameof(mesh));
        foreach (int i in mesh.Indices)
            if ((uint)i >= (uint)vertexCount) throw new ArgumentException("A triangle index lies outside the vertices.", nameof(mesh));
        if (material.BaseColor.Channels != 3)
            throw new ArgumentException("Base color needs three channels.", nameof(material));
        if (material.Metallic.Channels != 1 || material.Roughness.Channels != 1 || material.Alpha is { Channels: not 1 })
            throw new ArgumentException("Metallic, roughness and alpha need one channel each.", nameof(material));
        if (material.Metallic.Width != material.Roughness.Width || material.Metallic.Height != material.Roughness.Height)
            throw new ArgumentException("Metallic and roughness must have the same size.", nameof(material));
        if (material.Alpha is { } a && (a.Width != material.BaseColor.Width || a.Height != material.BaseColor.Height))
            throw new ArgumentException("Alpha must match the base color size.", nameof(material));
    }

    private static TextureImage CombineBaseColor(PbrMaterial material)
    {
        if (material.Alpha is not { } alpha)
            return material.BaseColor;
        int count = material.BaseColor.Width * material.BaseColor.Height;
        byte[] rgba = new byte[count * 4];
        for (int i = 0; i < count; i++)
        {
            rgba[i * 4] = material.BaseColor.Pixels[i * 3];
            rgba[i * 4 + 1] = material.BaseColor.Pixels[i * 3 + 1];
            rgba[i * 4 + 2] = material.BaseColor.Pixels[i * 3 + 2];
            rgba[i * 4 + 3] = alpha.Pixels[i];
        }
        return new TextureImage(material.BaseColor.Width, material.BaseColor.Height, 4, rgba);
    }

    private static TextureImage PackMetallicRoughness(PbrMaterial material)
    {
        int count = material.Metallic.Width * material.Metallic.Height;
        byte[] rgb = new byte[count * 3];
        for (int i = 0; i < count; i++)
        {
            rgb[i * 3] = 0;
            rgb[i * 3 + 1] = material.Roughness.Pixels[i];
            rgb[i * 3 + 2] = material.Metallic.Pixels[i];
        }
        return new TextureImage(material.Metallic.Width, material.Metallic.Height, 3, rgb);
    }

    private static void ZUpToYUp(float[] xyz)
    {
        for (int i = 0; i < xyz.Length; i += 3)
        {
            float y = xyz[i + 1], z = xyz[i + 2];
            xyz[i + 1] = z;
            xyz[i + 2] = -y;
        }
    }

    private static byte[] FloatBytes(float[] values)
    {
        byte[] bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        return bytes;
    }

    private static byte[] IndexBytes(int[] indices)
    {
        byte[] bytes = new byte[indices.Length * 4];
        for (int i = 0; i < indices.Length; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(i * 4), (uint)indices[i]);
        return bytes;
    }

    private static void AddView(MemoryStream bin, List<(int Offset, int Length)> views, byte[] data)
    {
        int offset = (int)bin.Position;
        bin.Write(data);
        views.Add((offset, data.Length));
        for (int pad = GlbContainer.Align(data.Length) - data.Length; pad > 0; pad--)
            bin.WriteByte(0);
    }

    private static byte[] BuildJson(PbrGlbOptions options, List<(int Offset, int Length)> views, int vertexCount, int indexCount,
        float[] positions)
    {
        using MemoryStream ms = new();
        using (Utf8JsonWriter w = new(ms))
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
            w.WriteNumberValue(0);
            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();
            w.WriteStartArray("nodes");
            w.WriteStartObject();
            w.WriteString("name", options.MeshName);
            w.WriteNumber("mesh", 0);
            w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("meshes");
            w.WriteStartObject();
            w.WriteString("name", options.MeshName);
            w.WriteStartArray("primitives");
            w.WriteStartObject();
            w.WriteStartObject("attributes");
            w.WriteNumber("POSITION", 0);
            w.WriteNumber("NORMAL", 1);
            w.WriteNumber("TEXCOORD_0", 2);
            w.WriteEndObject();
            w.WriteNumber("indices", 3);
            w.WriteNumber("material", 0);
            w.WriteNumber("mode", 4);
            w.WriteEndObject();
            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("materials");
            w.WriteStartObject();
            w.WriteString("name", "pbr");
            w.WriteBoolean("doubleSided", options.DoubleSided);
            w.WriteString("alphaMode", options.AlphaMode switch
            {
                PbrAlphaMode.Mask => "MASK",
                PbrAlphaMode.Blend => "BLEND",
                _ => "OPAQUE",
            });
            if (options.AlphaMode == PbrAlphaMode.Mask)
                w.WriteNumber("alphaCutoff", options.AlphaCutoff);
            w.WriteStartObject("pbrMetallicRoughness");
            w.WriteStartObject("baseColorTexture");
            w.WriteNumber("index", 0);
            w.WriteEndObject();
            w.WriteStartObject("metallicRoughnessTexture");
            w.WriteNumber("index", 1);
            w.WriteEndObject();
            w.WriteNumber("metallicFactor", 1.0);
            w.WriteNumber("roughnessFactor", 1.0);
            w.WriteEndObject();
            w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("textures");
            for (int t = 0; t < 2; t++)
            {
                w.WriteStartObject();
                w.WriteNumber("sampler", 0);
                w.WriteNumber("source", t);
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteStartArray("images");
            for (int t = 0; t < 2; t++)
            {
                w.WriteStartObject();
                w.WriteNumber("bufferView", 4 + t);
                w.WriteString("mimeType", "image/png");
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteStartArray("samplers");
            w.WriteStartObject();
            w.WriteNumber("magFilter", 9729);
            w.WriteNumber("minFilter", 9987);
            w.WriteNumber("wrapS", 33071);
            w.WriteNumber("wrapT", 33071);
            w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("bufferViews");
            for (int i = 0; i < views.Count; i++)
            {
                w.WriteStartObject();
                w.WriteNumber("buffer", 0);
                w.WriteNumber("byteOffset", views[i].Offset);
                w.WriteNumber("byteLength", views[i].Length);
                if (i < 3) w.WriteNumber("target", ArrayBuffer);
                else if (i == 3) w.WriteNumber("target", ElementArrayBuffer);
                w.WriteEndObject();
            }
            w.WriteEndArray();

            (float[] min, float[] max) = Bounds(positions);
            w.WriteStartArray("accessors");
            WriteAccessor(w, 0, ComponentFloat, vertexCount, "VEC3", min, max);
            WriteAccessor(w, 1, ComponentFloat, vertexCount, "VEC3", null, null);
            WriteAccessor(w, 2, ComponentFloat, vertexCount, "VEC2", null, null);
            WriteAccessor(w, 3, ComponentUnsignedInt, indexCount, "SCALAR", null, null);
            w.WriteEndArray();

            int total = views[^1].Offset + GlbContainer.Align(views[^1].Length);
            w.WriteStartArray("buffers");
            w.WriteStartObject();
            w.WriteNumber("byteLength", total);
            w.WriteEndObject();
            w.WriteEndArray();
            w.WriteEndObject();
        }
        return ms.ToArray();
    }

    private static void WriteAccessor(Utf8JsonWriter w, int view, int component, int count, string type, float[]? min, float[]? max)
    {
        w.WriteStartObject();
        w.WriteNumber("bufferView", view);
        w.WriteNumber("componentType", component);
        w.WriteNumber("count", count);
        w.WriteString("type", type);
        if (min is not null && max is not null)
        {
            w.WriteStartArray("min");
            foreach (float v in min) w.WriteNumberValue(v);
            w.WriteEndArray();
            w.WriteStartArray("max");
            foreach (float v in max) w.WriteNumberValue(v);
            w.WriteEndArray();
        }
        w.WriteEndObject();
    }

    private static (float[] Min, float[] Max) Bounds(float[] xyz)
    {
        float[] min = [float.MaxValue, float.MaxValue, float.MaxValue];
        float[] max = [float.MinValue, float.MinValue, float.MinValue];
        for (int i = 0; i < xyz.Length; i += 3)
            for (int a = 0; a < 3; a++)
            {
                min[a] = Math.Min(min[a], xyz[i + a]);
                max[a] = Math.Max(max[a], xyz[i + a]);
            }
        return (min, max);
    }
}

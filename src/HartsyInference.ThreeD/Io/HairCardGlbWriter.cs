using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>
/// Binary glTF 2.0 writer for scene-root meshes with TEXCOORD_0, COLOR_0 (linear RGB, unsigned bytes) and one PBR material
/// per node. Additive to <see cref="GlbWriter"/>, which keeps writing plain POSITION/NORMAL meshes.
/// </summary>
public static class HairCardGlbWriter
{
    private const uint Magic = 0x46546C67;
    private const uint JsonChunk = 0x4E4F534A;
    private const uint BinChunk = 0x004E4942;
    private const int ArrayBuffer = 34962;
    private const int ElementArrayBuffer = 34963;
    private const int ComponentUnsignedByte = 5121;
    private const int ComponentUnsignedInt = 5125;
    private const int ComponentFloat = 5126;
    private const int ModeTriangles = 4;
    private const int PositionComponents = 3;

    /// <summary>Serializes <paramref name="nodes"/> (one mesh per scene-root node, in order) to a self-contained GLB.</summary>
    public static byte[] Write(IReadOnlyList<HairGlbNode> nodes)
    {
        ArgumentNullException.ThrowIfNull(nodes);
        if (nodes.Count == 0)
            throw new ArgumentException("At least one node is required.", nameof(nodes));
        foreach (HairGlbNode node in nodes)
            Validate(node);

        Layout layout = new();
        List<PrimitiveAccessors> primitives = [];
        foreach (HairGlbNode node in nodes)
            primitives.Add(layout.AddMesh(node.Mesh));

        byte[] json = BuildJson(nodes, primitives, layout);
        byte[] jsonPad = Pad(json, 0x20);
        byte[] binPad = Pad(layout.Bin.ToArray(), 0x00);

        int total = 12 + 8 + jsonPad.Length + 8 + binPad.Length;
        byte[] glb = new byte[total];
        int p = 0;
        WriteU32(glb, ref p, Magic);
        WriteU32(glb, ref p, 2);
        WriteU32(glb, ref p, (uint)total);
        WriteU32(glb, ref p, (uint)jsonPad.Length);
        WriteU32(glb, ref p, JsonChunk);
        jsonPad.CopyTo(glb, p);
        p += jsonPad.Length;
        WriteU32(glb, ref p, (uint)binPad.Length);
        WriteU32(glb, ref p, BinChunk);
        binPad.CopyTo(glb, p);
        return glb;
    }

    /// <summary>Writes <paramref name="nodes"/> as GLB to <paramref name="path"/>.</summary>
    public static void Save(string path, IReadOnlyList<HairGlbNode> nodes) => File.WriteAllBytes(path, Write(nodes));

    private static void Validate(HairGlbNode node)
    {
        ArgumentNullException.ThrowIfNull(node);
        ArgumentException.ThrowIfNullOrWhiteSpace(node.Name);
        ArgumentNullException.ThrowIfNull(node.Mesh);
        ArgumentNullException.ThrowIfNull(node.Material);
        Mesh mesh = node.Mesh;
        int vertexCount = mesh.VertexCount;
        if (mesh.Vertices.Length == 0 || mesh.Vertices.Length % PositionComponents != 0)
            throw new ArgumentException($"Node '{node.Name}' needs xyz vertex positions.", nameof(node));
        if (mesh.Indices.Length == 0 || mesh.Indices.Length % 3 != 0)
            throw new ArgumentException($"Node '{node.Name}' needs whole triangles.", nameof(node));
        RequireFinite(mesh.Vertices, node.Name, "position");
        foreach (int index in mesh.Indices)
            if (index < 0 || index >= vertexCount)
                throw new ArgumentException($"Node '{node.Name}' has an index outside its vertices.", nameof(node));
        if (mesh.Normals is { } normals)
        {
            RequireLength(normals, vertexCount * 3, node.Name, "normal");
            RequireFinite(normals, node.Name, "normal");
        }
        if (mesh.Uvs is { } uvs)
        {
            RequireLength(uvs, vertexCount * 2, node.Name, "UV");
            RequireFinite(uvs, node.Name, "UV");
        }
        if (mesh.VertexColors is { } colors)
        {
            RequireLength(colors, vertexCount * 3, node.Name, "colour");
            RequireFinite(colors, node.Name, "colour");
        }
        HairGlbMaterial material = node.Material;
        if (material.BaseColorFactor is not { Length: 4 } baseColor)
            throw new ArgumentException($"Material '{material.Name}' needs an RGBA base colour.", nameof(node));
        RequireFinite(baseColor, material.Name, "base colour");
        if (!(material.MetallicFactor is >= 0f and <= 1f) || !(material.RoughnessFactor is >= 0f and <= 1f))
            throw new ArgumentException($"Material '{material.Name}' has metallic or roughness outside [0, 1].", nameof(node));
    }

    private static void RequireLength(float[] values, int expected, string node, string what)
    {
        if (values.Length != expected)
            throw new ArgumentException($"Node '{node}' has {values.Length} {what} values, expected {expected}.");
    }

    private static void RequireFinite(float[] values, string node, string what)
    {
        foreach (float v in values)
            if (!float.IsFinite(v))
                throw new ArgumentException($"Node '{node}' has a non-finite {what} value.");
    }

    private static byte[] BuildJson(IReadOnlyList<HairGlbNode> nodes, List<PrimitiveAccessors> primitives, Layout layout)
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
            for (int i = 0; i < nodes.Count; i++)
                w.WriteNumberValue(i);
            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("nodes");
            for (int i = 0; i < nodes.Count; i++)
            {
                w.WriteStartObject();
                w.WriteString("name", nodes[i].Name);
                w.WriteNumber("mesh", i);
                w.WriteEndObject();
            }
            w.WriteEndArray();

            w.WriteStartArray("meshes");
            for (int i = 0; i < nodes.Count; i++)
            {
                PrimitiveAccessors a = primitives[i];
                w.WriteStartObject();
                w.WriteString("name", nodes[i].Name);
                w.WriteStartArray("primitives");
                w.WriteStartObject();
                w.WriteStartObject("attributes");
                w.WriteNumber("POSITION", a.Position);
                if (a.Normal is int normal) w.WriteNumber("NORMAL", normal);
                if (a.TexCoord is int texCoord) w.WriteNumber("TEXCOORD_0", texCoord);
                if (a.Color is int color) w.WriteNumber("COLOR_0", color);
                w.WriteEndObject();
                w.WriteNumber("indices", a.Indices);
                w.WriteNumber("material", i);
                w.WriteNumber("mode", ModeTriangles);
                w.WriteEndObject();
                w.WriteEndArray();
                w.WriteEndObject();
            }
            w.WriteEndArray();

            w.WriteStartArray("materials");
            foreach (HairGlbNode node in nodes)
                WriteMaterial(w, node.Material);
            w.WriteEndArray();

            w.WriteStartArray("bufferViews");
            foreach (BufferViewInfo view in layout.Views)
            {
                w.WriteStartObject();
                w.WriteNumber("buffer", 0);
                w.WriteNumber("byteOffset", view.Offset);
                w.WriteNumber("byteLength", view.Length);
                w.WriteNumber("target", view.Target);
                w.WriteEndObject();
            }
            w.WriteEndArray();

            w.WriteStartArray("accessors");
            foreach (AccessorInfo accessor in layout.Accessors)
                WriteAccessor(w, accessor);
            w.WriteEndArray();

            w.WriteStartArray("buffers");
            w.WriteStartObject();
            w.WriteNumber("byteLength", layout.Bin.Length);
            w.WriteEndObject();
            w.WriteEndArray();

            w.WriteEndObject();
        }
        return ms.ToArray();
    }

    private static void WriteMaterial(Utf8JsonWriter w, HairGlbMaterial material)
    {
        w.WriteStartObject();
        w.WriteString("name", material.Name);
        w.WriteStartObject("pbrMetallicRoughness");
        w.WriteStartArray("baseColorFactor");
        foreach (float c in material.BaseColorFactor)
            w.WriteNumberValue(c);
        w.WriteEndArray();
        w.WriteNumber("metallicFactor", material.MetallicFactor);
        w.WriteNumber("roughnessFactor", material.RoughnessFactor);
        w.WriteEndObject();
        w.WriteBoolean("doubleSided", material.DoubleSided);
        w.WriteEndObject();
    }

    private static void WriteAccessor(Utf8JsonWriter w, AccessorInfo accessor)
    {
        w.WriteStartObject();
        w.WriteNumber("bufferView", accessor.View);
        w.WriteNumber("componentType", accessor.ComponentType);
        if (accessor.Normalized)
            w.WriteBoolean("normalized", true);
        w.WriteNumber("count", accessor.Count);
        w.WriteString("type", accessor.Type);
        if (accessor.Min is not null)
        {
            w.WriteStartArray("min");
            foreach (float m in accessor.Min) w.WriteNumberValue(m);
            w.WriteEndArray();
        }
        if (accessor.Max is not null)
        {
            w.WriteStartArray("max");
            foreach (float m in accessor.Max) w.WriteNumberValue(m);
            w.WriteEndArray();
        }
        w.WriteEndObject();
    }

    private static byte[] Pad(byte[] data, byte padByte)
    {
        int rem = data.Length % 4;
        if (rem == 0)
            return data;
        byte[] padded = new byte[data.Length + (4 - rem)];
        data.CopyTo(padded, 0);
        Array.Fill(padded, padByte, data.Length, padded.Length - data.Length);
        return padded;
    }

    private static void WriteU32(byte[] dst, ref int off, uint v)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(dst.AsSpan(off), v);
        off += 4;
    }

    /// <summary>Buffer views and accessors of all nodes, in one binary buffer; every view starts on a 4-byte boundary.</summary>
    private sealed class Layout
    {
        public MemoryStream Bin { get; } = new();

        public List<BufferViewInfo> Views { get; } = [];

        public List<AccessorInfo> Accessors { get; } = [];

        public PrimitiveAccessors AddMesh(Mesh mesh)
        {
            int count = mesh.VertexCount;
            int position = AddFloats(mesh.Vertices, ArrayBuffer, count, "VEC3", withBounds: true);
            int? normal = mesh.Normals is { } n ? AddFloats(n, ArrayBuffer, count, "VEC3", withBounds: false) : null;
            int? texCoord = mesh.Uvs is { } uv ? AddFloats(uv, ArrayBuffer, count, "VEC2", withBounds: false) : null;
            int? color = mesh.VertexColors is { } rgb ? AddColors(rgb, count) : null;
            int indices = AddIndices(mesh.Indices);
            return new PrimitiveAccessors(position, normal, texCoord, color, indices);
        }

        private int AddFloats(float[] values, int target, int count, string type, bool withBounds)
        {
            byte[] data = new byte[values.Length * 4];
            for (int i = 0; i < values.Length; i++)
                BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(4 * i), values[i]);
            int view = Append(data, target);
            float[]? min = null;
            float[]? max = null;
            if (withBounds)
                (min, max) = Bounds(values, PositionComponentsOf(type));
            return AddAccessor(new AccessorInfo(view, ComponentFloat, count, type, min, max, false));
        }

        private int AddColors(float[] rgb, int count)
        {
            byte[] data = new byte[count * 4];
            for (int v = 0; v < count; v++)
            {
                for (int c = 0; c < 3; c++)
                    data[4 * v + c] = (byte)Math.Clamp(rgb[3 * v + c] * 255f + 0.5f, 0f, 255f);
                data[4 * v + 3] = 255;
            }
            int view = Append(data, ArrayBuffer);
            return AddAccessor(new AccessorInfo(view, ComponentUnsignedByte, count, "VEC4", null, null, true));
        }

        private int AddIndices(int[] indices)
        {
            byte[] data = new byte[indices.Length * 4];
            for (int i = 0; i < indices.Length; i++)
                BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(4 * i), (uint)indices[i]);
            int view = Append(data, ElementArrayBuffer);
            return AddAccessor(new AccessorInfo(view, ComponentUnsignedInt, indices.Length, "SCALAR", null, null, false));
        }

        private int Append(byte[] data, int target)
        {
            while (Bin.Length % 4 != 0)
                Bin.WriteByte(0);
            Views.Add(new BufferViewInfo((int)Bin.Length, data.Length, target));
            Bin.Write(data);
            return Views.Count - 1;
        }

        private int AddAccessor(AccessorInfo accessor)
        {
            Accessors.Add(accessor);
            return Accessors.Count - 1;
        }

        private static int PositionComponentsOf(string type) => type == "VEC2" ? 2 : PositionComponents;

        private static (float[] Min, float[] Max) Bounds(float[] values, int components)
        {
            float[] min = new float[components];
            float[] max = new float[components];
            Array.Fill(min, float.MaxValue);
            Array.Fill(max, float.MinValue);
            for (int i = 0; i < values.Length; i++)
            {
                int c = i % components;
                if (values[i] < min[c]) min[c] = values[i];
                if (values[i] > max[c]) max[c] = values[i];
            }
            return (min, max);
        }
    }

    private readonly record struct BufferViewInfo(int Offset, int Length, int Target);

    private readonly record struct AccessorInfo(int View, int ComponentType, int Count, string Type, float[]? Min,
        float[]? Max, bool Normalized);

    private readonly record struct PrimitiveAccessors(int Position, int? Normal, int? TexCoord, int? Color, int Indices);
}

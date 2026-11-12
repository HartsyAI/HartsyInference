using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>Writes meshes to binary glTF 2.0 (<c>.glb</c>) via manual chunk framing. Each mesh becomes one node and one
/// mesh with a single triangle primitive, exporting POSITION, optional NORMAL, and indices.</summary>
public static class GlbWriter
{
    private const uint Magic = 0x46546C67;      // "glTF"
    private const uint JsonChunk = 0x4E4F534A;  // "JSON"
    private const uint BinChunk = 0x004E4942;   // "BIN\0"

    /// <summary>Serializes <paramref name="mesh"/> to a self-contained GLB byte array.</summary>
    /// <param name="mesh">The mesh.</param>
    /// <param name="vertexColors">Also write <see cref="Mesh.VertexColors"/> as COLOR_0 (float RGB) when the mesh has them; off by default, which keeps every existing output as it was.</param>
    public static byte[] Write(Mesh mesh, bool vertexColors = false)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        return Write([mesh], vertexColors);
    }

    /// <summary>Serializes every mesh in <paramref name="meshes"/> into one scene, one node per mesh, in order.</summary>
    public static byte[] Write(IReadOnlyList<Mesh> meshes, bool vertexColors = false)
    {
        ArgumentNullException.ThrowIfNull(meshes);
        if (meshes.Count == 0)
            throw new ArgumentException("No meshes to export.", nameof(meshes));

        // --- Lay out the binary buffer: per mesh, positions, [normals], indices (all 4-byte aligned by construction). ---
        List<Layout> layouts = [];
        int binLength = 0, accessorCount = 0;
        foreach (Mesh mesh in meshes)
        {
            if (mesh.VertexCount == 0 || mesh.TriangleCount == 0)
                throw new ArgumentException("Mesh has no geometry to export.", nameof(meshes));
            bool hasNormals = mesh.Normals is { Length: > 0 } && mesh.Normals.Length == mesh.Vertices.Length;
            bool hasColors = vertexColors && mesh.VertexColors is { Length: > 0 } && mesh.VertexColors.Length == mesh.Vertices.Length;
            Layout layout = new(mesh, hasNormals, hasColors, binLength) { Accessor = accessorCount };
            layouts.Add(layout);
            binLength = checked(binLength + layout.TotalBytes);
            accessorCount += layout.ViewCount;
        }

        byte[] bin = new byte[binLength];
        foreach (Layout layout in layouts)
        {
            int off = layout.BinOffset;
            WriteFloats(bin, ref off, layout.Mesh.Vertices);
            if (layout.HasNormals) WriteFloats(bin, ref off, layout.Mesh.Normals!);
            if (layout.HasColors) WriteFloats(bin, ref off, layout.Mesh.VertexColors!);
            foreach (int i in layout.Mesh.Indices) { BinaryPrimitives.WriteUInt32LittleEndian(bin.AsSpan(off), (uint)i); off += 4; }
        }

        // --- JSON chunk ---
        byte[] json = BuildJson(layouts, binLength);

        // --- Pad chunks to 4-byte alignment (JSON: spaces; BIN: zeros). ---
        byte[] jsonPad = Pad(json, 0x20);
        byte[] binPad = Pad(bin, 0x00);

        int total = 12 + 8 + jsonPad.Length + 8 + binPad.Length;
        byte[] glb = new byte[total];
        int p = 0;
        WriteU32(glb, ref p, Magic);
        WriteU32(glb, ref p, 2);
        WriteU32(glb, ref p, (uint)total);
        WriteU32(glb, ref p, (uint)jsonPad.Length);
        WriteU32(glb, ref p, JsonChunk);
        Array.Copy(jsonPad, 0, glb, p, jsonPad.Length); p += jsonPad.Length;
        WriteU32(glb, ref p, (uint)binPad.Length);
        WriteU32(glb, ref p, BinChunk);
        Array.Copy(binPad, 0, glb, p, binPad.Length);
        return glb;
    }

    /// <summary>Writes <paramref name="mesh"/> as GLB to <paramref name="path"/>.</summary>
    public static void Save(string path, Mesh mesh) => File.WriteAllBytes(path, Write(mesh));

    private static byte[] BuildJson(List<Layout> layouts, int binLen)
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
            for (int m = 0; m < layouts.Count; m++) w.WriteNumberValue(m);
            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("nodes");
            for (int m = 0; m < layouts.Count; m++) { w.WriteStartObject(); w.WriteNumber("mesh", m); w.WriteEndObject(); }
            w.WriteEndArray();

            // Accessors and views are numbered per mesh: positions, [normals], indices.
            w.WriteStartArray("meshes");
            foreach (Layout layout in layouts)
            {
                w.WriteStartObject();
                w.WriteStartArray("primitives");
                w.WriteStartObject();
                w.WriteStartObject("attributes");
                w.WriteNumber("POSITION", layout.Accessor);
                if (layout.HasNormals) w.WriteNumber("NORMAL", layout.Accessor + 1);
                if (layout.HasColors) w.WriteNumber("COLOR_0", layout.Accessor + (layout.HasNormals ? 2 : 1));
                w.WriteEndObject();
                w.WriteNumber("indices", layout.Accessor + layout.ViewCount - 1);
                w.WriteNumber("mode", 4); // TRIANGLES
                w.WriteEndObject();
                w.WriteEndArray();
                w.WriteEndObject();
            }
            w.WriteEndArray();

            w.WriteStartArray("bufferViews");
            foreach (Layout layout in layouts)
            {
                WriteBufferView(w, layout.BinOffset, layout.Mesh.Vertices.Length * 4, 34962);
                int next = layout.BinOffset + layout.Mesh.Vertices.Length * 4;
                if (layout.HasNormals)
                {
                    WriteBufferView(w, next, layout.Mesh.Normals!.Length * 4, 34962);
                    next += layout.Mesh.Normals.Length * 4;
                }

                if (layout.HasColors) WriteBufferView(w, next, layout.Mesh.VertexColors!.Length * 4, 34962);
                WriteBufferView(w, layout.IndexOffset, layout.Mesh.Indices.Length * 4, 34963);
            }
            w.WriteEndArray();

            w.WriteStartArray("accessors");
            foreach (Layout layout in layouts)
            {
                (float[] min, float[] max) = Bounds(layout.Mesh.Vertices);
                WritePositionAccessor(w, layout.Accessor, layout.Mesh.VertexCount, min, max);
                if (layout.HasNormals) WriteVec3Accessor(w, layout.Accessor + 1, layout.Mesh.VertexCount);
                if (layout.HasColors) WriteVec3Accessor(w, layout.Accessor + (layout.HasNormals ? 2 : 1), layout.Mesh.VertexCount);
                WriteIndexAccessor(w, layout.Accessor + layout.ViewCount - 1, layout.Mesh.Indices.Length);
            }
            w.WriteEndArray();

            w.WriteStartArray("buffers");
            w.WriteStartObject(); w.WriteNumber("byteLength", binLen); w.WriteEndObject();
            w.WriteEndArray();

            w.WriteEndObject();
        }
        return ms.ToArray();
    }

    private static void WriteBufferView(Utf8JsonWriter w, int byteOffset, int byteLength, int target)
    {
        w.WriteStartObject();
        w.WriteNumber("buffer", 0);
        w.WriteNumber("byteOffset", byteOffset);
        w.WriteNumber("byteLength", byteLength);
        w.WriteNumber("target", target);
        w.WriteEndObject();
    }

    private static void WritePositionAccessor(Utf8JsonWriter w, int bufferView, int count, float[] min, float[] max)
    {
        w.WriteStartObject();
        w.WriteNumber("bufferView", bufferView);
        w.WriteNumber("componentType", 5126); // FLOAT
        w.WriteNumber("count", count);
        w.WriteString("type", "VEC3");
        w.WriteStartArray("min"); foreach (float m in min) w.WriteNumberValue(m); w.WriteEndArray();
        w.WriteStartArray("max"); foreach (float m in max) w.WriteNumberValue(m); w.WriteEndArray();
        w.WriteEndObject();
    }

    private static void WriteVec3Accessor(Utf8JsonWriter w, int bufferView, int count)
    {
        w.WriteStartObject();
        w.WriteNumber("bufferView", bufferView);
        w.WriteNumber("componentType", 5126);
        w.WriteNumber("count", count);
        w.WriteString("type", "VEC3");
        w.WriteEndObject();
    }

    private static void WriteIndexAccessor(Utf8JsonWriter w, int bufferView, int count)
    {
        w.WriteStartObject();
        w.WriteNumber("bufferView", bufferView);
        w.WriteNumber("componentType", 5125); // UNSIGNED_INT
        w.WriteNumber("count", count);
        w.WriteString("type", "SCALAR");
        w.WriteEndObject();
    }

    private static (float[] min, float[] max) Bounds(float[] verts)
    {
        float[] min = [float.MaxValue, float.MaxValue, float.MaxValue];
        float[] max = [float.MinValue, float.MinValue, float.MinValue];
        for (int i = 0; i < verts.Length; i += 3)
            for (int a = 0; a < 3; a++)
            {
                float v = verts[i + a];
                if (v < min[a]) min[a] = v;
                if (v > max[a]) max[a] = v;
            }
        return (min, max);
    }

    private static void WriteFloats(byte[] dst, ref int off, float[] src)
    {
        foreach (float f in src) { BinaryPrimitives.WriteSingleLittleEndian(dst.AsSpan(off), f); off += 4; }
    }

    private static void WriteU32(byte[] dst, ref int off, uint v)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(dst.AsSpan(off), v); off += 4;
    }

    private static byte[] Pad(byte[] data, byte padByte)
    {
        int rem = data.Length % 4;
        if (rem == 0) return data;
        byte[] padded = new byte[data.Length + (4 - rem)];
        Array.Copy(data, padded, data.Length);
        for (int i = data.Length; i < padded.Length; i++) padded[i] = padByte;
        return padded;
    }

    /// <summary>Where one mesh sits in the binary buffer and in the accessor and view numbering.</summary>
    private sealed class Layout
    {
        public Layout(Mesh mesh, bool hasNormals, bool hasColors, int binOffset)
        {
            Mesh = mesh;
            HasNormals = hasNormals;
            HasColors = hasColors;
            BinOffset = binOffset;
            IndexOffset = binOffset + mesh.Vertices.Length * 4 + (hasNormals ? mesh.Normals!.Length * 4 : 0) + (hasColors ? mesh.VertexColors!.Length * 4 : 0);
        }

        public Mesh Mesh { get; }

        public bool HasNormals { get; }

        public bool HasColors { get; }

        /// <summary>Byte offset of this mesh's positions in the buffer.</summary>
        public int BinOffset { get; }

        /// <summary>Byte offset of this mesh's indices in the buffer.</summary>
        public int IndexOffset { get; }

        public int TotalBytes => IndexOffset + Mesh.Indices.Length * 4 - BinOffset;

        public int ViewCount => 2 + (HasNormals ? 1 : 0) + (HasColors ? 1 : 0);

        /// <summary>Index of this mesh's position accessor and view; its views and accessors follow in order.</summary>
        public int Accessor { get; set; }
    }
}

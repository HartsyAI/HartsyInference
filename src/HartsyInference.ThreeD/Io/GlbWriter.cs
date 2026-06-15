using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>Writes a <see cref="Mesh"/> to binary glTF 2.0 (<c>.glb</c>) — the standard interchange format
/// read by Blender, three.js, Windows 3D Viewer, and SwarmUI's 3D output. Pure managed C# (System.Text.Json
/// + manual chunk framing), no external dependencies. Exports POSITION, optional NORMAL, and triangle
/// indices as a single mesh/primitive.</summary>
public static class GlbWriter
{
    private const uint Magic = 0x46546C67;      // "glTF"
    private const uint JsonChunk = 0x4E4F534A;  // "JSON"
    private const uint BinChunk = 0x004E4942;   // "BIN\0"

    /// <summary>Serializes <paramref name="mesh"/> to a self-contained GLB byte array.</summary>
    public static byte[] Write(Mesh mesh)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        if (mesh.VertexCount == 0 || mesh.TriangleCount == 0)
            throw new ArgumentException("Mesh has no geometry to export.", nameof(mesh));

        bool hasNormals = mesh.Normals is { Length: > 0 } && mesh.Normals.Length == mesh.Vertices.Length;

        // --- Build the binary buffer: positions, [normals], indices (all 4-byte aligned by construction). ---
        int posBytes = mesh.Vertices.Length * 4;
        int normBytes = hasNormals ? mesh.Normals!.Length * 4 : 0;
        int idxBytes = mesh.Indices.Length * 4;
        byte[] bin = new byte[posBytes + normBytes + idxBytes];
        int off = 0;
        WriteFloats(bin, ref off, mesh.Vertices);
        if (hasNormals) WriteFloats(bin, ref off, mesh.Normals!);
        int idxOffset = off;
        foreach (int i in mesh.Indices) { BinaryPrimitives.WriteUInt32LittleEndian(bin.AsSpan(off), (uint)i); off += 4; }

        // POSITION accessor requires min/max bounds.
        (float[] min, float[] max) = Bounds(mesh.Vertices);

        // --- JSON chunk ---
        byte[] json = BuildJson(mesh, hasNormals, posBytes, normBytes, idxBytes, idxOffset, bin.Length, min, max);

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

    private static byte[] BuildJson(Mesh mesh, bool hasNormals, int posBytes, int normBytes, int idxBytes,
        int idxOffset, int binLen, float[] min, float[] max)
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
            w.WriteStartObject(); w.WriteStartArray("nodes"); w.WriteNumberValue(0); w.WriteEndArray(); w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("nodes");
            w.WriteStartObject(); w.WriteNumber("mesh", 0); w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("meshes");
            w.WriteStartObject();
            w.WriteStartArray("primitives");
            w.WriteStartObject();
            w.WriteStartObject("attributes");
            w.WriteNumber("POSITION", 0);
            if (hasNormals) w.WriteNumber("NORMAL", 1);
            w.WriteEndObject();
            w.WriteNumber("indices", hasNormals ? 2 : 1);
            w.WriteNumber("mode", 4); // TRIANGLES
            w.WriteEndObject();
            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();

            // bufferViews: 0=positions, [1=normals], last=indices.
            w.WriteStartArray("bufferViews");
            WriteBufferView(w, 0, posBytes, 34962);
            if (hasNormals) WriteBufferView(w, posBytes, normBytes, 34962);
            WriteBufferView(w, idxOffset, idxBytes, 34963);
            w.WriteEndArray();

            // accessors mirror the bufferView order.
            w.WriteStartArray("accessors");
            WritePositionAccessor(w, 0, mesh.VertexCount, min, max);
            if (hasNormals) WriteVec3Accessor(w, 1, mesh.VertexCount);
            WriteIndexAccessor(w, hasNormals ? 2 : 1, mesh.Indices.Length);
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
}

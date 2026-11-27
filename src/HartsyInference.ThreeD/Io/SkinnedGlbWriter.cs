using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>
/// Writes a skinned triangle mesh to binary glTF 2.0: one mesh node bound to a skin, one node per bone (rest pose is translation
/// only), inverse bind matrices, and up to four influences per vertex as <c>JOINTS_0</c> / <c>WEIGHTS_0</c>. Positions, normals,
/// UVs and vertex colors of the mesh are kept when present. Separate from the hair-card writer, whose records it does not touch.
/// </summary>
public static class SkinnedGlbWriter
{
    private const uint Magic = 0x46546C67;
    private const uint JsonChunk = 0x4E4F534A;
    private const uint BinChunk = 0x004E4942;
    private const int Influences = 4;
    private const int FloatType = 5126;
    private const int UnsignedShortType = 5123;
    private const int UnsignedIntType = 5125;
    private const int ArrayBuffer = 34962;
    private const int ElementBuffer = 34963;

    /// <param name="mesh">The mesh to export, in the same space as the bone positions.</param>
    /// <param name="bones">Rest-pose bones; at least one.</param>
    /// <param name="joints">Bone index per influence slot, length <c>4 * VertexCount</c>.</param>
    /// <param name="weights">Weight per slot, length <c>4 * VertexCount</c>; each vertex's slots sum to 1.</param>
    public static byte[] Write(Mesh mesh, IReadOnlyList<SkinnedBone> bones, int[] joints, float[] weights)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        ArgumentNullException.ThrowIfNull(bones);
        ArgumentNullException.ThrowIfNull(joints);
        ArgumentNullException.ThrowIfNull(weights);
        Validate(mesh, bones, joints, weights);

        List<View> views = [];
        using MemoryStream bin = new();
        int Add(ReadOnlySpan<byte> bytes, int target, int componentType, int count, string type, string? name, float[]? min = null,
            float[]? max = null)
        {
            while (bin.Length % 4 != 0)
            {
                bin.WriteByte(0);
            }

            views.Add(new View((int)bin.Length, bytes.Length, target, componentType, count, type, name, min, max));
            bin.Write(bytes);
            return views.Count - 1;
        }

        int position = Add(Floats(mesh.Vertices), ArrayBuffer, FloatType, mesh.VertexCount, "VEC3", "POSITION", Bounds(mesh.Vertices,
            true), Bounds(mesh.Vertices, false));
        int normal = mesh.Normals is { } n && n.Length == mesh.Vertices.Length ? Add(Floats(n), ArrayBuffer, FloatType,
            mesh.VertexCount, "VEC3", "NORMAL") : -1;
        int uv = mesh.Uvs is { } t && t.Length == 2 * mesh.VertexCount ? Add(Floats(t), ArrayBuffer, FloatType, mesh.VertexCount,
            "VEC2", "TEXCOORD_0") : -1;
        int color = mesh.VertexColors is { } c && c.Length == mesh.Vertices.Length ? Add(Floats(c), ArrayBuffer, FloatType,
            mesh.VertexCount, "VEC3", "COLOR_0") : -1;
        int jointsView = Add(UShorts(joints), ArrayBuffer, UnsignedShortType, mesh.VertexCount, "VEC4", "JOINTS_0");
        int weightsView = Add(Floats(weights), ArrayBuffer, FloatType, mesh.VertexCount, "VEC4", "WEIGHTS_0");
        int indices = Add(UInts(mesh.Indices), ElementBuffer, UnsignedIntType, mesh.Indices.Length, "SCALAR", null);
        int bind = Add(Floats(InverseBind(bones)), 0, FloatType, bones.Count, "MAT4", null);

        byte[] json = BuildJson(views, bones, position, normal, uv, color, jointsView, weightsView, indices, bind, (int)bin.Length);
        return Frame(json, bin.ToArray());
    }

    private static void Validate(Mesh mesh, IReadOnlyList<SkinnedBone> bones, int[] joints, float[] weights)
    {
        if (mesh.VertexCount == 0 || mesh.TriangleCount == 0)
        {
            throw new HartsyInferenceException("Skinned GLB needs a mesh with geometry.");
        }

        if (bones.Count == 0)
        {
            throw new HartsyInferenceException("Skinned GLB needs at least one bone.");
        }

        if (joints.Length != Influences * mesh.VertexCount || weights.Length != joints.Length)
        {
            throw new HartsyInferenceException(
                $"Skinned GLB needs {Influences} influences per vertex; "
                + $"got {joints.Length} joints, {weights.Length} weights, {mesh.VertexCount} vertices.");
        }

        for (int i = 0; i < bones.Count; i++)
        {
            if (bones[i].Parent >= i || bones[i].Parent < -1)
            {
                throw new HartsyInferenceException($"Bone {i} has parent {bones[i].Parent}; parents must precede their children.");
            }
        }

        foreach (int joint in joints)
        {
            if (joint < 0 || joint >= bones.Count || joint > ushort.MaxValue)
            {
                throw new HartsyInferenceException($"Skin joint index {joint} is outside the {bones.Count} bones.");
            }
        }
    }

    private static byte[] BuildJson(List<View> views, IReadOnlyList<SkinnedBone> bones, int position, int normal, int uv, int color,
        int jointsView, int weightsView, int indices, int bind, int binLength)
    {
        using MemoryStream stream = new();
        using (Utf8JsonWriter w = new(stream))
        {
            w.WriteStartObject();
            w.WriteStartObject("asset");
            w.WriteString("version", "2.0");
            w.WriteString("generator", "HartsyInference");
            w.WriteEndObject();
            w.WriteNumber("scene", 0);
            w.WriteStartArray("scenes");
            w.WriteStartObject();
            w.WriteStartArray("nodes");
            w.WriteNumberValue(0);
            for (int i = 0; i < bones.Count; i++)
            {
                if (bones[i].Parent < 0)
                {
                    w.WriteNumberValue(1 + i);
                }
            }

            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("nodes");
            w.WriteStartObject();
            w.WriteString("name", "mesh");
            w.WriteNumber("mesh", 0);
            w.WriteNumber("skin", 0);
            w.WriteEndObject();
            for (int i = 0; i < bones.Count; i++)
            {
                WriteBoneNode(w, bones, i);
            }

            w.WriteEndArray();

            w.WriteStartArray("skins");
            w.WriteStartObject();
            w.WriteNumber("inverseBindMatrices", bind);
            int root = 0;
            while (bones[root].Parent >= 0)
            {
                root++;
            }

            w.WriteNumber("skeleton", 1 + root);
            w.WriteStartArray("joints");
            for (int i = 0; i < bones.Count; i++)
            {
                w.WriteNumberValue(1 + i);
            }

            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("meshes");
            w.WriteStartObject();
            w.WriteStartArray("primitives");
            w.WriteStartObject();
            w.WriteStartObject("attributes");
            w.WriteNumber("POSITION", position);
            if (normal >= 0)
            {
                w.WriteNumber("NORMAL", normal);
            }

            if (uv >= 0)
            {
                w.WriteNumber("TEXCOORD_0", uv);
            }

            if (color >= 0)
            {
                w.WriteNumber("COLOR_0", color);
            }

            w.WriteNumber("JOINTS_0", jointsView);
            w.WriteNumber("WEIGHTS_0", weightsView);
            w.WriteEndObject();
            w.WriteNumber("indices", indices);
            w.WriteNumber("mode", 4);
            w.WriteEndObject();
            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("accessors");
            for (int i = 0; i < views.Count; i++)
            {
                View v = views[i];
                w.WriteStartObject();
                w.WriteNumber("bufferView", i);
                w.WriteNumber("componentType", v.ComponentType);
                w.WriteNumber("count", v.Count);
                w.WriteString("type", v.Type);
                if (v.Min is not null && v.Max is not null)
                {
                    WriteFloatArray(w, "min", v.Min);
                    WriteFloatArray(w, "max", v.Max);
                }

                w.WriteEndObject();
            }

            w.WriteEndArray();

            w.WriteStartArray("bufferViews");
            foreach (View v in views)
            {
                w.WriteStartObject();
                w.WriteNumber("buffer", 0);
                w.WriteNumber("byteOffset", v.Offset);
                w.WriteNumber("byteLength", v.Length);
                if (v.Target != 0)
                {
                    w.WriteNumber("target", v.Target);
                }

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

    private static void WriteBoneNode(Utf8JsonWriter w, IReadOnlyList<SkinnedBone> bones, int i)
    {
        SkinnedBone bone = bones[i];
        (float X, float Y, float Z) parentHead = bone.Parent < 0 ? (0f, 0f, 0f) : bones[bone.Parent].Head;
        w.WriteStartObject();
        w.WriteString("name", bone.Name);
        WriteFloatArray(w, "translation", [bone.Head.X - parentHead.X, bone.Head.Y - parentHead.Y, bone.Head.Z - parentHead.Z]);
        List<int> children = [];
        for (int c = i + 1; c < bones.Count; c++)
        {
            if (bones[c].Parent == i)
            {
                children.Add(1 + c);
            }
        }

        if (children.Count > 0)
        {
            w.WriteStartArray("children");
            foreach (int child in children)
            {
                w.WriteNumberValue(child);
            }

            w.WriteEndArray();
        }

        w.WriteStartObject("extras");
        WriteFloatArray(w, "tail", [bone.Tail.X, bone.Tail.Y, bone.Tail.Z]);
        w.WriteEndObject();
        w.WriteEndObject();
    }

    private static void WriteFloatArray(Utf8JsonWriter w, string name, float[] values)
    {
        w.WriteStartArray(name);
        foreach (float value in values)
        {
            w.WriteNumberValue(value);
        }

        w.WriteEndArray();
    }

    private static float[] InverseBind(IReadOnlyList<SkinnedBone> bones)
    {
        float[] matrices = new float[16 * bones.Count];
        for (int i = 0; i < bones.Count; i++)
        {
            int at = 16 * i;
            matrices[at] = matrices[at + 5] = matrices[at + 10] = matrices[at + 15] = 1f;
            matrices[at + 12] = -bones[i].Head.X;
            matrices[at + 13] = -bones[i].Head.Y;
            matrices[at + 14] = -bones[i].Head.Z;
        }

        return matrices;
    }

    private static float[] Bounds(float[] positions, bool min)
    {
        float[] result = min ? [float.MaxValue, float.MaxValue, float.MaxValue] : [float.MinValue, float.MinValue, float.MinValue];
        for (int i = 0; i < positions.Length; i++)
        {
            result[i % 3] = min ? Math.Min(result[i % 3], positions[i]) : Math.Max(result[i % 3], positions[i]);
        }

        return result;
    }

    private static byte[] Floats(float[] values)
    {
        byte[] bytes = new byte[4 * values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(4 * i), values[i]);
        }

        return bytes;
    }

    private static byte[] UShorts(int[] values)
    {
        byte[] bytes = new byte[2 * values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(2 * i), (ushort)values[i]);
        }

        return bytes;
    }

    private static byte[] UInts(int[] values)
    {
        byte[] bytes = new byte[4 * values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4 * i), (uint)values[i]);
        }

        return bytes;
    }

    private static byte[] Frame(byte[] json, byte[] bin)
    {
        byte[] jsonPadded = Pad(json, 0x20);
        byte[] binPadded = Pad(bin, 0x00);
        byte[] glb = new byte[12 + 8 + jsonPadded.Length + 8 + binPadded.Length];
        int p = 0;
        Write(glb, ref p, Magic);
        Write(glb, ref p, 2);
        Write(glb, ref p, (uint)glb.Length);
        Write(glb, ref p, (uint)jsonPadded.Length);
        Write(glb, ref p, JsonChunk);
        jsonPadded.CopyTo(glb, p);
        p += jsonPadded.Length;
        Write(glb, ref p, (uint)binPadded.Length);
        Write(glb, ref p, BinChunk);
        binPadded.CopyTo(glb, p);
        return glb;
    }

    private static byte[] Pad(byte[] data, byte fill)
    {
        int padded = (data.Length + 3) & ~3;
        if (padded == data.Length)
        {
            return data;
        }

        byte[] result = new byte[padded];
        data.CopyTo(result, 0);
        Array.Fill(result, fill, data.Length, padded - data.Length);
        return result;
    }

    private static void Write(byte[] target, ref int offset, uint value)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(target.AsSpan(offset), value);
        offset += 4;
    }

    private sealed record View(int Offset, int Length, int Target, int ComponentType, int Count, string Type, string? Name,
        float[]? Min, float[]? Max);
}

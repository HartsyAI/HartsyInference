using System.Buffers.Binary;
using System.Numerics;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>
/// Reads triangle primitives from a binary glTF 2.0 file. Each primitive is placed by the node graph of the default scene
/// (every root node when the file has no scenes), so positions come out in world space. Materials and animation are
/// ignored; sparse, compressed and buffer-less accessors are refused.
/// </summary>
public static class GlbReader
{
    private const int TrianglesMode = 4;
    private const int FloatType = 5126;
    private const int UnsignedByteType = 5121;
    private const int UnsignedShortType = 5123;
    private const int UnsignedIntType = 5125;

    /// <summary>Parses a GLB container and concatenates every triangle primitive its scene places.</summary>
    public static Mesh Parse(byte[] glb)
    {
        ArgumentNullException.ThrowIfNull(glb);
        (byte[] json, byte[] bin) = SplitChunks(glb);
        using JsonDocument doc = ParseJson(json);
        JsonElement root = doc.RootElement;
        if (!TryArray(root, "meshes", out JsonElement meshes) || meshes.GetArrayLength() == 0)
            throw new HartsyInferenceException("GLB file has no meshes.");

        Collector collector = new(root, bin, meshes);
        collector.PlaceScene();
        if (collector.Indices.Count == 0)
            throw new HartsyInferenceException("GLB file has no triangle primitives.");
        return new Mesh { Vertices = [.. collector.Vertices], Indices = [.. collector.Indices] };
    }

    private static (byte[] Json, byte[] Bin) SplitChunks(byte[] glb)
    {
        GlbContainer.Chunks chunks = GlbContainer.Split(glb);
        return (chunks.Json, chunks.Bin);
    }

    private static JsonDocument ParseJson(byte[] json)
    {
        try
        {
            return JsonDocument.Parse(json);
        }
        catch (JsonException ex)
        {
            throw new HartsyInferenceException("GLB JSON chunk is not valid JSON.", ex);
        }
    }

    /// <summary>Walks the scene graph and appends each placed primitive to the output buffers.</summary>
    private sealed class Collector(JsonElement root, byte[] bin, JsonElement meshes)
    {
        public List<float> Vertices { get; } = [];

        public List<int> Indices { get; } = [];

        /// <summary>Places every mesh the scene reaches. A file with no node list places each mesh at the origin.</summary>
        public void PlaceScene()
        {
            if (!TryArray(root, "nodes", out JsonElement nodes))
            {
                for (int m = 0; m < meshes.GetArrayLength(); m++)
                    AddMesh(m, Matrix4x4.Identity);
                return;
            }

            // A node reached twice has two parents or sits in a cycle; a tree visit reaches each node once.
            // Pushed in reverse so that the stack visits siblings in file order.
            bool[] visited = new bool[nodes.GetArrayLength()];
            List<int> roots = SceneRoots(nodes);
            Stack<(int Node, Matrix4x4 Parent)> pending = new();
            for (int i = roots.Count - 1; i >= 0; i--)
                pending.Push((roots[i], Matrix4x4.Identity));
            while (pending.Count > 0)
            {
                (int index, Matrix4x4 parent) = pending.Pop();
                if (visited[index])
                    throw new HartsyInferenceException("GLB node graph is not a tree: a node has two parents or sits in a cycle.");
                visited[index] = true;

                JsonElement node = Item(nodes, index, "node");
                Matrix4x4 world = LocalTransform(node) * parent;
                if (TryMember(node, "mesh", out JsonElement mesh))
                    AddMesh(Natural(mesh, "mesh index"), world);
                if (TryArray(node, "children", out JsonElement children))
                {
                    for (int i = children.GetArrayLength() - 1; i >= 0; i--)
                        pending.Push((NodeIndex(children[i], nodes), world));
                }
            }
        }

        // The default scene's roots, or without scenes, the nodes no other node names as a child.
        private List<int> SceneRoots(JsonElement nodes)
        {
            List<int> roots = [];
            if (TryArray(root, "scenes", out JsonElement scenes) && scenes.GetArrayLength() > 0)
            {
                JsonElement scene = Item(scenes, IntMember(root, "scene", 0), "scene");
                if (TryArray(scene, "nodes", out JsonElement sceneNodes))
                {
                    foreach (JsonElement node in sceneNodes.EnumerateArray())
                        roots.Add(NodeIndex(node, nodes));
                }
                return roots;
            }

            bool[] named = new bool[nodes.GetArrayLength()];
            for (int i = 0; i < named.Length; i++)
            {
                if (TryArray(Item(nodes, i, "node"), "children", out JsonElement children))
                {
                    foreach (JsonElement child in children.EnumerateArray())
                        named[NodeIndex(child, nodes)] = true;
                }
            }
            for (int i = 0; i < named.Length; i++)
            {
                if (!named[i])
                    roots.Add(i);
            }
            return roots;
        }

        private void AddMesh(int meshIndex, Matrix4x4 world)
        {
            JsonElement mesh = Item(meshes, meshIndex, "mesh");
            bool mirrored = world.GetDeterminant() < 0;
            foreach (JsonElement primitive in RequiredArray(mesh, "primitives").EnumerateArray())
            {
                if (IntMember(primitive, "mode", TrianglesMode) != TrianglesMode)
                    continue;
                if (!TryMember(primitive, "indices", out JsonElement indexAccessor))
                    throw new HartsyInferenceException("GLB triangle primitive has no index accessor.");

                int baseVertex = Vertices.Count / 3;
                ReadPositions(NaturalMember(Member(primitive, "attributes"), "POSITION"), world);
                int vertexCount = Vertices.Count / 3 - baseVertex;
                int firstIndex = Indices.Count;
                ReadIndices(Natural(indexAccessor, "accessor index"), baseVertex, vertexCount);
                if ((Indices.Count - firstIndex) % 3 != 0)
                    throw new HartsyInferenceException("GLB triangle index count is not a multiple of three.");

                // A mirroring transform turns each triangle inside out; swapping two corners restores its facing.
                if (mirrored)
                {
                    for (int i = firstIndex; i < Indices.Count; i += 3)
                        (Indices[i + 1], Indices[i + 2]) = (Indices[i + 2], Indices[i + 1]);
                }
            }
        }

        private void ReadPositions(int accessorIndex, Matrix4x4 world)
        {
            JsonElement accessor = Item(RequiredArray(root, "accessors"), accessorIndex, "accessor");
            if (IntMember(accessor, "componentType") != FloatType || StringMember(accessor, "type") != "VEC3")
                throw new HartsyInferenceException("GLB POSITION accessor must be float VEC3.");
            int count = NaturalMember(accessor, "count");
            byte[] src = AccessorBytes(accessor, 12, count);
            for (int i = 0; i < count; i++)
            {
                Vector3 position = Vector3.Transform(new Vector3(
                    BinaryPrimitives.ReadSingleLittleEndian(src.AsSpan(i * 12)),
                    BinaryPrimitives.ReadSingleLittleEndian(src.AsSpan(i * 12 + 4)),
                    BinaryPrimitives.ReadSingleLittleEndian(src.AsSpan(i * 12 + 8))), world);
                if (!float.IsFinite(position.X) || !float.IsFinite(position.Y) || !float.IsFinite(position.Z))
                    throw new HartsyInferenceException("GLB POSITION accessor holds a non-finite coordinate.");
                Vertices.Add(position.X);
                Vertices.Add(position.Y);
                Vertices.Add(position.Z);
            }
        }

        private void ReadIndices(int accessorIndex, int baseVertex, int vertexCount)
        {
            JsonElement accessor = Item(RequiredArray(root, "accessors"), accessorIndex, "accessor");
            int componentType = IntMember(accessor, "componentType");
            int size = componentType switch
            {
                UnsignedByteType => 1,
                UnsignedShortType => 2,
                UnsignedIntType => 4,
                _ => throw new HartsyInferenceException($"GLB index accessor component type {componentType} is not supported."),
            };
            int count = NaturalMember(accessor, "count");
            byte[] src = AccessorBytes(accessor, size, count);
            for (int i = 0; i < count; i++)
            {
                uint value = size switch
                {
                    1 => src[i],
                    2 => BinaryPrimitives.ReadUInt16LittleEndian(src.AsSpan(i * 2)),
                    _ => BinaryPrimitives.ReadUInt32LittleEndian(src.AsSpan(i * 4)),
                };
                if (value >= (uint)vertexCount)
                    throw new HartsyInferenceException($"GLB index {value} is outside its primitive's {vertexCount} vertices.");
                Indices.Add(baseVertex + (int)value);
            }
        }

        // Returns the accessor's elements packed at elementSize, honouring its buffer view's byte stride and bounds.
        private byte[] AccessorBytes(JsonElement accessor, int elementSize, int count)
        {
            if (!TryMember(accessor, "bufferView", out JsonElement viewIndex))
                throw new HartsyInferenceException("GLB accessor has no buffer view; sparse, Draco and meshopt accessors are not supported.");
            JsonElement view = Item(RequiredArray(root, "bufferViews"), Natural(viewIndex, "buffer view index"), "bufferView");
            int buffer = NaturalMember(view, "buffer");
            if (buffer != 0)
                throw new HartsyInferenceException($"GLB buffer view reads buffer {buffer}; only the GLB binary chunk (buffer 0) is supported.");
            long viewOffset = NaturalMember(view, "byteOffset", 0);
            long viewEnd = viewOffset + NaturalMember(view, "byteLength");
            long stride = IntMember(view, "byteStride", elementSize);
            if (stride < elementSize)
                throw new HartsyInferenceException($"GLB byte stride {stride} is smaller than the element size {elementSize}.");
            long start = viewOffset + NaturalMember(accessor, "byteOffset", 0);
            long end = start + stride * Math.Max(count - 1, 0) + elementSize;
            if (end > viewEnd || end > bin.Length)
                throw new HartsyInferenceException("GLB accessor reads past the end of its buffer view.");

            byte[] packed = new byte[count * elementSize];
            for (int i = 0; i < count; i++)
                bin.AsSpan((int)(start + i * stride), elementSize).CopyTo(packed.AsSpan(i * elementSize));
            return packed;
        }

        private int NodeIndex(JsonElement value, JsonElement nodes)
        {
            int index = Natural(value, "node index");
            return index < nodes.GetArrayLength() ? index : throw new HartsyInferenceException($"GLB node index {index} is out of range.");
        }
    }

    // The node's local transform: its matrix when present, otherwise scale, then rotation, then translation.
    private static Matrix4x4 LocalTransform(JsonElement node)
    {
        float[]? m = FloatsMember(node, "matrix", 16);
        if (m is not null)
        {
            // glTF lists each matrix column in turn; the row-vector Matrix4x4 reads the same sixteen values as its rows.
            return new Matrix4x4(
                m[0], m[1], m[2], m[3],
                m[4], m[5], m[6], m[7],
                m[8], m[9], m[10], m[11],
                m[12], m[13], m[14], m[15]);
        }

        Quaternion rotation = Quaternion.Identity;
        float[]? r = FloatsMember(node, "rotation", 4);
        if (r is not null)
        {
            rotation = new Quaternion(r[0], r[1], r[2], r[3]);
            if (rotation.LengthSquared() == 0)
                throw new HartsyInferenceException("GLB node rotation has zero length.");
            rotation = Quaternion.Normalize(rotation);
        }
        float[]? s = FloatsMember(node, "scale", 3);
        float[]? t = FloatsMember(node, "translation", 3);
        return Matrix4x4.CreateScale(s is null ? Vector3.One : new Vector3(s[0], s[1], s[2]))
            * Matrix4x4.CreateFromQuaternion(rotation)
            * Matrix4x4.CreateTranslation(t is null ? Vector3.Zero : new Vector3(t[0], t[1], t[2]));
    }

    private static float[]? FloatsMember(JsonElement obj, string name, int count)
    {
        if (!TryMember(obj, name, out JsonElement values))
            return null;
        if (values.ValueKind != JsonValueKind.Array || values.GetArrayLength() != count)
            throw new HartsyInferenceException($"GLB node '{name}' must hold {count} numbers.");
        float[] result = new float[count];
        int i = 0;
        foreach (JsonElement element in values.EnumerateArray())
        {
            if (!element.TryGetSingle(out float value) || !float.IsFinite(value))
                throw new HartsyInferenceException($"GLB node '{name}' must hold finite numbers.");
            result[i++] = value;
        }
        return result;
    }

    private static bool TryMember(JsonElement obj, string name, out JsonElement value)
    {
        value = default;
        return obj.ValueKind == JsonValueKind.Object && obj.TryGetProperty(name, out value);
    }

    private static bool TryArray(JsonElement obj, string name, out JsonElement array)
    {
        if (!TryMember(obj, name, out array))
            return false;
        if (array.ValueKind != JsonValueKind.Array)
            throw new HartsyInferenceException($"GLB '{name}' must be an array.");
        return true;
    }

    private static JsonElement RequiredArray(JsonElement obj, string name) =>
        TryArray(obj, name, out JsonElement array) ? array : throw new HartsyInferenceException($"GLB is missing '{name}'.");

    private static JsonElement Member(JsonElement obj, string name) =>
        TryMember(obj, name, out JsonElement value) ? value : throw new HartsyInferenceException($"GLB is missing '{name}'.");

    private static string StringMember(JsonElement obj, string name) =>
        TryMember(obj, name, out JsonElement value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()!
            : throw new HartsyInferenceException($"GLB '{name}' must be a string.");

    private static int IntMember(JsonElement obj, string name, int? fallback = null)
    {
        if (!TryMember(obj, name, out JsonElement value))
            return fallback ?? throw new HartsyInferenceException($"GLB is missing '{name}'.");
        return value.TryGetInt32(out int number) ? number : throw new HartsyInferenceException($"GLB '{name}' must be an integer.");
    }

    private static int NaturalMember(JsonElement obj, string name, int? fallback = null)
    {
        int value = IntMember(obj, name, fallback);
        return value >= 0 ? value : throw new HartsyInferenceException($"GLB '{name}' must not be negative.");
    }

    private static int Natural(JsonElement value, string what) =>
        value.TryGetInt32(out int number) && number >= 0
            ? number
            : throw new HartsyInferenceException($"GLB {what} must be a non-negative integer.");

    private static JsonElement Item(JsonElement array, int index, string what) =>
        (uint)index < (uint)array.GetArrayLength()
            ? array[index]
            : throw new HartsyInferenceException($"GLB {what} index {index} is out of range.");
}

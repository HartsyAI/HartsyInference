using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>
/// Reads triangle primitives from a binary glTF 2.0 file. Positions and indices are read through their accessors and
/// buffer views; node transforms and materials are ignored, so the result is in the file's raw mesh coordinates.
/// </summary>
public static class GlbReader
{
    private const uint Magic = 0x46546C67;
    private const uint JsonChunk = 0x4E4F534A;
    private const uint BinChunk = 0x004E4942;
    private const int TrianglesMode = 4;
    private const int FloatType = 5126;
    private const int UnsignedByteType = 5121;
    private const int UnsignedShortType = 5123;
    private const int UnsignedIntType = 5125;

    /// <summary>Parses a GLB container and concatenates every triangle primitive it holds.</summary>
    public static Mesh Parse(byte[] glb)
    {
        ArgumentNullException.ThrowIfNull(glb);
        (byte[] json, byte[] bin) = SplitChunks(glb);
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;
        if (!root.TryGetProperty("meshes", out JsonElement meshes) || meshes.GetArrayLength() == 0)
            throw new HartsyInferenceException("GLB file has no meshes.");

        List<float> vertices = [];
        List<int> indices = [];
        foreach (JsonElement mesh in meshes.EnumerateArray())
        {
            foreach (JsonElement primitive in mesh.GetProperty("primitives").EnumerateArray())
            {
                if (primitive.TryGetProperty("mode", out JsonElement mode) && mode.GetInt32() != TrianglesMode)
                    continue;
                int baseVertex = vertices.Count / 3;
                ReadPositions(root, bin, primitive.GetProperty("attributes").GetProperty("POSITION").GetInt32(), vertices);
                int vertexCount = vertices.Count / 3 - baseVertex;
                if (!primitive.TryGetProperty("indices", out JsonElement indexAccessor))
                    throw new HartsyInferenceException("GLB triangle primitive has no index accessor.");
                ReadIndices(root, bin, indexAccessor.GetInt32(), baseVertex, vertexCount, indices);
            }
        }

        if (indices.Count == 0)
            throw new HartsyInferenceException("GLB file has no triangle primitives.");
        return new Mesh { Vertices = [.. vertices], Indices = [.. indices] };
    }

    private static (byte[] Json, byte[] Bin) SplitChunks(byte[] glb)
    {
        if (glb.Length < 20 || BinaryPrimitives.ReadUInt32LittleEndian(glb) != Magic)
            throw new HartsyInferenceException("File is not a binary glTF (GLB) container.");
        if (BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(4)) != 2)
            throw new HartsyInferenceException("Only glTF 2.0 GLB containers are supported.");

        byte[]? json = null, bin = null;
        int offset = 12;
        while (offset + 8 <= glb.Length)
        {
            int length = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(offset)));
            uint type = BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(offset + 4));
            int start = offset + 8;
            if (length < 0 || start + length > glb.Length)
                throw new HartsyInferenceException("GLB chunk runs past the end of the file.");
            byte[] chunk = glb.AsSpan(start, length).ToArray();
            if (type == JsonChunk) json = chunk;
            else if (type == BinChunk) bin = chunk;
            offset = start + length;
        }

        return json is null
            ? throw new HartsyInferenceException("GLB file has no JSON chunk.")
            : (json, bin ?? []);
    }

    private static void ReadPositions(JsonElement root, byte[] bin, int accessorIndex, List<float> vertices)
    {
        JsonElement accessor = Accessor(root, accessorIndex);
        if (accessor.GetProperty("componentType").GetInt32() != FloatType || accessor.GetProperty("type").GetString() != "VEC3")
            throw new HartsyInferenceException("GLB POSITION accessor must be float VEC3.");
        int count = accessor.GetProperty("count").GetInt32();
        ReadOnlySpan<byte> src = AccessorBytes(root, bin, accessor, 12, count);
        for (int i = 0; i < count; i++)
        {
            for (int a = 0; a < 3; a++)
            {
                float value = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(src.Slice(i * 12 + a * 4)));
                if (!float.IsFinite(value))
                    throw new HartsyInferenceException("GLB POSITION accessor holds a non-finite coordinate.");
                vertices.Add(value);
            }
        }
    }

    private static void ReadIndices(JsonElement root, byte[] bin, int accessorIndex, int baseVertex, int vertexCount, List<int> indices)
    {
        JsonElement accessor = Accessor(root, accessorIndex);
        int componentType = accessor.GetProperty("componentType").GetInt32();
        int size = componentType switch
        {
            UnsignedByteType => 1,
            UnsignedShortType => 2,
            UnsignedIntType => 4,
            _ => throw new HartsyInferenceException($"GLB index accessor component type {componentType} is not supported."),
        };
        int count = accessor.GetProperty("count").GetInt32();
        ReadOnlySpan<byte> src = AccessorBytes(root, bin, accessor, size, count);
        for (int i = 0; i < count; i++)
        {
            int value = size switch
            {
                1 => src[i],
                2 => BinaryPrimitives.ReadUInt16LittleEndian(src.Slice(i * 2)),
                _ => checked((int)BinaryPrimitives.ReadUInt32LittleEndian(src.Slice(i * 4))),
            };
            if (value >= vertexCount)
                throw new HartsyInferenceException($"GLB index {value} is outside its primitive's {vertexCount} vertices.");
            indices.Add(baseVertex + value);
        }
    }

    private static JsonElement Accessor(JsonElement root, int index) =>
        root.GetProperty("accessors")[index];

    // Returns the accessor's elements packed at `elementSize`, honouring its buffer view's byte stride.
    private static ReadOnlySpan<byte> AccessorBytes(JsonElement root, byte[] bin, JsonElement accessor, int elementSize, int count)
    {
        JsonElement view = root.GetProperty("bufferViews")[accessor.GetProperty("bufferView").GetInt32()];
        int viewOffset = view.TryGetProperty("byteOffset", out JsonElement vo) ? vo.GetInt32() : 0;
        int stride = view.TryGetProperty("byteStride", out JsonElement st) ? st.GetInt32() : elementSize;
        int accessorOffset = accessor.TryGetProperty("byteOffset", out JsonElement ao) ? ao.GetInt32() : 0;
        if (stride < elementSize)
            throw new HartsyInferenceException($"GLB byte stride {stride} is smaller than the element size {elementSize}.");
        long end = viewOffset + accessorOffset + (long)stride * Math.Max(count - 1, 0) + elementSize;
        if (end > bin.Length)
            throw new HartsyInferenceException("GLB accessor reads past the end of the binary chunk.");

        byte[] packed = new byte[count * elementSize];
        for (int i = 0; i < count; i++)
            bin.AsSpan(viewOffset + accessorOffset + i * stride, elementSize).CopyTo(packed.AsSpan(i * elementSize));
        return packed;
    }
}

using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>Writes a <see cref="MeshDocument"/> as a binary glTF 2.0: one node holding one mesh with a primitive per document primitive (POSITION, NORMAL, TEXCOORD_0, COLOR_0, 32-bit indices), the materials with their images embedded as they came, and the document's <c>asset.extras</c>. Animation and skins are not written; the document does not hold them.</summary>
public static class MeshDocumentGlbWriter
{
    /// <summary>Serializes <paramref name="document"/>.</summary>
    public static byte[] Write(MeshDocument document)
    {
        ArgumentNullException.ThrowIfNull(document);
        if (document.Primitives.Count == 0)
            throw new ArgumentException("The document has no primitives to write.", nameof(document));

        BufferBuilder bin = new();
        List<AccessorInfo> accessors = [];
        List<int[]> primitiveAccessors = [];
        foreach (MeshPrimitive primitive in document.Primitives)
        {
            Mesh g = primitive.Geometry;
            int position = AddAccessor(bin, accessors, FloatBytes(g.Vertices), 5126, g.VertexCount, "VEC3", 34962, Bounds(g.Vertices));
            int normal = g.Normals is null ? -1 : AddAccessor(bin, accessors, FloatBytes(g.Normals), 5126, g.VertexCount, "VEC3", 34962, null);
            int uv = g.Uvs is null ? -1 : AddAccessor(bin, accessors, FloatBytes(g.Uvs), 5126, g.VertexCount, "VEC2", 34962, null);
            int colour = g.VertexColors is null ? -1 : AddAccessor(bin, accessors, FloatBytes(g.VertexColors), 5126, g.VertexCount, "VEC3", 34962, null);
            int index = AddAccessor(bin, accessors, IntBytes(g.Indices), 5125, g.Indices.Length, "SCALAR", 34963, null);
            primitiveAccessors.Add([position, normal, uv, colour, index]);
        }

        Dictionary<MeshImage, int> imageIndex = new(ReferenceEqualityComparer.Instance);
        List<(MeshImage Image, int View)> images = [];
        foreach (MeshImage image in document.Materials.SelectMany(m => m.Images))
        {
            if (imageIndex.ContainsKey(image))
                continue;
            imageIndex[image] = images.Count;
            images.Add((image, bin.AddView(image.Data, null)));
        }

        using MemoryStream stream = new();
        using (Utf8JsonWriter w = new(stream))
        {
            w.WriteStartObject();
            w.WriteStartObject("asset");
            w.WriteString("version", "2.0");
            w.WriteString("generator", "HartsyInference");
            if (document.AssetExtrasJson is not null)
            {
                w.WritePropertyName("extras");
                using JsonDocument extras = JsonDocument.Parse(document.AssetExtrasJson);
                extras.RootElement.WriteTo(w);
            }
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
            w.WriteNumber("mesh", 0);
            w.WriteEndObject();
            w.WriteEndArray();

            w.WriteStartArray("meshes");
            w.WriteStartObject();
            w.WriteStartArray("primitives");
            for (int p = 0; p < document.Primitives.Count; p++)
            {
                int[] a = primitiveAccessors[p];
                w.WriteStartObject();
                w.WriteStartObject("attributes");
                w.WriteNumber("POSITION", a[0]);
                if (a[1] >= 0) w.WriteNumber("NORMAL", a[1]);
                if (a[2] >= 0) w.WriteNumber("TEXCOORD_0", a[2]);
                if (a[3] >= 0) w.WriteNumber("COLOR_0", a[3]);
                w.WriteEndObject();
                w.WriteNumber("indices", a[4]);
                if ((uint)document.Primitives[p].Material < (uint)document.Materials.Count)
                    w.WriteNumber("material", document.Primitives[p].Material);
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();

            if (document.Materials.Count > 0)
                WriteMaterials(w, document, imageIndex);
            if (images.Count > 0)
            {
                w.WriteStartArray("images");
                foreach ((MeshImage image, int view) in images)
                {
                    w.WriteStartObject();
                    w.WriteString("mimeType", image.MimeType);
                    w.WriteNumber("bufferView", view);
                    w.WriteEndObject();
                }
                w.WriteEndArray();
                w.WriteStartArray("textures");
                for (int i = 0; i < images.Count; i++)
                {
                    w.WriteStartObject();
                    w.WriteNumber("source", i);
                    w.WriteEndObject();
                }
                w.WriteEndArray();
            }

            w.WriteStartArray("accessors");
            foreach (AccessorInfo a in accessors)
            {
                w.WriteStartObject();
                w.WriteNumber("bufferView", a.View);
                w.WriteNumber("componentType", a.ComponentType);
                w.WriteNumber("count", a.Count);
                w.WriteString("type", a.Type);
                if (a.Bounds is { } bounds)
                {
                    w.WriteStartArray("min");
                    foreach (float v in bounds.Min) w.WriteNumberValue(v);
                    w.WriteEndArray();
                    w.WriteStartArray("max");
                    foreach (float v in bounds.Max) w.WriteNumberValue(v);
                    w.WriteEndArray();
                }
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteStartArray("bufferViews");
            foreach (BufferView view in bin.Views)
            {
                w.WriteStartObject();
                w.WriteNumber("buffer", 0);
                w.WriteNumber("byteOffset", view.Offset);
                w.WriteNumber("byteLength", view.Length);
                if (view.Target is int target)
                    w.WriteNumber("target", target);
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteStartArray("buffers");
            w.WriteStartObject();
            w.WriteNumber("byteLength", bin.Length);
            w.WriteEndObject();
            w.WriteEndArray();
            w.WriteEndObject();
        }
        return GlbContainer.Assemble(stream.ToArray(), bin.ToArray());
    }

    private static void WriteMaterials(Utf8JsonWriter w, MeshDocument document, Dictionary<MeshImage, int> imageIndex)
    {
        w.WriteStartArray("materials");
        foreach (MeshMaterial m in document.Materials)
        {
            w.WriteStartObject();
            if (m.Name.Length > 0)
                w.WriteString("name", m.Name);
            w.WriteStartObject("pbrMetallicRoughness");
            WriteFloats(w, "baseColorFactor", m.BaseColorFactor);
            if (m.BaseColor is not null)
                Texture(w, "baseColorTexture", imageIndex[m.BaseColor]);
            w.WriteNumber("metallicFactor", m.Metallic);
            w.WriteNumber("roughnessFactor", m.Roughness);
            if (m.MetallicRoughness is not null)
                Texture(w, "metallicRoughnessTexture", imageIndex[m.MetallicRoughness]);
            w.WriteEndObject();
            if (m.Normal is not null)
                Texture(w, "normalTexture", imageIndex[m.Normal]);
            if (m.Occlusion is not null)
                Texture(w, "occlusionTexture", imageIndex[m.Occlusion]);
            if (m.EmissiveFactor.Any(v => v != 0))
                WriteFloats(w, "emissiveFactor", m.EmissiveFactor);
            if (m.Emissive is not null)
                Texture(w, "emissiveTexture", imageIndex[m.Emissive]);
            if (m.AlphaMode != MeshAlphaMode.Opaque)
                w.WriteString("alphaMode", m.AlphaMode == MeshAlphaMode.Mask ? "MASK" : "BLEND");
            if (m.AlphaMode == MeshAlphaMode.Mask)
                w.WriteNumber("alphaCutoff", m.AlphaCutoff);
            if (m.DoubleSided)
                w.WriteBoolean("doubleSided", true);
            w.WriteEndObject();
        }
        w.WriteEndArray();
    }

    private static void Texture(Utf8JsonWriter w, string name, int texture)
    {
        w.WriteStartObject(name);
        w.WriteNumber("index", texture);
        w.WriteEndObject();
    }

    private static void WriteFloats(Utf8JsonWriter w, string name, float[] values)
    {
        w.WriteStartArray(name);
        foreach (float v in values)
            w.WriteNumberValue(v);
        w.WriteEndArray();
    }

    private static int AddAccessor(BufferBuilder bin, List<AccessorInfo> accessors, byte[] data, int componentType, int count, string type, int target, (float[] Min, float[] Max)? bounds)
    {
        accessors.Add(new AccessorInfo(bin.AddView(data, target), componentType, count, type, bounds));
        return accessors.Count - 1;
    }

    private static (float[] Min, float[] Max) Bounds(float[] positions)
    {
        float[] min = [float.MaxValue, float.MaxValue, float.MaxValue], max = [float.MinValue, float.MinValue, float.MinValue];
        for (int i = 0; i < positions.Length; i += 3)
        {
            for (int k = 0; k < 3; k++)
            {
                min[k] = Math.Min(min[k], positions[i + k]);
                max[k] = Math.Max(max[k], positions[i + k]);
            }
        }
        return (min, max);
    }

    private static byte[] FloatBytes(float[] values)
    {
        byte[] bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        return bytes;
    }

    private static byte[] IntBytes(int[] values)
    {
        byte[] bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(i * 4), (uint)values[i]);
        return bytes;
    }

    private sealed record AccessorInfo(int View, int ComponentType, int Count, string Type, (float[] Min, float[] Max)? Bounds);

    private sealed record BufferView(int Offset, int Length, int? Target);

    /// <summary>The binary chunk under construction: each view starts on a four-byte boundary.</summary>
    private sealed class BufferBuilder
    {
        private readonly MemoryStream _stream = new();

        public List<BufferView> Views { get; } = [];

        public int Length => (int)_stream.Length;

        public int AddView(byte[] data, int? target)
        {
            while (_stream.Length % 4 != 0)
                _stream.WriteByte(0);
            Views.Add(new BufferView((int)_stream.Length, data.Length, target));
            _stream.Write(data);
            return Views.Count - 1;
        }

        public byte[] ToArray()
        {
            while (_stream.Length % 4 != 0)
                _stream.WriteByte(0);
            return _stream.ToArray();
        }
    }
}

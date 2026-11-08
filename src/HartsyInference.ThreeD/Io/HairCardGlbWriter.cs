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
    private const int ComponentUnsignedShort = 5123;
    private const int ModeTriangles = 4;
    private const int PositionComponents = 3;

    /// <summary>Serializes <paramref name="nodes"/> (one mesh per scene-root node, in order) to a self-contained GLB. With
    /// <paramref name="rig"/>, the joints are scene-root nodes, nodes with a <see cref="HairGlbNode.Skin"/> are skinned to them,
    /// and the clips become glTF animations.</summary>
    public static byte[] Write(IReadOnlyList<HairGlbNode> nodes, HairGlbRig? rig = null)
    {
        ArgumentNullException.ThrowIfNull(nodes);
        if (nodes.Count == 0)
            throw new ArgumentException("At least one node is required.", nameof(nodes));
        if (rig is not null)
            ValidateRig(rig);
        foreach (HairGlbNode node in nodes)
            Validate(node, rig);

        Layout layout = new();
        List<PrimitiveAccessors> primitives = [];
        foreach (HairGlbNode node in nodes)
            primitives.Add(layout.AddMesh(node.Mesh, node.Skin));
        RigAccessors? rigAccessors = rig is null ? null : layout.AddRig(rig);

        byte[] json = BuildJson(nodes, primitives, layout, rig, rigAccessors);
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

    private static void ValidateRig(HairGlbRig rig)
    {
        ArgumentNullException.ThrowIfNull(rig);
        if (rig.Joints.Count == 0)
            throw new ArgumentException("A rig needs at least one joint.", nameof(rig));
        foreach (HairGlbJoint joint in rig.Joints)
        {
            if (joint.Translation.Length != 3 || joint.Rotation.Length != 4 || joint.InverseBindMatrix.Length != 16)
                throw new ArgumentException($"Joint '{joint.Name}' needs a 3-value translation, a 4-value rotation and a 16-value matrix.", nameof(rig));
            RequireFinite(joint.Translation, joint.Name, "joint translation");
            RequireFinite(joint.Rotation, joint.Name, "joint rotation");
            RequireFinite(joint.InverseBindMatrix, joint.Name, "inverse bind matrix");
        }
        foreach (HairGlbAnimation animation in rig.Animations)
        {
            int frames = animation.Times.Length;
            if (frames == 0)
                throw new ArgumentException($"Animation '{animation.Name}' has no key frames.", nameof(rig));
            RequireFinite(animation.Times, animation.Name, "time");
            if (animation.Channels.Count != rig.Joints.Count)
                throw new ArgumentException($"Animation '{animation.Name}' needs one channel per joint.", nameof(rig));
            foreach (HairGlbChannel channel in animation.Channels)
            {
                RequireLength(channel.Translations, 3 * frames, animation.Name, "translation");
                RequireLength(channel.Rotations, 4 * frames, animation.Name, "rotation");
                RequireFinite(channel.Translations, animation.Name, "translation");
                RequireFinite(channel.Rotations, animation.Name, "rotation");
            }
        }
    }

    private static void ValidateSkin(HairGlbNode node, HairGlbSkin skin, int vertexCount, int jointCount)
    {
        if (skin.JointIndices.Length != 4 * vertexCount || skin.Weights.Length != 4 * vertexCount)
            throw new ArgumentException($"Node '{node.Name}' needs four joint indices and weights per vertex.", nameof(node));
        for (int v = 0; v < vertexCount; v++)
        {
            double sum = 0;
            for (int i = 0; i < 4; i++)
            {
                float w = skin.Weights[4 * v + i];
                if (!float.IsFinite(w) || w < 0f)
                    throw new ArgumentException($"Node '{node.Name}' has a negative or non-finite weight.", nameof(node));
                if (w > 0f && skin.JointIndices[4 * v + i] >= jointCount)
                    throw new ArgumentException($"Node '{node.Name}' refers to joint {skin.JointIndices[4 * v + i]}, but the rig has {jointCount}.", nameof(node));
                sum += w;
            }
            if (Math.Abs(sum - 1.0) > 1e-3)
                throw new ArgumentException($"Node '{node.Name}' has vertex weights that sum to {sum:G6}, not 1.", nameof(node));
        }
    }

    private static void Validate(HairGlbNode node, HairGlbRig? rig)
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
        if (node.Skin is { } skin)
        {
            if (rig is null)
                throw new ArgumentException($"Node '{node.Name}' is skinned but no rig was given.", nameof(node));
            ValidateSkin(node, skin, vertexCount, rig.Joints.Count);
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

    private static byte[] BuildJson(IReadOnlyList<HairGlbNode> nodes, List<PrimitiveAccessors> primitives, Layout layout,
        HairGlbRig? rig, RigAccessors? rigAccessors)
    {
        int jointCount = rig?.Joints.Count ?? 0;
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
            for (int i = 0; i < nodes.Count + jointCount; i++)
                w.WriteNumberValue(i);
            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndArray();

            // Mesh nodes first (index i holds mesh i), then the joints as root-level nodes (index nodes.Count + j).
            w.WriteStartArray("nodes");
            for (int i = 0; i < nodes.Count; i++)
            {
                w.WriteStartObject();
                w.WriteString("name", nodes[i].Name);
                w.WriteNumber("mesh", i);
                if (nodes[i].Skin is not null)
                    w.WriteNumber("skin", 0);
                w.WriteEndObject();
            }
            if (rig is not null)
            {
                for (int j = 0; j < jointCount; j++)
                {
                    HairGlbJoint joint = rig.Joints[j];
                    w.WriteStartObject();
                    w.WriteString("name", joint.Name);
                    w.WriteStartArray("translation");
                    foreach (float t in joint.Translation) w.WriteNumberValue(t);
                    w.WriteEndArray();
                    w.WriteStartArray("rotation");
                    foreach (float q in joint.Rotation) w.WriteNumberValue(q);
                    w.WriteEndArray();
                    w.WriteEndObject();
                }
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
                if (a.Joints is int joints) w.WriteNumber("JOINTS_0", joints);
                if (a.Weights is int weights) w.WriteNumber("WEIGHTS_0", weights);
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
                if (view.Target != 0)
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

            if (rig is not null && rigAccessors is not null)
            {
                w.WriteStartArray("skins");
                w.WriteStartObject();
                w.WriteStartArray("joints");
                for (int j = 0; j < jointCount; j++)
                    w.WriteNumberValue(nodes.Count + j);
                w.WriteEndArray();
                w.WriteNumber("inverseBindMatrices", rigAccessors.InverseBind);
                w.WriteEndObject();
                w.WriteEndArray();

                w.WriteStartArray("animations");
                for (int a = 0; a < rig.Animations.Count; a++)
                {
                    HairGlbAnimation animation = rig.Animations[a];
                    AnimationAccessors accessors = rigAccessors.Animations[a];
                    w.WriteStartObject();
                    w.WriteString("name", animation.Name);
                    w.WriteStartArray("samplers");
                    for (int j = 0; j < jointCount; j++)
                    {
                        WriteSampler(w, accessors.Time, accessors.Translations[j]);
                        WriteSampler(w, accessors.Time, accessors.Rotations[j]);
                    }
                    w.WriteEndArray();
                    w.WriteStartArray("channels");
                    for (int j = 0; j < jointCount; j++)
                    {
                        WriteChannel(w, 2 * j, nodes.Count + j, "translation");
                        WriteChannel(w, 2 * j + 1, nodes.Count + j, "rotation");
                    }
                    w.WriteEndArray();
                    w.WriteEndObject();
                }
                w.WriteEndArray();
            }

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

        public PrimitiveAccessors AddMesh(Mesh mesh, HairGlbSkin? skin)
        {
            int count = mesh.VertexCount;
            int position = AddFloats(mesh.Vertices, ArrayBuffer, count, "VEC3", withBounds: true);
            int? normal = mesh.Normals is { } n ? AddFloats(n, ArrayBuffer, count, "VEC3", withBounds: false) : null;
            int? texCoord = mesh.Uvs is { } uv ? AddFloats(uv, ArrayBuffer, count, "VEC2", withBounds: false) : null;
            int? color = mesh.VertexColors is { } rgb ? AddColors(rgb, count) : null;
            int? joints = null;
            int? weights = null;
            if (skin is not null)
            {
                joints = AddJoints(skin.JointIndices, count);
                weights = AddFloats(skin.Weights, ArrayBuffer, count, "VEC4", withBounds: false);
            }
            int indices = AddIndices(mesh.Indices);
            return new PrimitiveAccessors(position, normal, texCoord, color, indices, joints, weights);
        }

        /// <summary>Joint and inverse-bind accessors and the animation accessors of the rig.</summary>
        public RigAccessors AddRig(HairGlbRig rig)
        {
            int joints = rig.Joints.Count;
            float[] inverseBind = new float[16 * joints];
            for (int j = 0; j < joints; j++)
                rig.Joints[j].InverseBindMatrix.CopyTo(inverseBind, 16 * j);
            int inverseBindAccessor = AddFloats(inverseBind, 0, joints, "MAT4", withBounds: false);

            List<AnimationAccessors> animations = [];
            foreach (HairGlbAnimation animation in rig.Animations)
            {
                int frames = animation.Times.Length;
                float[] min = [animation.Times.Min()];
                float[] max = [animation.Times.Max()];
                int time = AddAccessor(new AccessorInfo(Append(FloatBytes(animation.Times), 0), ComponentFloat, frames, "SCALAR", min, max, false));
                List<int> translations = [];
                List<int> rotations = [];
                foreach (HairGlbChannel channel in animation.Channels)
                {
                    translations.Add(AddFloats(channel.Translations, 0, frames, "VEC3", withBounds: false));
                    rotations.Add(AddFloats(channel.Rotations, 0, frames, "VEC4", withBounds: false));
                }
                animations.Add(new AnimationAccessors(time, translations, rotations));
            }
            return new RigAccessors(inverseBindAccessor, animations);
        }

        private int AddJoints(ushort[] values, int count)
        {
            byte[] data = new byte[values.Length * 2];
            for (int i = 0; i < values.Length; i++)
                BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(2 * i), values[i]);
            int view = Append(data, ArrayBuffer);
            return AddAccessor(new AccessorInfo(view, ComponentUnsignedShort, count, "VEC4", null, null, false));
        }

        private static byte[] FloatBytes(float[] values)
        {
            byte[] data = new byte[values.Length * 4];
            for (int i = 0; i < values.Length; i++)
                BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(4 * i), values[i]);
            return data;
        }

        private int AddFloats(float[] values, int target, int count, string type, bool withBounds)
        {
            int view = Append(FloatBytes(values), target);
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

    private readonly record struct PrimitiveAccessors(int Position, int? Normal, int? TexCoord, int? Color, int Indices,
        int? Joints = null, int? Weights = null);

    private sealed record RigAccessors(int InverseBind, IReadOnlyList<AnimationAccessors> Animations);

    private sealed record AnimationAccessors(int Time, IReadOnlyList<int> Translations, IReadOnlyList<int> Rotations);

    private static void WriteSampler(Utf8JsonWriter w, int input, int output)
    {
        w.WriteStartObject();
        w.WriteNumber("input", input);
        w.WriteNumber("output", output);
        w.WriteString("interpolation", "LINEAR");
        w.WriteEndObject();
    }

    private static void WriteChannel(Utf8JsonWriter w, int sampler, int node, string path)
    {
        w.WriteStartObject();
        w.WriteNumber("sampler", sampler);
        w.WriteStartObject("target");
        w.WriteNumber("node", node);
        w.WriteString("path", path);
        w.WriteEndObject();
        w.WriteEndObject();
    }
}

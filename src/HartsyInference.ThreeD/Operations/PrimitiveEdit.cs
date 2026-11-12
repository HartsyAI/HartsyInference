using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Operations;

/// <summary>The shared part of an operation that rewrites primitives one by one: it applies the edit, then makes the document honest about what the edit cost. A primitive whose UVs are gone cannot show its textures, so it gets a copy of its material without them, and a note says so.</summary>
internal static class PrimitiveEdit
{
    /// <summary>Applies <paramref name="edit"/> to every primitive. The edit returns the same instance when it changed nothing; otherwise the new primitive, and a note about what it did.</summary>
    /// <param name="document">The model.</param>
    /// <param name="operation">The operation id, for notes.</param>
    /// <param name="edit">The edit of one primitive: its index and the primitive; returns the result and an optional note.</param>
    /// <param name="cancel">Checked between primitives.</param>
    public static MeshOperationOutcome Map(MeshDocument document, string operation, Func<int, MeshPrimitive, (MeshPrimitive Result, string? Note)> edit, CancellationToken cancel)
    {
        List<MeshPrimitive> primitives = [];
        List<MeshMaterial> materials = [.. document.Materials];
        List<string> notes = [];
        for (int i = 0; i < document.Primitives.Count; i++)
        {
            cancel.ThrowIfCancellationRequested();
            MeshPrimitive source = document.Primitives[i];
            (MeshPrimitive result, string? note) = edit(i, source);
            if (note is not null)
                notes.Add(note);
            if (!ReferenceEquals(result, source) && source.Geometry.Uvs is not null && result.Geometry.Uvs is null
                && (uint)result.Material < (uint)materials.Count && materials[result.Material].Images.Any())
            {
                MeshMaterial plain = materials[result.Material] with { BaseColor = null, MetallicRoughness = null, Normal = null, Occlusion = null, Emissive = null };
                materials.Add(plain);
                result = new MeshPrimitive { Geometry = result.Geometry, Material = materials.Count - 1 };
                notes.Add($"Primitive {i}: its textures were dropped, because {operation} does not carry UV coordinates; the material colour is kept.");
            }
            primitives.Add(result);
        }
        return new MeshOperationOutcome(document.With(primitives, materials), notes);
    }

    /// <summary>A copy of <paramref name="mesh"/> with its vertex positions replaced and every other array shared.</summary>
    public static Mesh WithPositions(Mesh mesh, float[] positions) => new()
    {
        Vertices = positions,
        Indices = mesh.Indices,
        Normals = mesh.Normals,
        Uvs = mesh.Uvs,
        VertexColors = mesh.VertexColors,
    };

    /// <summary>The smooth normal of every vertex: the area-weighted normal of the faces around the surface point, shared by all the vertices at that position (so a UV seam is not a crease). A vertex no face uses gets +Y.</summary>
    public static float[] WeldedNormals(Mesh mesh, WeldedVertices welded)
    {
        System.Numerics.Vector3[] sum = new System.Numerics.Vector3[welded.Count];
        float[] p = welded.Positions;
        for (int t = 0; t < welded.Indices.Length; t += 3)
        {
            int a = welded.Indices[t], b = welded.Indices[t + 1], c = welded.Indices[t + 2];
            System.Numerics.Vector3 pa = At(p, a);
            System.Numerics.Vector3 face = System.Numerics.Vector3.Cross(At(p, b) - pa, At(p, c) - pa);
            sum[a] += face;
            sum[b] += face;
            sum[c] += face;
        }
        float[] normals = new float[mesh.Vertices.Length];
        for (int v = 0; v < welded.Group.Length; v++)
        {
            System.Numerics.Vector3 n = sum[welded.Group[v]];
            n = n.LengthSquared() > 0 ? System.Numerics.Vector3.Normalize(n) : System.Numerics.Vector3.UnitY;
            normals[v * 3] = n.X;
            normals[v * 3 + 1] = n.Y;
            normals[v * 3 + 2] = n.Z;
        }
        return normals;
    }

    /// <summary>The position of vertex <paramref name="index"/>.</summary>
    public static System.Numerics.Vector3 At(float[] positions, int index) => new(positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2]);

    /// <summary>The diagonal of the bounding box of <paramref name="positions"/>.</summary>
    public static float Diagonal(float[] positions)
    {
        System.Numerics.Vector3 min = new(float.PositiveInfinity), max = new(float.NegativeInfinity);
        for (int i = 0; i < positions.Length; i += 3)
        {
            System.Numerics.Vector3 v = new(positions[i], positions[i + 1], positions[i + 2]);
            min = System.Numerics.Vector3.Min(min, v);
            max = System.Numerics.Vector3.Max(max, v);
        }
        return (max - min).Length();
    }
}

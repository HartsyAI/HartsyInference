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
        Dictionary<int, int> plainCopies = [];
        List<string> notes = [];
        for (int i = 0; i < document.Primitives.Count; i++)
        {
            cancel.ThrowIfCancellationRequested();
            MeshPrimitive source = document.Primitives[i];
            (MeshPrimitive result, string? note) = edit(i, source);
            if (note is not null)
                notes.Add(note);
            if (!ReferenceEquals(result, source))
            {
                if (source.Geometry.Uvs is not null && result.Geometry.Uvs is null
                    && (uint)result.Material < (uint)materials.Count && materials[result.Material].Images.Any())
                {
                    // One image-free copy per source material, shared by every primitive that used it.
                    if (!plainCopies.TryGetValue(result.Material, out int plain))
                    {
                        materials.Add(materials[result.Material] with { BaseColor = null, MetallicRoughness = null, Normal = null, Occlusion = null, Emissive = null });
                        plain = plainCopies[result.Material] = materials.Count - 1;
                    }
                    result = new MeshPrimitive { Geometry = result.Geometry, Material = plain };
                    notes.Add($"Primitive {i}: its textures were dropped, because {operation} does not carry UV coordinates; the material colour is kept.");
                }
                if (source.Geometry.VertexColors is not null && result.Geometry.VertexColors is null)
                    notes.Add($"Primitive {i}: its vertex colours were dropped, because {operation} does not carry them.");
            }
            primitives.Add(result);
        }
        (List<MeshMaterial> kept, List<MeshPrimitive> remapped) = DropUnusedMaterials(materials, primitives);
        return new MeshOperationOutcome(document.With(remapped, kept), notes);
    }

    /// <summary>Removes the materials no primitive uses, and with them the images only they held, and renumbers what is left. Without this an edit that orphans a texture (decimate, fill-holes) would still write the whole image into the file it was meant to make smaller.</summary>
    private static (List<MeshMaterial>, List<MeshPrimitive>) DropUnusedMaterials(List<MeshMaterial> materials, List<MeshPrimitive> primitives)
    {
        HashSet<int> used = [.. primitives.Select(p => p.Material).Where(m => (uint)m < (uint)materials.Count)];
        if (used.Count == materials.Count)
            return (materials, primitives);
        Dictionary<int, int> renumber = [];
        List<MeshMaterial> kept = [];
        for (int m = 0; m < materials.Count; m++)
        {
            if (!used.Contains(m))
                continue;
            renumber[m] = kept.Count;
            kept.Add(materials[m]);
        }
        List<MeshPrimitive> remapped = [.. primitives.Select(p => renumber.TryGetValue(p.Material, out int to)
            ? (p.Material == to ? p : new MeshPrimitive { Geometry = p.Geometry, Material = to })
            : p.Material < 0 ? p : new MeshPrimitive { Geometry = p.Geometry, Material = -1 })];
        return (kept, remapped);
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
}

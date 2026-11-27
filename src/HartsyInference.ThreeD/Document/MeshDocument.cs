using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Document;

/// <summary>A static triangle model as a file describes it, whatever the format: primitives in world space with their materials, plus what the format held that a static model cannot carry. Readers fill it, exporters and operations consume it, so a format or an operation works for every model.</summary>
public sealed class MeshDocument
{
    /// <summary>The triangle batches, in file order.</summary>
    public required IReadOnlyList<MeshPrimitive> Primitives { get; init; }

    /// <summary>The materials the primitives refer to.</summary>
    public IReadOnlyList<MeshMaterial> Materials { get; init; } = [];

    /// <summary>The names of the animations the source held and a static model drops (an unnamed one is <c>animation_i</c>).</summary>
    public IReadOnlyList<string> Animations { get; init; } = [];

    /// <summary>The source had a skin: the geometry is its bind pose and the joints and weights are dropped.</summary>
    public bool HasSkin { get; init; }

    /// <summary>What the reader could not carry into this document, in words a host can show (skipped primitive modes, external images, other UV sets).</summary>
    public IReadOnlyList<string> Notes { get; init; } = [];

    /// <summary>The source's <c>asset.extras</c> object as JSON text (the engine's settings block lives there), carried through so a conversion or an operation keeps it; null when the source had none.</summary>
    public string? AssetExtrasJson { get; init; }

    /// <summary>A copy with other primitives (and, when given, other materials), everything else carried over: animations, skin, notes and the source's extras.</summary>
    public MeshDocument With(IReadOnlyList<MeshPrimitive> primitives, IReadOnlyList<MeshMaterial>? materials = null) => new()
    {
        Primitives = primitives,
        Materials = materials ?? Materials,
        Animations = Animations,
        HasSkin = HasSkin,
        Notes = Notes,
        AssetExtrasJson = AssetExtrasJson,
    };

    /// <summary>Vertices over all primitives.</summary>
    public long VertexCount => Primitives.Sum(p => (long)p.Geometry.VertexCount);

    /// <summary>Triangles over all primitives.</summary>
    public long TriangleCount => Primitives.Sum(p => (long)p.Geometry.TriangleCount);

    /// <summary>Whether any material carries an image.</summary>
    public bool HasTextures => Materials.Any(m => m.Images.Any());

    /// <summary>Whether the source had animation or a skin, which a static format drops.</summary>
    public bool HadMotion => HasSkin || Animations.Count > 0;

    /// <summary>The primitives joined into one mesh. Normals, UVs and vertex colours are kept only when every primitive has them.</summary>
    public Mesh Merge()
    {
        int vertices = checked((int)VertexCount);
        float[] positions = new float[vertices * 3];
        int[] indices = new int[checked((int)TriangleCount * 3)];
        bool normals = Primitives.All(p => p.Geometry.Normals is not null);
        bool uvs = Primitives.All(p => p.Geometry.Uvs is not null);
        bool colors = Primitives.All(p => p.Geometry.VertexColors is not null);
        float[]? n = normals ? new float[vertices * 3] : null;
        float[]? t = uvs ? new float[vertices * 2] : null;
        float[]? c = colors ? new float[vertices * 3] : null;
        int vo = 0, io = 0;
        foreach (MeshPrimitive primitive in Primitives)
        {
            Mesh g = primitive.Geometry;
            g.Vertices.CopyTo(positions, vo * 3);
            if (n is not null)
                g.Normals!.CopyTo(n, vo * 3);
            if (t is not null)
                g.Uvs!.CopyTo(t, vo * 2);
            if (c is not null)
                g.VertexColors!.CopyTo(c, vo * 3);
            foreach (int i in g.Indices)
                indices[io++] = i + vo;
            vo += g.VertexCount;
        }
        return new Mesh { Vertices = positions, Indices = indices, Normals = n, Uvs = t, VertexColors = c };
    }

    /// <summary>The primitives joined into one mesh of positions and indices only, which is what the mesh pipelines take.</summary>
    public Mesh MergeGeometry()
    {
        Mesh merged = Merge();
        return new Mesh { Vertices = merged.Vertices, Indices = merged.Indices };
    }
}

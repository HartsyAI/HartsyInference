using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using HartsyInference.ThreeD.Document;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Export;

/// <summary>Binary STL of the merged mesh: triangles with a facet normal each, geometry only. Materials, textures, colours and UVs are dropped with a note.</summary>
public sealed class StlExporter : IMeshExporter
{
    /// <inheritdoc/>
    public string Id => "stl";

    /// <inheritdoc/>
    public MeshExport Export(MeshDocument document, string name)
    {
        ArgumentNullException.ThrowIfNull(document);
        List<string> notes = MeshExporters.CommonNotes(document, carriesMotion: false);
        if (document.HasTextures || document.Materials.Count > 0 || document.Primitives.Any(p => p.Geometry.VertexColors is not null))
            notes.Add("Materials, textures and colours were dropped; STL is geometry only.");
        Mesh mesh = document.MergeGeometry();
        int triangles = mesh.TriangleCount;
        byte[] data = new byte[84 + triangles * 50];
        Encoding.ASCII.GetBytes("HartsyInference binary STL").CopyTo(data, 0);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(80), (uint)triangles);
        int off = 84;
        for (int t = 0; t < triangles; t++)
        {
            Vector3 a = At(mesh, mesh.Indices[t * 3]), b = At(mesh, mesh.Indices[t * 3 + 1]), c = At(mesh, mesh.Indices[t * 3 + 2]);
            Vector3 normal = Vector3.Cross(b - a, c - a);
            normal = normal.LengthSquared() > 0 ? Vector3.Normalize(normal) : Vector3.Zero;
            Put(data, ref off, normal);
            Put(data, ref off, a);
            Put(data, ref off, b);
            Put(data, ref off, c);
            off += 2;
        }
        return new MeshExport([new MeshExportFile(MeshFileNames.Stem(name) + ".stl", "model/stl", data)], notes);
    }

    private static Vector3 At(Mesh mesh, int vertex) => new(mesh.Vertices[vertex * 3], mesh.Vertices[vertex * 3 + 1], mesh.Vertices[vertex * 3 + 2]);

    private static void Put(byte[] data, ref int off, Vector3 v)
    {
        BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off), v.X);
        BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 4), v.Y);
        BinaryPrimitives.WriteSingleLittleEndian(data.AsSpan(off + 8), v.Z);
        off += 12;
    }
}

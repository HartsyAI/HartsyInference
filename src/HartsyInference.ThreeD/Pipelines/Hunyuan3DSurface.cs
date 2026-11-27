using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>Meshes the Hunyuan3D-2 ShapeVAE occupancy logits, which are positive inside the shape.</summary>
internal static class Hunyuan3DSurface
{
    /// <summary>Extracts the surface at <paramref name="iso"/> from <paramref name="logits"/>, which are negated in place.</summary>
    /// <remarks>MarchingCubes treats values below the iso as inside, so the negation winds the triangles outward.</remarks>
    internal static Mesh Extract(ScalarField3D logits, float iso)
    {
        ArgumentNullException.ThrowIfNull(logits);
        float[] values = logits.Values;
        for (int i = 0; i < values.Length; i++)
            values[i] = -values[i];
        return MeshOps.ComputeVertexNormals(MarchingCubes.Extract(logits, -iso));
    }
}

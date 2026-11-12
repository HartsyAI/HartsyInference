using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Sam3dObjects;

namespace HartsyInference.Engine.Services;

/// <summary>Turns the mesh of SAM 3D's FlexiCubes extraction into a <see cref="Mesh"/> for the GLB writer.</summary>
internal static class Sam3dObjectsMesh
{
    /// <summary>
    /// The mesh is in the decoder's frame (z up), the same one the splats are in before the pipeline turns them: the splats are written as <c>(x, -z, y)</c>, a proper rotation to the y-up frame the
    /// PLY and glTF use, and so is the mesh, so the two outputs agree and the winding is kept. The vertex colour is the sigmoid RGB (the first three of the six colour channels; the last three are
    /// a normal map and are not read).
    /// </summary>
    public static Mesh ToMesh(SamMeshResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        float[] vertices = new float[result.Vertices.Length];
        float[] colours = new float[result.VertexCount * 3];
        for (int v = 0; v < result.VertexCount; v++)
        {
            float x = result.Vertices[v * 3], y = result.Vertices[v * 3 + 1], z = result.Vertices[v * 3 + 2];
            vertices[v * 3] = x;
            vertices[v * 3 + 1] = -z;
            vertices[v * 3 + 2] = y;
            colours[v * 3] = result.Colors[v * 6];
            colours[v * 3 + 1] = result.Colors[v * 6 + 1];
            colours[v * 3 + 2] = result.Colors[v * 6 + 2];
        }

        return new Mesh { Vertices = vertices, Indices = result.Faces, VertexColors = colours };
    }
}

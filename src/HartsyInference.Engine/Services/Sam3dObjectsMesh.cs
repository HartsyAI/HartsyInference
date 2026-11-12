using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.Sam3dObjects;

namespace HartsyInference.Engine.Services;

/// <summary>Turns the mesh of SAM 3D's FlexiCubes extraction into a <see cref="Mesh"/> for the GLB writer.</summary>
internal static class Sam3dObjectsMesh
{
    /// <summary>
    /// The extraction's output frame turned to glTF's (y up, front toward +z): <c>(x, y, z)</c> becomes <c>(x, z, -y)</c>, a proper rotation, so the winding is kept. This is not the splats' turn,
    /// <c>(x, -z, y)</c>: the first attempt used that and the real bust came out upside down; the turn here was found by rendering that bust, which stands upright and faces the camera. The vertex colour
    /// is the sigmoid RGB (the first three of the six colour channels; the last three are a normal map and are not read). It is carried on the mesh but the service does not write it: on the real bust
    /// it came out noisy purple and green, not the stone's grey. Upstream's <c>to_glb</c> reads the same first three channels as vertex colours (<c>use_vertex_color</c>), but its default
    /// path bakes a texture from renders of the splats; neither was compared with this port.
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
            vertices[v * 3 + 1] = z;
            vertices[v * 3 + 2] = -y;
            colours[v * 3] = result.Colors[v * 6];
            colours[v * 3 + 1] = result.Colors[v * 6 + 1];
            colours[v * 3 + 2] = result.Colors[v * 6 + 2];
        }

        return new Mesh { Vertices = vertices, Indices = result.Faces, VertexColors = colours };
    }

    /// <summary>
    /// The shared shape-mesh repair (<see cref="ShapeMeshFinisher"/>) with components under 0.1 % of the total area dropped, 100 times the default. FlexiCubes leaves floating pieces much larger than
    /// the default's specks: on the real bust the default left 360 components and 552 boundary edges (906 and 772 before), 0.1 % left 4 and 466, and the render shows the bust whole. One object, one threshold;
    /// the remaining boundary is larger openings the hole filler's perimeter limit does not close.
    /// </summary>
    public static Mesh Finish(Mesh mesh) => ShapeMeshFinisher.Finish(mesh, clean: true, smallComponentAreaRatio: 1e-3);
}

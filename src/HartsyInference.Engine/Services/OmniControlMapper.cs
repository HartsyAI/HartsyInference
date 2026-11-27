using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.Hunyuan3D.Omni;
using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.Engine.Services;

/// <summary>Checks a <see cref="MeshControl"/> before a checkpoint loads and turns it into the rows the Omni encoder takes.</summary>
internal static class OmniControlMapper
{
    /// <summary>Throws the exception generation would throw for a control Omni cannot use.</summary>
    /// <exception cref="NotSupportedException">The control is missing, or carries data its kind does not take.</exception>
    /// <exception cref="ArgumentException">A value is out of range.</exception>
    public static void Validate(string id, MeshControl? control)
    {
        if (control is null)
            throw new NotSupportedException($"3D model '{id}' needs a control signal; supply MeshRequest.Control (a point cloud, voxels, a bounding box or a skeleton).").ForField(nameof(MeshRequest.Control));
        if (!Enum.IsDefined(control.Kind))
            throw new ArgumentOutOfRangeException(nameof(control), control.Kind, "The control kind must be Point, Voxel, BoundingBox or Pose.").ForField(nameof(MeshRequest.Control));
        switch (control.Kind)
        {
            case MeshControlKind.Point or MeshControlKind.Voxel:
                RequireGeometry(control);
                if (control.Values is not null)
                    throw new NotSupportedException($"A {control.Kind} control takes geometry, not values.").ForField(nameof(MeshRequest.Control));
                break;
            default:
                if (control.Geometry is not null)
                    throw new NotSupportedException($"A {control.Kind} control takes values, not geometry.").ForField(nameof(MeshRequest.Control));
                RequireValues(control);
                break;
        }
    }

    /// <summary>The control rows for <paramref name="control"/> (already validated); <paramref name="seed"/> fixes the surface samples of a voxel control.</summary>
    public static OmniControl ToControl(MeshControl control, int seed)
    {
        switch (control.Kind)
        {
            case MeshControlKind.Point:
            {
                SceneMeshData geometry = control.Geometry!;
                try
                {
                    return new OmniControl(OmniControlKind.Point, OmniControlInputs.Point(PointVertices(geometry), seed));
                }
                catch (Exception error) when (error is HartsyInference.Core.Exceptions.HartsyInferenceException or ArgumentException or InvalidDataException)
                {
                    throw new NotSupportedException($"A Point control needs vertices it can read: {error.Message}", error).ForField(nameof(MeshRequest.Control));
                }
            }
            case MeshControlKind.Voxel:
            {
                ThreeD.Geometry.Mesh mesh;
                try
                {
                    mesh = SceneMeshReader.Read(control.Geometry!.Data, control.Geometry.Format);
                }
                catch (HartsyInference.Core.Exceptions.HartsyInferenceException error)
                {
                    throw new NotSupportedException($"A Voxel control needs a mesh with faces: {error.Message}", error).ForField(nameof(MeshRequest.Control));
                }

                return new OmniControl(OmniControlKind.Voxel, OmniControlInputs.Voxel(mesh.Vertices, mesh.Indices, seed));
            }
            case MeshControlKind.BoundingBox:
                return new OmniControl(OmniControlKind.BoundingBox, OmniControlInputs.BoundingBox(control.Values![0], control.Values[1], control.Values[2]));
            default:
                return new OmniControl(OmniControlKind.Pose, OmniControlInputs.Pose(control.Values!.ToArray()));
        }
    }

    // The points of a point control: the vertices of the file, whether or not it has faces (a point cloud ply, or an obj with only v lines, or a glb mesh).
    private static float[] PointVertices(SceneMeshData geometry)
    {
        string format = geometry.Format.ToLowerInvariant();
        if (format == "ply")
            return PointPlyReader.Read(geometry.Data).Cloud.Positions;
        if (format == "obj")
        {
            List<float> vertices = [];
            foreach (string line in System.Text.Encoding.UTF8.GetString(geometry.Data).Split('\n'))
            {
                string[] parts = line.Split([' ', '\t', '\r'], StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 4 && parts[0] == "v")
                {
                    for (int c = 1; c <= 3; c++)
                    {
                        if (!float.TryParse(parts[c], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float value))
                            throw new ArgumentException($"The Point control has a vertex value that is not a number: '{parts[c]}'.").ForField(nameof(MeshRequest.Control));
                        vertices.Add(value);
                    }
                }
            }

            return [.. vertices];
        }

        return SceneMeshReader.Read(geometry.Data, geometry.Format).Vertices;
    }

    private static void RequireGeometry(MeshControl control)
    {
        if (control.Geometry is null)
            throw new NotSupportedException($"A {control.Kind} control needs Geometry.").ForField(nameof(MeshRequest.Control));
        string format = control.Geometry.Format.ToLowerInvariant();
        bool allowed = format is "obj" or "glb" || (control.Kind == MeshControlKind.Point && format == "ply");
        if (!allowed)
        {
            throw new NotSupportedException(control.Kind == MeshControlKind.Point
                ? $"A Point control reads obj, glb or ply; got '{control.Geometry.Format}'."
                : $"A Voxel control reads an obj or glb mesh with faces; got '{control.Geometry.Format}'. Convert the file first (upstream's voxel demos are ply meshes).").ForField(nameof(MeshRequest.Control));
        }
        if (control.Geometry.Data.Length == 0)
            throw new ArgumentException($"The {control.Kind} control geometry is empty.", nameof(control)).ForField(nameof(MeshRequest.Control));
    }

    private static void RequireValues(MeshControl control)
    {
        IReadOnlyList<float>? values = control.Values;
        if (values is null || values.Count == 0)
            throw new NotSupportedException($"A {control.Kind} control needs Values.").ForField(nameof(MeshRequest.Control));
        if (values.Any(v => !float.IsFinite(v)))
            throw new ArgumentException($"The {control.Kind} control has a value that is not finite.", nameof(control)).ForField(nameof(MeshRequest.Control));
        if (control.Kind == MeshControlKind.BoundingBox)
        {
            if (values.Count != 3 || values.Any(v => v <= 0f || v > 1f))
                throw new ArgumentException("A BoundingBox control is three extents (length, height, width), each in (0, 1].", nameof(control)).ForField(nameof(MeshRequest.Control));
        }
        else if (values.Count % 6 != 0)
        {
            throw new ArgumentException($"A Pose control is six values per bone (x1 y1 z1 x2 y2 z2); got {values.Count}.", nameof(control)).ForField(nameof(MeshRequest.Control));
        }
    }
}

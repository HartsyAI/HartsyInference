using System.Numerics;
using System.Text.Json.Nodes;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Io;

/// <summary>Puts a GLB into a standard frame without touching its vertex data: the scene's root nodes move under one new node whose matrix rotates, centres and scales them, which every glTF viewer applies.</summary>
public static class GlbFrame
{
    /// <summary>The name of the node <see cref="Normalize"/> adds.</summary>
    public const string NodeName = "hartsy_frame";

    /// <summary>Rotates the scene by <paramref name="rotation"/>, centres its bounding box on the origin and scales it so its longest side is <paramref name="longestSide"/>.</summary>
    /// <param name="rotation">A rotation in System.Numerics' row-vector convention; identity keeps the orientation.</param>
    /// <exception cref="HartsyInferenceException">The GLB has no scene, or no POSITION accessor declares its bounds.</exception>
    public static byte[] Normalize(byte[] glb, Matrix4x4 rotation, float longestSide = 1f)
    {
        GlbSummary summary = GlbInspector.Read(glb);
        if (summary.Min is null || summary.Max is null)
            throw new HartsyInferenceException("The GLB declares no POSITION bounds, so its frame cannot be normalised.");
        Vector3 min = new(float.PositiveInfinity), max = new(float.NegativeInfinity);
        for (int corner = 0; corner < 8; corner++)
        {
            Vector3 point = Vector3.Transform(
                new Vector3((corner & 1) == 0 ? summary.Min[0] : summary.Max[0], (corner & 2) == 0 ? summary.Min[1] : summary.Max[1], (corner & 4) == 0 ? summary.Min[2] : summary.Max[2]),
                rotation);
            min = Vector3.Min(min, point);
            max = Vector3.Max(max, point);
        }
        Vector3 extent = max - min;
        float longest = Math.Max(extent.X, Math.Max(extent.Y, extent.Z));
        float scale = longest > 0 ? longestSide / longest : 1f;
        Matrix4x4 matrix = rotation * Matrix4x4.CreateTranslation(-(min + max) / 2f) * Matrix4x4.CreateScale(scale);
        return Wrap(glb, matrix);
    }

    /// <summary>Wraps the default scene's root nodes in a node carrying <paramref name="matrix"/> (row-vector convention).</summary>
    public static byte[] Wrap(byte[] glb, Matrix4x4 matrix)
    {
        JsonObject root = JsonNode.Parse(GlbInspector.SplitJson(glb).Json)!.AsObject();
        if (root["scenes"] is not JsonArray scenes || scenes.Count == 0 || root["nodes"] is not JsonArray nodes)
            throw new HartsyInferenceException("The GLB has no scene to put in a frame.");
        int sceneIndex = root["scene"]?.GetValue<int>() ?? 0;
        JsonObject scene = scenes[sceneIndex]!.AsObject();
        JsonArray children = scene["nodes"] is JsonArray existing ? (JsonArray)existing.DeepClone() : [];
        JsonObject frame = new()
        {
            ["name"] = NodeName,
            ["matrix"] = new JsonArray(
                matrix.M11, matrix.M12, matrix.M13, matrix.M14, matrix.M21, matrix.M22, matrix.M23, matrix.M24,
                matrix.M31, matrix.M32, matrix.M33, matrix.M34, matrix.M41, matrix.M42, matrix.M43, matrix.M44),
            ["children"] = children,
        };
        nodes.Add(frame);
        scene["nodes"] = new JsonArray(nodes.Count - 1);
        return GlbInspector.ReplaceJson(glb, System.Text.Encoding.UTF8.GetBytes(root.ToJsonString()));
    }
}

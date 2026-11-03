using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Cli.Infra;

/// <summary>Writes structured vision results as files: every float array as a little-endian <c>.npy</c>
/// (<see cref="VisionArrayCodec"/>, the same payload the API sends), a PNG preview of each map, and a
/// <c>result.json</c> sidecar with the camera, poses, faces, track summaries, depth kind, the file list and the
/// conventions. A frame stream writes numbered frames and a <c>frames.json</c> index with exact timestamps.</summary>
internal static class VisionOutputWriter
{
    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
    };

    /// <summary>True when <paramref name="result"/> has a field only files can carry.</summary>
    public static bool HasStructured(VisionResult result) =>
        result.DepthMap is not null || result.PointMap is not null || result.Normals is not null || result.Matte is not null
        || result.Camera is not null || result.Poses is not null || result.Faces is not null || result.IntrinsicMaps is not null
        || result.Tracks is not null || result.Views is not null;

    /// <summary>A fresh directory <c>&lt;stem&gt;-vision-NNN</c> under <paramref name="baseDir"/>.</summary>
    public static string NewRunDirectory(string baseDir, string inputPath)
    {
        string stem = Path.GetFileNameWithoutExtension(inputPath.TrimEnd('/', '\\'));
        if (string.IsNullOrWhiteSpace(stem))
            stem = "vision";
        Directory.CreateDirectory(baseDir);
        for (int i = 1; i < 1000; i++)
        {
            string candidate = Path.Combine(baseDir, $"{stem}-vision-{i:D3}");
            if (!Directory.Exists(candidate))
            {
                Directory.CreateDirectory(candidate);
                return candidate;
            }
        }
        string fallback = Path.Combine(baseDir, $"{stem}-vision-{Guid.NewGuid():N}");
        Directory.CreateDirectory(fallback);
        return fallback;
    }

    /// <summary>Writes <paramref name="result"/> into <paramref name="dir"/> with every file name starting
    /// <paramref name="prefix"/>, and returns the files written (sidecar last).</summary>
    public static IReadOnlyList<string> WriteResult(VisionResult result, string dir, string prefix = "")
    {
        List<string> files = [];
        if (result.Image is { } image)
            files.Add(Write(dir, $"{prefix}image.png", PngEncoder.Encode(image)));
        if (result.DepthMap is { } depth)
        {
            files.Add(WriteArray(dir, $"{prefix}depth", depth.Values, Preview.Gray));
            if (depth.ValidMask is { } valid)
                files.Add(WriteArray(dir, $"{prefix}depth-valid", valid, preview: null));
        }
        if (result.PointMap is { } points)
            files.Add(WriteArray(dir, $"{prefix}points", points, preview: null));
        if (result.Normals is { } normals)
            files.Add(WriteArray(dir, $"{prefix}normals", normals, Preview.Normals));
        if (result.Matte is { } matte)
            files.Add(WriteArray(dir, $"{prefix}matte", matte, Preview.Unit));
        foreach ((string name, VisionArray map) in result.IntrinsicMaps ?? new Dictionary<string, VisionArray>())
            files.Add(WriteArray(dir, $"{prefix}{name}", map, Preview.Unit));
        foreach (VisionTrackedObject track in result.Tracks ?? [])
            files.Add(WriteArray(dir, $"{prefix}track-{track.ObjectId}", track.Mask, Preview.Unit));
        for (int i = 0; i < (result.Views?.Count ?? 0); i++)
            files.AddRange(WriteResult(result.Views![i], dir, $"{prefix}view-{i:D2}."));

        if (HasStructured(result))
        {
            var sidecar = new
            {
                depthKind = result.DepthMap?.Kind,
                camera = result.Camera,
                poses = result.Poses,
                faces = result.Faces,
                tracks = result.Tracks?.Select(t => new { t.ObjectId, t.Score, t.Box }).ToList(),
                files = files.Select(Path.GetFileName).ToList(),
                conventions = new
                {
                    cameraFrame = "opencv: x right, y down, z forward",
                    pixelCenter = 0.5,
                    units = "metres for metric depth, points and camera translation; pixels for boxes, keypoints and intrinsics",
                    arrays = "float32 .npy, little-endian, C order",
                },
            };
            files.Add(Write(dir, $"{prefix}result.json", JsonSerializer.SerializeToUtf8Bytes(sidecar, Json)));
        }
        return files;
    }

    /// <summary>Writes each streamed frame (<c>frame-NNNNN.*</c>) and a <c>frames.json</c> index of index, time, source
    /// frame and files. Returns how many frames were written.</summary>
    public static async Task<int> WriteFramesAsync(IAsyncEnumerable<VisionFrameResult> frames, string dir, CancellationToken cancel)
    {
        List<object> index = [];
        await foreach (VisionFrameResult frame in frames.WithCancellation(cancel).ConfigureAwait(false))
        {
            IReadOnlyList<string> files = WriteResult(frame.Result, dir, $"frame-{frame.Index:D5}.");
            index.Add(new { frame.Index, frame.TimestampSeconds, frame.SourceIndex, Files = files.Select(Path.GetFileName).ToList() });
        }
        await File.WriteAllBytesAsync(Path.Combine(dir, "frames.json"), JsonSerializer.SerializeToUtf8Bytes(index, Json), cancel)
            .ConfigureAwait(false);
        return index.Count;
    }

    private enum Preview { Gray, Unit, Normals }

    private static string WriteArray(string dir, string name, VisionArray array, Preview? preview)
    {
        string npy = Write(dir, name + ".npy", VisionArrayCodec.ToNpy(array));
        if (preview is { } kind && PreviewRgb(array, kind) is { } rgb)
            Write(dir, name + ".png", PngEncoder.Encode(rgb, array.Width, array.Height));
        return npy;
    }

    /// <summary>An 8-bit preview: min-max gray for depth, [0, 1] gray (H×W) or colour (H×W×3) for unit maps, and
    /// <c>(n + 1) / 2</c> for normals. Null for shapes with no image form.</summary>
    private static byte[]? PreviewRgb(VisionArray array, Preview kind)
    {
        int channels = array.Shape.Length == 2 ? 1 : array.Shape.Length == 3 ? array.Shape[2] : 0;
        if (channels != 1 && channels != 3)
            return null;
        int pixels = array.Height * array.Width;
        float lo = 0, scale = 1;
        if (kind == Preview.Gray)
        {
            float min = float.PositiveInfinity, max = float.NegativeInfinity;
            foreach (float v in array.Data)
            {
                if (!float.IsFinite(v))
                    continue;
                min = Math.Min(min, v);
                max = Math.Max(max, v);
            }
            lo = float.IsFinite(min) ? min : 0;
            scale = max > min ? 1 / (max - min) : 1;
        }
        byte[] rgb = new byte[pixels * 3];
        for (int p = 0; p < pixels; p++)
        {
            for (int c = 0; c < 3; c++)
            {
                float v = array.Data[p * channels + (channels == 3 ? c : 0)];
                float unit = kind == Preview.Normals ? (v + 1) * 0.5f : (v - lo) * scale;
                rgb[p * 3 + c] = (byte)Math.Clamp((int)MathF.Round(unit * 255f), 0, 255);
            }
        }
        return rgb;
    }

    private static string Write(string dir, string name, byte[] bytes)
    {
        string path = Path.Combine(dir, name);
        File.WriteAllBytes(path, bytes);
        return path;
    }
}

using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using NativeImageData = HartsyInference.Engine.Requests.ImageData;

namespace HartsyInference.API.Endpoints;

/// <summary>The JSON shape of a <see cref="VisionResult"/> on the native vision routes. Images and masks travel as base64
/// PNG (RGBA when there is an alpha plane). Float arrays travel as <c>{ dtype: "float32", shape, data }</c> with
/// <c>data</c> the base64 of the little-endian bytes, row-major with the last dimension fastest: the same payload the CLI
/// writes into <c>.npy</c> files (<see cref="VisionArrayCodec"/>). A result with any structured field also carries
/// <c>conventions</c>.</summary>
internal static class VisionWire
{
    /// <summary>The coordinate and unit conventions every structured field follows (docs/Agents/ENGINE_PATTERNS.md).</summary>
    public static readonly object Conventions = new
    {
        cameraFrame = "opencv: x right, y down, z forward",
        pixelCenter = 0.5,
        units = "metres for metric depth, points and camera translation; pixels for boxes, keypoints and intrinsics",
        arrays = "float32, little-endian, row-major, last dimension fastest",
    };

    /// <summary>The response body for one result.</summary>
    public static object Result(VisionResult result) => new
    {
        embedding = result.Embedding,
        detections = result.Detections,
        masks = result.Masks?.Select(EncodePng).ToList(),
        image = result.Image is { } img ? EncodePng(img) : null,
        depthMap = result.DepthMap is { } depth
            ? new { kind = depth.Kind, values = Array(depth.Values), validMask = Optional(depth.ValidMask) }
            : null,
        pointMap = Optional(result.PointMap),
        normals = Optional(result.Normals),
        matte = Optional(result.Matte),
        camera = result.Camera,
        poses = result.Poses,
        faces = result.Faces,
        intrinsicMaps = result.IntrinsicMaps?.ToDictionary(kv => kv.Key, kv => Array(kv.Value)),
        tracks = result.Tracks?.Select(t => new { objectId = t.ObjectId, score = t.Score, box = t.Box, mask = Array(t.Mask) }).ToList(),
        views = result.Views?.Select(Result).ToList(),
        conventions = IsStructured(result) ? Conventions : null,
    };

    /// <summary>One float array: dtype, shape and base64 little-endian data.</summary>
    public static object Array(VisionArray array) => new
    {
        dtype = VisionArrayCodec.Dtype,
        shape = array.Shape,
        data = Convert.ToBase64String(VisionArrayCodec.ToLittleEndianBytes(array)),
    };

    private static object? Optional(VisionArray? array) => array is null ? null : Array(array);

    private static bool IsStructured(VisionResult r) =>
        r.DepthMap is not null || r.PointMap is not null || r.Normals is not null || r.Matte is not null || r.Camera is not null
        || r.Poses is not null || r.Faces is not null || r.IntrinsicMaps is not null || r.Tracks is not null || r.Views is not null;

    private static string EncodePng(NativeImageData image) => Convert.ToBase64String(PngEncoder.Encode(image));
}

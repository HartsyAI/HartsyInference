using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Face;
using HartsyInference.Vision.FaceDetection;
using EngineIpAdapter = HartsyInference.Diffusion.Adapters.IpAdapter;

namespace HartsyInference.Engine.Features;

/// <summary>A loaded IP-Adapter plus its image encoder, held across generations (the weights are identical for repeat gens and CLIP-Vision-H is a ~600 MB upload — don't thrash). Keyed by the adapter file path. Standard/Plus entries hold CLIP-Vision; FaceID entries hold the ArcFace embedder + pose/face detectors + the companion LoRA path.</summary>
public sealed class IpAdapterCacheEntry : IDisposable
{
    /// <summary>Path the adapter was loaded from; the cache key.</summary>
    public required string FilePath { get; init; }

    /// <summary>The mmap-backed adapter checkpoint.</summary>
    public required IpAdapterFile File { get; init; }

    /// <summary>The constructed, weights-loaded adapter.</summary>
    public required EngineIpAdapter IpAdapter { get; init; }

    /// <summary>CLIP-Vision encoder (standard/Plus, and FaceID-Plus); null for plain FaceID.</summary>
    public ClipVisionEncoder? ClipVision { get; init; }

    /// <summary>Loader owning the CLIP-Vision weights' mmap; null when there is no CLIP-Vision.</summary>
    public SafeTensorsLoader? ClipVisionLoader { get; init; }

    /// <summary>ArcFace IR-50 face embedder (FaceID entries only).</summary>
    public ArcFaceModel? ArcFace { get; init; }

    /// <summary>Loader owning the ArcFace weights' mmap (FaceID entries only).</summary>
    public SafeTensorsLoader? ArcFaceLoader { get; init; }

    /// <summary>YOLO11-pose keypoint detector used for the fallback face alignment (FaceID entries only).</summary>
    public YoloPosePipeline? PosePipeline { get; init; }

    /// <summary>Dedicated YOLOv8-Face detector; when present it supersedes <see cref="PosePipeline"/> for locating and aligning the face, and null falls back to the pose keypoints.</summary>
    public FaceDetector? FaceDetector { get; init; }

    /// <summary>Path of the FaceID companion UNet LoRA (kohya format), or null.</summary>
    public string? FaceIdLoraPath { get; init; }

    /// <summary>Last time this entry served a generation, for cache eviction.</summary>
    public DateTime LastUsedUtc { get; set; } = DateTime.UtcNow;

    private bool _disposed;

    /// <summary>Drops whichever halves are present; disposing the safetensors loaders invalidates the underlying tensors.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        IpAdapter.Dispose();
        File.Dispose();
        ClipVisionLoader?.Dispose();
        FaceDetector?.Dispose();
        PosePipeline?.Dispose();
        ArcFaceLoader?.Dispose();
    }
}

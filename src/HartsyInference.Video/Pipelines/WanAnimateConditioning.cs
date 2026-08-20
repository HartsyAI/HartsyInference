using HartsyInference.Core.Tensors;

namespace HartsyInference.Video.Pipelines;

/// <summary>Host-materialized Wan-Animate conditioning — the VAE-encoded concat conditioning + pose latent and the
/// face-motion features for both CFG branches. These depend only on the driving video, reference image, and geometry
/// (not the prompt/seed/steps), so caching them across generations lets a seed/prompt sweep skip the entire VAE +
/// StyleGAN motion-encode phase. Host tensors re-fault to device inside the denoise loop (the S2V reference-latent
/// cache pattern); the owner (the pipeline cache entry) disposes them.</summary>
public sealed class WanAnimateConditioning : IDisposable
{
    /// <summary>Concat conditioning <c>[1, tp+z, tTotal, hLat, wLat]</c> (mask channels then reference/gray latent).</summary>
    public required Tensor Condition { get; init; }

    /// <summary>Pose-video latent <c>[1, poseC, tLat, hLat, wLat]</c> added in-DiT to the latent frames.</summary>
    public required Tensor PoseLatent { get; init; }

    /// <summary>Positive-branch face-motion features <c>[gt, N+1, dim]</c>.</summary>
    public required Tensor MotionCond { get; init; }

    /// <summary>Negative-branch (black-face) face-motion features <c>[gt, N+1, dim]</c>; null when the entry was built
    /// for a single-forward (guidance ≤ 1) denoise, which makes it un-reusable for a CFG one.</summary>
    public Tensor? MotionUncond { get; init; }

    /// <summary>Latent frames the chunked extension's motion prefix occupied when this was built; 0 for a normal
    /// (first/only) chunk. Non-zero makes the entry un-reusable — the prefix differs on every chunk.</summary>
    public int RefMotionLatentLength { get; init; }

    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Condition?.Dispose();
        PoseLatent?.Dispose();
        MotionCond?.Dispose();
        MotionUncond?.Dispose();
    }
}

using HartsyInference.Conditioning;
using HartsyInference.Core.Tensors;
using HartsyInference.Interactive.Pipelines;
using HartsyInference.Video;

namespace HartsyInference.Interactive.Sessions;

/// <summary>Drives <see cref="HunyuanGameCraftPipeline"/> as a live <see cref="IFrameStepper"/>: each
/// <see cref="Step"/> consumes one per-chunk action (WASD/camera payload), denoises + decodes one 33-frame chunk,
/// feeds the denoised latent forward as the next chunk's history, and returns the decoded frames. Pre-computed
/// text conditioning (Llava tokens + pooled CLIP) is supplied at construction.
/// <para><b>Numerics validation-pending</b> — history selection + mask schedule across chunks are validation-gated
/// (v1 uses reference-frame mode each chunk). Real use is GPU + checkpoint; structurally complete here.</para></summary>
public sealed class GameCraftFrameStepper : IFrameStepper, IDisposable
{
    private readonly HunyuanGameCraftPipeline _pipeline;
    private readonly Tensor _promptEmbeds, _pooled, _negEmbeds, _negPooled;
    private readonly float[] _mask;
    private readonly int _steps, _seed;
    private readonly float _guidance;
    private Tensor _history;
    private long _frameCounter;
    private int _chunk;

    public int FrameWidth { get; }
    public int FrameHeight { get; }

    public GameCraftFrameStepper(HunyuanGameCraftPipeline pipeline, Tensor promptEmbeds, Tensor pooled,
        Tensor negEmbeds, Tensor negPooled, Tensor seedHistory, int frameWidth, int frameHeight,
        int steps = 50, float guidance = 2.0f, int seed = 0)
    {
        _pipeline = pipeline;
        _promptEmbeds = promptEmbeds; _pooled = pooled; _negEmbeds = negEmbeds; _negPooled = negPooled;
        _history = seedHistory;
        FrameWidth = frameWidth; FrameHeight = frameHeight;
        _steps = steps; _guidance = guidance; _seed = seed;
        int tLat = (int)seedHistory.Shape[2];
        _mask = new float[tLat];
        _mask[0] = 1f; // reference-frame mode (frame 0 is history) — [VG] cross-chunk schedule
    }

    public IReadOnlyList<VideoFrame> Step(in ActionInput action)
    {
        Tensor latent = _pipeline.DenoiseChunk(_promptEmbeds, _pooled, _negEmbeds, _negPooled,
            _history, _mask, action.Payload.Span, _steps, _guidance, _seed + _chunk);
        byte[][] rgb = _pipeline.DecodeLatentToFrames(latent);

        // Carry the denoised latent forward as the next chunk's history.
        _history.Dispose();
        _history = latent;
        _chunk++;

        List<VideoFrame> frames = new(rgb.Length);
        foreach (byte[] f in rgb) frames.Add(new VideoFrame((int)_frameCounter++, FrameWidth, FrameHeight, f));
        return frames;
    }

    public void ApplyQualityProfile(QualityProfile profile) { /* steps/CFG trade-off — [VG] */ }

    public void Dispose() => _history.Dispose();
}

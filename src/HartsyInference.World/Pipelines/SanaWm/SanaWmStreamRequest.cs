using HartsyInference.Core.Tensors;
using HartsyInference.World.Camera;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>Everything a SANA-WM stream is opened from. The camera is either a fixed path (<see cref="CameraToWorld"/>, one
/// pose per frame of a bounded stream) or live: the path starts at the identity and grows a chunk at a time under the
/// actions sent to the stream, with <see cref="InitialActions"/> queued first.</summary>
public sealed record SanaWmStreamRequest
{
    /// <summary>Upstream's distilled four-step schedule.</summary>
    public static IReadOnlyList<int> DefaultDenoisingTimesteps { get; } = [1000, 960, 889, 727, 0];

    /// <summary>The seed image as RGB24, row-major, <see cref="ImageWidth"/> by <see cref="ImageHeight"/>.</summary>
    public required byte[] ImageRgb { get; init; }

    /// <summary>Width of the seed image in pixels.</summary>
    public required int ImageWidth { get; init; }

    /// <summary>Height of the seed image in pixels.</summary>
    public required int ImageHeight { get; init; }

    /// <summary>The prompt; the CHI prefix is added by the text path.</summary>
    public required string Prompt { get; init; }

    /// <summary>The negative prompt for the unconditional branch; read only when <see cref="Guidance"/> is above 1.</summary>
    public string NegativePrompt { get; init; } = "";

    /// <summary>Intrinsics in seed-image pixels: one entry for every frame, or, with a fixed path, a per-frame sequence fitted
    /// over the stream's frames.</summary>
    public required IReadOnlyList<PinholeIntrinsics> Intrinsics { get; init; }

    /// <summary>A fixed camera path: one camera-to-world pose per frame (16 row-major floats), frame 0 first. Needs
    /// <see cref="Frames"/>. Null steers the camera live.</summary>
    public IReadOnlyList<float[]>? CameraToWorld { get; init; }

    /// <summary>An action script queued before the first chunk of a live stream (<c>&lt;keys&gt;-&lt;frames&gt;</c> segments).</summary>
    public string? InitialActions { get; init; }

    /// <summary>Frames to generate, counting the seed frame, on the <c>8n + 1</c> grid with at least one chunk of latents;
    /// null runs until the stream is disposed.</summary>
    public int? Frames { get; init; }

    /// <summary>The distilled student's timesteps, descending to 0.</summary>
    public IReadOnlyList<int> DenoisingTimesteps { get; init; } = DefaultDenoisingTimesteps;

    /// <summary>Classifier-free guidance scale; at or below 1 only the conditional branch runs (the student's training setup).</summary>
    public double Guidance { get; init; } = 1.0;

    /// <summary>Seed of the per-chunk noise.</summary>
    public long Seed { get; init; }

    /// <summary>Latent frames per chunk after the first; the first chunk holds one more (the seed frame).</summary>
    public int ChunkFrames { get; init; } = SanaWmStreamingPipeline.DefaultChunkFrames;

    /// <summary>Earlier chunks the softmax blocks attend to (upstream <c>num_cached_blocks</c>); zero or less keeps all.</summary>
    public int CachedChunks { get; init; } = SanaWmStreamingPipeline.DefaultCachedChunks;

    /// <summary>Keeps the first chunk in the softmax window for the whole stream (upstream <c>sink_token</c>).</summary>
    public bool SinkChunk { get; init; } = true;

    /// <summary>Replaces the seeded noise of a chunk with a host F32 <c>[1, 128, T, 22, 40]</c> tensor (for parity against a
    /// recorded upstream latent); a null result draws the seeded noise. The tensor is borrowed for the call.</summary>
    public Func<int, Tensor?>? ChunkNoise { get; init; }

    /// <summary>Replaces the encoded seed latent with a host F32 <c>[1, 128, 1, 22, 40]</c> tensor; the VAE encoder stage is
    /// skipped. Borrowed; the caller keeps it.</summary>
    public Tensor? FirstLatent { get; init; }

    /// <summary>Observes intermediates by name: <c>first_latent</c>, and per chunk <c>chunk{k}_noise</c> and
    /// <c>chunk{k}_latent</c>. The tensors are borrowed for the duration of the call.</summary>
    public Action<string, Tensor>? Tap { get; init; }
}

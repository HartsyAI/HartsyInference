using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Streaming;
using HartsyInference.World.Camera;
using HartsyInference.World.Pipelines.SanaWm;

namespace HartsyInference.World.Pipelines;

/// <summary>SANA-WM streaming generation (upstream <c>generate_streaming</c>): the chunk-causal distilled student rolls the
/// world out a chunk of latent frames at a time with the self-forcing sampler, optionally the chunk-causal LTX-2 refiner
/// refines each chunk, and the causal LTX-2 VAE decodes it, so frames are available while later chunks are still to come
/// and the camera can be steered between chunks. <see cref="Open"/> encodes the prompts and the seed frame, one component
/// at a time, then loads the DiT, the decoder and (when asked for) the refiner side by side and hands back the running
/// <see cref="SanaWmStream"/>.</summary>
public sealed class SanaWmStreamingPipeline
{
    /// <summary>Latent frames per chunk after the first (upstream <c>chunk_size</c>).</summary>
    public const int DefaultChunkFrames = 3;

    /// <summary>Earlier chunks the softmax blocks attend to (upstream <c>num_cached_blocks</c>).</summary>
    public const int DefaultCachedChunks = 2;

    /// <summary>Upstream's default streaming length: 241 frames, about 15 seconds at 16 fps.</summary>
    public const int DefaultFrames = 241;

    /// <summary>Frames one chunk after the first adds: the frame-count grid of a stream is this many per step, plus the seed frame.</summary>
    public const int FramesPerChunk = DefaultChunkFrames * SanaWmCameraConditioning.TemporalStride;

    private readonly ISanaWmStreamingComponents _components;

    /// <summary>Creates the pipeline over <paramref name="components"/>.</summary>
    public SanaWmStreamingPipeline(ISanaWmStreamingComponents components)
    {
        ArgumentNullException.ThrowIfNull(components);
        _components = components;
    }

    /// <summary>Encodes the conditioning and opens the stream. A stage that is open when this fails or is cancelled is released
    /// before the exception leaves.</summary>
    public SanaWmStream Open(SanaWmStreamRequest request, CancellationToken cancel)
    {
        Validate(request);
        SelfForcingFlowSchedule schedule = new(request.DenoisingTimesteps.ToArray());
        int? latentFrames = request.Frames is int frames ? (frames - 1) / SanaWmCameraConditioning.TemporalStride + 1 : null;
        ChunkPlanner planner = latentFrames is int total
            ? new ChunkPlanner(request.ChunkFrames, ChunkSplitStrategy.FirstAbsorbsRemainder, total)
            : new ChunkPlanner(request.ChunkFrames, ChunkSplitStrategy.FirstChunkPlusOne);
        SanaWmStreamCamera camera = SanaWmStreamCamera.Create(request, SanaWmPipeline.Width, SanaWmPipeline.Height);
        cancel.ThrowIfCancellationRequested();

        ISanaWmRefinerComponents? refinerComponents = null;
        if (request.Refine)
        {
            refinerComponents = _components as ISanaWmRefinerComponents
                ?? throw new NotSupportedException("These SANA-WM components have no refiner stage.");
        }

        SanaWmTextConditioning? text = null;
        Tensor? refinerPrompt = null;
        Tensor? firstLatent = null;
        SanaWmStage<ISanaWmChunkDenoiser>? denoiser = null;
        SanaWmStage<IStreamingVideoDecoder>? decoder = null;
        SanaWmStage<IStreamingLatentRefiner>? refiner = null;
        bool opened = false;
        try
        {
            text = SanaWmEncoding.EncodeText(_components.OpenTextEncoder, request.Prompt, request.NegativePrompt);
            cancel.ThrowIfCancellationRequested();
            if (refinerComponents is not null)
            {
                // Gemma-3-12B is larger than everything the stream keeps resident: it loads and leaves before the DiT.
                using (SanaWmStage<ISanaWmRefinerTextEncoder> encoder = refinerComponents.OpenRefinerTextEncoder())
                    refinerPrompt = encoder.Component.Encode(request.Prompt);
                cancel.ThrowIfCancellationRequested();
            }

            firstLatent = FirstLatent(request);
            cancel.ThrowIfCancellationRequested();
            denoiser = _components.OpenChunkDenoiser(text, TokenWindowPolicy.RecentChunks(request.CachedChunks, request.SinkChunk));
            cancel.ThrowIfCancellationRequested();
            decoder = _components.OpenStreamingDecoder();
            if (refinerComponents is not null)
            {
                // Last, so its weights are placed against what the DiT and the decoder left free. The seed frame's latent is
                // its sink, which does not depend on the stream: captured here, ahead of the first chunk.
                cancel.ThrowIfCancellationRequested();
                refiner = refinerComponents.OpenRefiner(refinerPrompt!, request.Refiner);
                refiner.Component.SetSink(firstLatent, cancel);
            }

            SanaWmStream stream = new(request, schedule, planner, camera, text, firstLatent, denoiser, decoder, refiner);
            opened = true;
            return stream;
        }
        finally
        {
            refinerPrompt?.Dispose();
            if (!opened)
            {
                refiner?.Dispose();
                decoder?.Dispose();
                denoiser?.Dispose();
                firstLatent?.Dispose();
                text?.Dispose();
            }
        }
    }

    private Tensor FirstLatent(SanaWmStreamRequest request)
    {
        if (request.FirstLatent is not Tensor injected)
        {
            return SanaWmEncoding.EncodeFirstFrame(_components.OpenImageEncoder, request.ImageRgb, request.ImageWidth,
                request.ImageHeight, request.Tap);
        }

        TensorShape expected = SanaWmEncoding.FirstLatentShape;
        if (injected.DType != DType.F32 || injected.Shape != expected)
            throw new ArgumentException($"The injected seed latent must be F32 {expected}; got {injected.DType} {injected.Shape}.",
                nameof(request));
        Tensor copy = new(expected, DType.F32, DeviceKind.Cpu);
        injected.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
        SanaWmPipeline.RoundToBf16(copy.AsSpan<float>());
        request.Tap?.Invoke("first_latent", copy);
        return copy;
    }

    private static void Validate(SanaWmStreamRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.ImageRgb);
        ArgumentNullException.ThrowIfNull(request.Prompt);
        ArgumentNullException.ThrowIfNull(request.NegativePrompt);
        ArgumentNullException.ThrowIfNull(request.Intrinsics);
        ArgumentNullException.ThrowIfNull(request.DenoisingTimesteps);
        int stride = SanaWmCameraConditioning.TemporalStride;
        if (request.ChunkFrames < 1)
            throw new ArgumentException($"A chunk holds at least one latent frame; {request.ChunkFrames} requested.", nameof(request));
        if (request.Frames is int frames)
        {
            if (frames < 1 || (frames - 1) % stride != 0)
                throw new ArgumentException($"A SANA-WM stream renders {stride}n+1 frames; {frames} requested.", nameof(request));
            if ((frames - 1) / stride + 1 <= request.ChunkFrames)
                throw new ArgumentException($"A stream needs more than one chunk of {request.ChunkFrames} latent frames; "
                    + $"{frames} frames give {(frames - 1) / stride + 1}.", nameof(request));
        }

        if (request.CameraToWorld is IReadOnlyList<float[]> path)
        {
            if (request.Frames is not int bounded)
                throw new ArgumentException("A fixed camera path needs a frame count.", nameof(request));
            if (path.Count != bounded)
                throw new ArgumentException($"The camera path has {path.Count} poses for {bounded} frames.", nameof(request));
            if (request.InitialActions is not null)
                throw new ArgumentException("A stream takes a fixed camera path or live actions, not both.", nameof(request));
        }
        else if (request.Intrinsics.Count != 1)
        {
            throw new ArgumentException("A live camera path takes one set of intrinsics for every frame.", nameof(request));
        }

        if (request.Intrinsics.Count == 0)
            throw new ArgumentException("SANA-WM needs intrinsics for its camera path.", nameof(request));
        if (request.Refine)
        {
            ArgumentNullException.ThrowIfNull(request.Refiner);
            if (request.Refiner.WindowFrames < 1 || request.Refiner.HistoryLayerStride < 1)
                throw new ArgumentException("The refiner's window holds at least the seed frame and its layer stride is at least 1.",
                    nameof(request));
        }

        if (!double.IsFinite(request.Guidance) || request.Guidance < 0)
            throw new ArgumentException($"Guidance must be a finite non-negative scale; {request.Guidance} requested.", nameof(request));
        if (request.ImageWidth < 1 || request.ImageHeight < 1 || request.ImageRgb.Length != request.ImageWidth * request.ImageHeight * 3)
            throw new ArgumentException($"The seed image is not {request.ImageWidth}x{request.ImageHeight} RGB24.", nameof(request));
    }
}

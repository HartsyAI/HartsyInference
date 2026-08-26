using HartsyInference.Core.Configuration;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Services;

/// <summary>Video-generation service: resolves the checkpoint's family recipe from the <see cref="VideoRecipeRegistry"/>, constructs (and caches) the pipeline, and returns the decoded frames plus the soundtrack <see cref="VideoAudioResolver"/> selects for them. Composition features are applied by the feature-resolver phase (E-IMG-4); a request that sets one is rejected rather than silently ignored.</summary>
public sealed class VideoService : IVideoService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal VideoService(InferenceEngine engine) => _engine = engine;

    /// <summary>The conditioning <paramref name="request"/> asks for, one bit per object actually set.</summary>
    internal static VideoFeatures RequestedFeatures(VideoRequest request)
    {
        VideoFeatures features = VideoFeatures.None;
        if (request.InitImage is not null)
        {
            features |= VideoFeatures.InitImage;
        }
        if (request.VideoEndFrame is not null)
        {
            features |= VideoFeatures.EndFrame;
        }
        if (request.Loras is not null)
        {
            features |= VideoFeatures.Lora;
        }
        if (request.ReferenceImages is { Count: > 0 })
        {
            features |= VideoFeatures.ReferenceImages;
        }
        if (request.ReferenceVideos is { Count: > 0 })
        {
            features |= VideoFeatures.ReferenceVideos;
        }
        if (request.ReferenceAudios is { Count: > 0 })
        {
            features |= VideoFeatures.ReferenceAudios;
        }
        if (request.DrivingVideo is not null || request.DrivingPoseVideo is not null || request.DrivingFaceVideo is not null)
        {
            features |= VideoFeatures.DrivingVideo;
        }
        return features;
    }

    /// <summary>Throws naming the family and the exact conditioning it cannot apply. Before this existed a video family that ignored <c>InitImage</c> returned a text-to-video clip with nothing to indicate the image was discarded.</summary>
    internal static void RejectUnsupported(ModelSpec spec, VideoRequest request)
    {
        VideoFeatures requested = RequestedFeatures(request);
        if (requested == VideoFeatures.None)
        {
            return;
        }
        VideoFeatures missing = requested & ~InferenceEngine.SupportedVideoFeatures(spec);
        if (missing != VideoFeatures.None)
        {
            throw new NotSupportedException(
                $"Video model family '{InferenceEngine.FamilyIdFor(spec)}' does not support: {missing}.");
        }
    }

    /// <inheritdoc/>
    public async Task<VideoGenerationResult> GenerateAsync(ModelSpec spec, VideoRequest request, IProgress<StepPreview>? progress = null,
        CancellationToken cancel = default)
    {
        RejectUnsupported(spec, request);

        return await Task.Run(
            () =>
            {
                using IDisposable vramScope = VramPolicyScope.Push(
                    request.Vram is null ? null : VramPolicyRegistry.Resolve(_engine.Backend, request.Vram));
                // Same lifetime as the VRAM scope, and for the same reason: settings read DURING the
                // generation would otherwise answer from the machine's configuration, since the pipeline is cached.
                using IDisposable settingsScope = KnobProfileScope.Push(request.Settings?.Resolve());
                IVideoRecipePipeline pipeline = _engine.GetOrConstructVideoRecipe(spec, request);
                VideoRequest resolved = InferenceEngine.VideoDefaultsFor(spec).Apply(request);
                VideoGenerationResult result = pipeline.Generate(resolved, progress, cancel);
                // An explicit request fps always wins; else a pipeline-pinned rate (e.g. a decoded driving clip's,
                // so motion speed matches the driving video); else the family default the resolver applied.
                int? fps = request.Fps ?? result.Fps ?? resolved.Fps;
                if (result.Fps != fps)
                {
                    result = result with { Fps = fps };
                }
                double seconds = VideoAudioResolver.VideoSeconds(result.Frames.Count, fps);
                return VideoAudioResolver.Resolve(result, resolved, seconds);
            },
            cancel).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<VideoFrame> GenerateFramesAsync(ModelSpec spec, VideoRequest request,
        IProgress<StepPreview>? progress = null, [EnumeratorCancellation] CancellationToken cancel = default)
    {
        RejectUnsupported(spec, request);
        IVideoRecipePipeline pipeline = _engine.GetOrConstructVideoRecipe(spec, request);
        if (!pipeline.SupportsStreaming)
        {
            throw new NotSupportedException($"Video model family '{InferenceEngine.FamilyIdFor(spec)}' does not support streaming generation.");
        }
        VideoRequest resolved = InferenceEngine.VideoDefaultsFor(spec).Apply(request);

        // Bridges the pipeline's own IAsyncEnumerable (whose first MoveNextAsync blocks synchronously through the
        // whole denoise loop — there's no await point until the VAE decode phase starts yielding groups) onto a
        // background thread via a bounded channel, so the calling thread (an ASP.NET request handler when the
        // extension awaits this) isn't pinned for the denoise duration. Mirrors GenerateAsync's Task.Run, but
        // needs a channel instead of a single Task<T> because this method yields incrementally rather than once.
        Channel<VideoFrame> channel = Channel.CreateBounded<VideoFrame>(
            new BoundedChannelOptions(4) { SingleReader = true, SingleWriter = true, FullMode = BoundedChannelFullMode.Wait });
        Task producer = Task.Run(async () =>
        {
            try
            {
                await foreach (VideoFrame frame in pipeline.GenerateFramesAsync(resolved, progress, cancel).WithCancellation(cancel).ConfigureAwait(false))
                {
                    await channel.Writer.WriteAsync(frame, cancel).ConfigureAwait(false);
                }
                channel.Writer.TryComplete();
            }
            catch (Exception ex)
            {
                channel.Writer.TryComplete(ex);
            }
        }, cancel);

        await foreach (VideoFrame frame in channel.Reader.ReadAllAsync(cancel).ConfigureAwait(false))
        {
            yield return frame;
        }
        // Observes the producer's result: rethrows whatever TryComplete(ex) captured (the channel reader itself
        // already threw that same exception via ReadAllAsync above, but awaiting here also surfaces a producer-side
        // failure that happened AFTER the channel completed successfully — there isn't one today, but this is the
        // same "don't let a background Task's fault go unobserved" discipline as GenerateAsync's Task.Run).
        await producer.ConfigureAwait(false);
    }
}

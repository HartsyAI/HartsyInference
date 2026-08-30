using System.Runtime.CompilerServices;
using System.Threading.Channels;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Cuda;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Video-generation service: resolves the checkpoint's family recipe from the <see cref="VideoRecipeRegistry"/>, constructs (and caches) the pipeline, and returns the decoded frames plus the soundtrack <see cref="VideoAudioResolver"/> selects for them. Composition features are applied by the feature-resolver phase (E-IMG-4); a request that sets one is rejected rather than silently ignored.</summary>
public sealed class VideoService : IVideoService, IVideoPlanningService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal VideoService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public async Task<VideoPlan> PlanAsync(ModelSpec spec, VideoRequest request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(request);
        VideoRequestExecutionBinding binding = VideoRequestExecutionBinding.Create(spec, request);
        ModelSpec plannedSpec = binding.Model;
        VideoRequest plannedRequest = binding.Request;
        string familyId = InferenceEngine.ResolveVideoFamilyId(plannedSpec);
        VideoDefaults defaults = InferenceEngine.VideoDefaultsFor(plannedSpec);
        VideoFeatures features = InferenceEngine.SupportedVideoFeatures(plannedSpec);
        VideoPlan plan = await VideoProfileResolver.ResolveAsync(
                plannedSpec, plannedRequest, familyId, defaults, features, cancel)
            .ConfigureAwait(false);
        plan = plan with
        {
            SourceRequest = request,
            RequestBinding = binding,
        };
        plan = ApplyRegisteredVideoFamilyCheck(plan, familyId);
        if (plan.Issues.Any(issue => issue.Code == "video.family.unregistered"))
        {
            return VideoRequestExecutionBinding.BindPlan(plan);
        }
        plan = ApplyExperimentalH3VsaGate(plan);
        plan = ApplyExperimentalH3ExpansionGate(plan, plannedRequest);
        if (plan.Issues.Any(issue => issue.Code == "video.vsa.experimental_disabled"))
        {
            return VideoRequestExecutionBinding.BindPlan(plan);
        }
        bool sparseBackendSupported = true;
        string? sparseBackendFailure = null;
        if (plan.Profile.Attention != VideoAttentionKind.Dense)
        {
            sparseBackendSupported = string.Equals(
                BackendFactory.Resolve(_engine.BackendSelector), "cuda", StringComparison.OrdinalIgnoreCase);
            if (sparseBackendSupported)
            {
                try
                {
                    sparseBackendSupported = _engine.Backend.SupportsVideoSparseAttention;
                }
                catch (Exception error) when (error is CudaException or DllNotFoundException
                    or EntryPointNotFoundException or BadImageFormatException or PlatformNotSupportedException
                    or NotSupportedException)
                {
                    RecordSparseBackendFailure(error);
                }
            }
        }

        void RecordSparseBackendFailure(Exception error)
        {
            sparseBackendSupported = false;
            sparseBackendFailure = error.Message;
            Logs.Error("[VideoPlan] Native sparse-attention backend capability probe failed.", error);
        }

        if (plan.Profile.Attention != VideoAttentionKind.Dense && !sparseBackendSupported)
        {
            List<VideoPlanIssue> issues = new List<VideoPlanIssue>(plan.Issues)
            {
                new VideoPlanIssue
                {
                    Code = "video.vsa.backend_unsupported",
                    Severity = VideoPlanIssueSeverity.Error,
                    Message = $"Profile '{plan.Profile.Id}' requires native CUDA sparse attention; "
                        + $"backend '{_engine.BackendDescription}' cannot execute it"
                        + (sparseBackendFailure is null ? "." : $": {sparseBackendFailure}"),
                    Field = nameof(VideoRequest.SparseAttentionPolicy),
                },
            };
            plan = plan with { Issues = issues };
        }
        return VideoRequestExecutionBinding.BindPlan(plan);
    }

    /// <summary>Rejects families that can be identified but have no video execution recipe. This uses the exact
    /// checkpoint-aware family id selected by <see cref="InferenceEngine.ResolveVideoFamilyId"/> and only inspects
    /// the recipe registry; it never constructs a pipeline, backend, or model weights.</summary>
    internal static VideoPlan ApplyRegisteredVideoFamilyCheck(VideoPlan plan, string familyId)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentException.ThrowIfNullOrWhiteSpace(familyId);
        if (VideoRecipeRegistry.Resolve(familyId) is not null)
        {
            return plan;
        }

        List<VideoPlanIssue> issues = new List<VideoPlanIssue>(plan.Issues)
        {
            new VideoPlanIssue
            {
                Code = "video.family.unregistered",
                Severity = VideoPlanIssueSeverity.Error,
                Message = $"Video model family '{familyId}' has no registered video recipe. "
                    + $"Currently drivable: {string.Join(", ", VideoRecipeRegistry.RegisteredNames)}.",
                Field = nameof(ModelSpec.Requested),
            },
        };
        return plan with { Issues = issues };
    }

    /// <summary>Applies the temporary production-release gate to a resolved VSA plan.</summary>
    internal static VideoPlan ApplyExperimentalH3VsaGate(VideoPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        if (plan.Profile.Attention == VideoAttentionKind.Dense || EngineKnobs.IsH3VsaExperimentalEnabled())
        {
            return plan;
        }
        List<VideoPlanIssue> issues = new List<VideoPlanIssue>(plan.Issues)
        {
            new VideoPlanIssue
            {
                Code = "video.vsa.experimental_disabled",
                Severity = VideoPlanIssueSeverity.Error,
                Message = $"Profile '{plan.Profile.Id}' requires the experimental MiniMax-H3 VSA runtime. "
                    + "Its real-weight, performance, and peak-memory release gates have not passed. "
                    + "Set HARTSY_EXPERIMENTAL_H3_VSA=1 only for controlled validation.",
                Field = nameof(VideoRequest.SparseAttentionPolicy),
            },
        };
        return plan with { Issues = issues };
    }

    /// <summary>Prevents structurally implemented expansion paths from becoming release claims before their
    /// operator-provided real-model generation and output-inspection gates have passed.</summary>
    internal static VideoPlan ApplyExperimentalH3ExpansionGate(VideoPlan plan, VideoRequest request)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(request);
        if (!string.Equals(plan.Profile.FamilyId, "minimax-h3", StringComparison.OrdinalIgnoreCase)
            || EngineKnobs.IsH3ExpansionExperimentalEnabled())
        {
            return plan;
        }

        List<string> pending = [];
        if (plan.Profile.Acceleration is VideoAccelerationKind.Turbo or VideoAccelerationKind.Pdd)
        {
            pending.Add(plan.Profile.Acceleration.ToString());
        }
        if (plan.Profile.Task == VideoTaskFamily.Hybrid)
        {
            pending.Add("Hybrid");
        }
        if (request.Controls is { Count: > 0 })
        {
            pending.Add("Fun ControlNet");
        }
        if (plan.ComponentFormats.TryGetValue("videoVae", out string? videoVaeFormat)
            && videoVaeFormat.Contains("int8", StringComparison.OrdinalIgnoreCase))
        {
            pending.Add("int8 video VAE");
        }
        if (pending.Count == 0)
        {
            return plan;
        }

        List<VideoPlanIssue> issues = new List<VideoPlanIssue>(plan.Issues)
        {
            new VideoPlanIssue
            {
                Code = "video.h3_expansion.experimental_disabled",
                Severity = VideoPlanIssueSeverity.Error,
                Message = $"Profile '{plan.Profile.Id}' requires validation-pending {string.Join(", ", pending.Distinct(StringComparer.Ordinal))}. "
                    + "Its operator-provided real-generation and output-inspection release gate has not passed. "
                    + "Set numerics.h3ExpansionExperimental=true (or HARTSY_EXPERIMENTAL_H3_EXPANSION=1) only for controlled validation.",
                Field = nameof(ModelSpec.ProfileId),
            },
        };
        return plan with { Issues = issues };
    }

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
        if (request.Guides is { Count: > 0 })
        {
            features |= VideoFeatures.Guides;
        }
        if (request.VideoDenoiseMask is not null)
        {
            features |= VideoFeatures.VideoDenoiseMask;
        }
        if (request.AudioDenoiseMask is not null)
        {
            features |= VideoFeatures.AudioDenoiseMask;
        }
        if (request.Controls is { Count: > 0 })
        {
            features |= VideoFeatures.VideoControlNet;
            if (request.Controls.Any(control => control.Kind == VideoControlKind.Inpaint))
            {
                features |= VideoFeatures.VideoInpaint;
            }
        }
        if (request.DrivingVideo is not null || request.DrivingPoseVideo is not null || request.DrivingFaceVideo is not null)
        {
            features |= VideoFeatures.DrivingVideo;
        }
        return features;
    }

    /// <inheritdoc/>
    public async Task<VideoGenerationResult> GenerateAsync(ModelSpec spec, VideoRequest request, IProgress<StepPreview>? progress = null,
        CancellationToken cancel = default)
    {
        VideoPlan plan = await PlanAsync(spec, request, cancel).ConfigureAwait(false);
        return await GenerateAsync(plan, request, progress, cancel).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async Task<VideoGenerationResult> GenerateAsync(VideoPlan plan, VideoRequest request,
        IProgress<StepPreview>? progress = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(request);
        VideoPlan executionPlan = VideoRequestExecutionBinding.RequirePlannedState(plan);
        VideoRequest executionRequest = VideoRequestExecutionBinding.RequireUnchanged(executionPlan, request);
        LogPlanWarnings(executionPlan);
        executionPlan.ThrowIfInvalid();
        VideoRequest effectiveRequest = executionPlan.EffectiveSettings.Apply(executionRequest);

        return await Task.Run(
            () =>
            {
                using IDisposable vramScope = VramPolicyScope.Push(
                    effectiveRequest.Vram is null ? null : VramPolicyRegistry.Resolve(_engine.Backend, effectiveRequest.Vram));
                // Same lifetime as the VRAM scope, and for the same reason: settings read DURING the
                // generation would otherwise answer from the machine's configuration, since the pipeline is cached.
                using IDisposable settingsScope = KnobProfileScope.Push(effectiveRequest.Settings?.Resolve());
                VideoArtifactFileBinding.RequireUnchanged(executionPlan);
                IVideoRecipePipeline pipeline = _engine.GetOrConstructVideoRecipe(
                    executionPlan.Model, effectiveRequest, executionPlan);
                VideoGenerationResult result = pipeline.Generate(effectiveRequest, progress, cancel);
                // An explicit request fps always wins; else a pipeline-pinned rate (e.g. a decoded driving clip's,
                // so motion speed matches the driving video); else the family default the resolver applied.
                int? fps = executionRequest.Fps ?? result.Fps ?? effectiveRequest.Fps;
                if (result.Fps != fps)
                {
                    result = result with { Fps = fps };
                }
                double seconds = VideoAudioResolver.VideoSeconds(result.Frames.Count, fps);
                VideoGenerationResult withAudio = VideoAudioResolver.Resolve(result, effectiveRequest, seconds);
                return withAudio with { Execution = BuildExecutionSummary(executionPlan) };
            },
            cancel).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<VideoFrame> GenerateFramesAsync(ModelSpec spec, VideoRequest request,
        IProgress<StepPreview>? progress = null, [EnumeratorCancellation] CancellationToken cancel = default)
    {
        VideoPlan plan = await PlanAsync(spec, request, cancel).ConfigureAwait(false);
        VideoPlan executionPlan = VideoRequestExecutionBinding.RequirePlannedState(plan);
        VideoRequest executionRequest = VideoRequestExecutionBinding.RequireUnchanged(executionPlan, request);
        LogPlanWarnings(executionPlan);
        executionPlan.ThrowIfInvalid();
        VideoRequest resolved = executionPlan.EffectiveSettings.Apply(executionRequest);
        VideoArtifactFileBinding.RequireUnchanged(executionPlan);
        IVideoRecipePipeline pipeline = _engine.GetOrConstructVideoRecipe(
            executionPlan.Model, resolved, executionPlan);
        if (!pipeline.SupportsStreaming)
        {
            throw new NotSupportedException(
                $"Video model family '{InferenceEngine.FamilyIdFor(executionPlan.Model)}' does not support streaming generation.");
        }
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

    /// <summary>Builds an exact H3 execution summary. Generic video families still perform family-specific
    /// geometry/schedule normalization inside their pipelines; reporting their pre-pipeline plan as "actual" would
    /// be false, so they remain null until each family exposes its own final-resolution contract.</summary>
    internal static VideoExecutionSummary? BuildExecutionSummary(VideoPlan plan)
    {
        if (!string.Equals(plan.Profile.FamilyId, "minimax-h3", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }
        VideoEffectiveSettings settings = plan.EffectiveSettings;
        return new VideoExecutionSummary
        {
            ProfileId = plan.Profile.Id,
            Task = plan.Profile.Task,
            Acceleration = plan.Profile.Acceleration,
            Attention = plan.Profile.Attention,
            Width = settings.Width,
            Height = settings.Height,
            Frames = settings.Frames,
            Fps = settings.Fps,
            Seed = settings.Seed,
            Steps = settings.Steps,
            CfgScale = settings.CfgScale,
            FlowShift = settings.FlowShift,
            AudioFlowShift = settings.AudioFlowShift,
            Sampler = settings.Sampler,
            Scheduler = settings.Scheduler,
            ExecutionPath = plan.Profile.Attention != VideoAttentionKind.Dense
                ? plan.Profile.Attention.ToString()
                : plan.Profile.Acceleration.ToString(),
            ComponentFormats = plan.ComponentFormats,
        };
    }

    private static void LogPlanWarnings(VideoPlan plan)
    {
        foreach (VideoPlanIssue issue in plan.Issues)
        {
            if (issue.Severity == VideoPlanIssueSeverity.Warning)
            {
                Logs.Warning($"[VideoPlan][{issue.Code}] {issue.Message}");
            }
        }
    }
}

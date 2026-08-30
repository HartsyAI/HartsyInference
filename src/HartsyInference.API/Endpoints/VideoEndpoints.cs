using HartsyInference.Core.Logging;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Microsoft.AspNetCore.OpenApi;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.OpenApi;

namespace HartsyInference.API.Endpoints;

/// <summary>Native video route. <see cref="IVideoService"/> only exposes a frame stream (<see cref="IAsyncEnumerable{T}"/> of <see cref="VideoFrame"/>, no <c>Task&lt;VideoResult&gt;</c>) — there's no way to build a non-streaming variant without buffering an entire video's frames in server memory first. An H.264 muxer DOES exist (<c>HartsyInference.Video.Encoding.FfmpegProcessEncoder</c>, ffmpeg subprocess — the CLI's restore path uses it), but it isn't reachable from this project (API references Core + Engine only), so this ships SSE-streamed raw frames; a muxed-download variant is future work. Runs through the long-running queue (video generation can take minutes), not the fast one every other route uses.</summary>
public static class VideoEndpoints
{
    /// <summary>Maps <c>/v1/native/video/stream</c>.</summary>
    public static void MapVideoEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/native/video/plan", async Task<IResult> (
            NativeVideoRequest req, IInferenceEngine engine,
            [FromKeyedServices(QueueKeys.LongRunning)] InferenceQueue queue, CancellationToken ct) =>
        {
            ModelSpec spec;
            VideoPlan plan;
            try
            {
                spec = ResolveSpec(req);
                plan = await PlanQueuedAsync(queue, engine, spec, req.Request, ct).ConfigureAwait(false);
            }
            catch (QueueFullException ex)
            {
                return GenerationErrors.Map(ex);
            }
            catch (Exception ex) when (IsPlanningInputFailure(ex))
            {
                return Results.Json(PlanningFailure(ex),
                    statusCode: StatusCodes.Status422UnprocessableEntity);
            }
            NativeVideoPlanResponse response = NativeVideoPlanResponse.Create(plan);
            return plan.IsValid
                ? Results.Ok(response)
                : Results.Json(new NativeVideoPlanProblem { Issues = response.Issues, Plan = response },
                    statusCode: StatusCodes.Status422UnprocessableEntity);
        })
            .Produces<NativeVideoPlanResponse>(StatusCodes.Status200OK)
            .Produces<NativeVideoPlanProblem>(StatusCodes.Status422UnprocessableEntity);

        app.MapPost("/v1/native/video/stream", async Task<IResult> (
            NativeVideoRequest req, IInferenceEngine engine,
            [FromKeyedServices(QueueKeys.LongRunning)] InferenceQueue queue, HttpContext ctx, CancellationToken ct) =>
        {
            ModelSpec spec;
            VideoPlan plan;
            // Resolve the identical plan generation consumes before SSE writes its 200 response headers. Invalid
            // profiles/combinations therefore remain ordinary typed HTTP 422 responses, never mid-stream errors.
            try
            {
                spec = ResolveSpec(req);
                // Preflight briefly owns the same long-running gate as generation because VSA hardware validation
                // may touch the singleton backend. The slot is released before SSE is opened and reacquired by
                // SseHelpers for the generation itself, avoiding both a race and a nested-queue deadlock.
                plan = await PlanQueuedAsync(queue, engine, spec, req.Request, ct).ConfigureAwait(false);
            }
            catch (QueueFullException ex)
            {
                return GenerationErrors.Map(ex);
            }
            catch (Exception ex) when (IsPlanningInputFailure(ex))
            {
                return Results.Json(PlanningFailure(ex),
                    statusCode: StatusCodes.Status422UnprocessableEntity);
            }
            if (!plan.IsValid)
            {
                NativeVideoPlanResponse response = NativeVideoPlanResponse.Create(plan);
                return Results.Json(new NativeVideoPlanProblem
                {
                    Issues = response.Issues,
                    Plan = response,
                },
                    statusCode: StatusCodes.Status422UnprocessableEntity);
            }
            await SseHelpers.RunAsync(ctx, queue, async (writer, jsonOptions) =>
            {
                // Runs inside the queue's held slot — call the service directly, not through a second EnqueueAsync.
                IProgress<StepPreview> progress = SseHelpers.InlineProgress<StepPreview>(p =>
                    writer.TryWrite(SseHelpers.Event("progress", StepPreviewPayload.Create(p), jsonOptions)));
                VideoGenerationResult result = await engine.Video.GenerateAsync(plan, req.Request, progress, ct);
                foreach (VideoFrame frame in result.Frames)
                {
                    ct.ThrowIfCancellationRequested();
                    writer.TryWrite(SseHelpers.Event("frame", new NativeVideoFrameEvent
                    {
                        Index = frame.Index,
                        Png = Convert.ToBase64String(PngEncoder.Encode(frame.Rgb, frame.Width, frame.Height)),
                    }, jsonOptions));
                }
                if (result.Audio is not null && !result.Audio.IsEmpty)
                {
                    writer.TryWrite(SseHelpers.Event("audio", new NativeVideoAudioEvent
                    {
                        SampleRate = result.Audio.SampleRate,
                        Channels = result.Audio.ChannelCount,
                        Wav = Convert.ToBase64String(AudioClipCodec.EncodeWav(result.Audio)),
                    }, jsonOptions));
                }
                writer.TryWrite(SseHelpers.Event("complete", new NativeVideoCompleteEvent
                {
                    Frames = result.Frames.Count,
                    SavedPath = Persist(req, result),
                    Execution = result.Execution,
                }, jsonOptions));
            }, ct);
            return Results.Empty;
        })
            .Produces(StatusCodes.Status200OK, contentType: "text/event-stream")
            .Produces<NativeVideoPlanProblem>(StatusCodes.Status422UnprocessableEntity)
            .AddOpenApiOperationTransformer(DocumentStreamResponseAsync);
    }

    /// <summary>Documents the named SSE events using the exact JSON payload types written by the stream.</summary>
    private static async Task DocumentStreamResponseAsync(OpenApiOperation operation,
        OpenApiOperationTransformerContext context, CancellationToken cancel)
    {
        if (operation.Responses is null
            || !operation.Responses.TryGetValue(StatusCodes.Status200OK.ToString(), out IOpenApiResponse? response)
            || response is not OpenApiResponse documentedResponse || context.Document is not OpenApiDocument document)
        {
            return;
        }

        OpenApiSchema progress = await context.GetOrCreateSchemaAsync(typeof(StepPreviewPayload), null, cancel)
            .ConfigureAwait(false);
        OpenApiSchema frame = await context.GetOrCreateSchemaAsync(typeof(NativeVideoFrameEvent), null, cancel)
            .ConfigureAwait(false);
        OpenApiSchema audio = await context.GetOrCreateSchemaAsync(typeof(NativeVideoAudioEvent), null, cancel)
            .ConfigureAwait(false);
        OpenApiSchema complete = await context.GetOrCreateSchemaAsync(typeof(NativeVideoCompleteEvent), null, cancel)
            .ConfigureAwait(false);
        OpenApiSchema error = await context.GetOrCreateSchemaAsync(typeof(NativeSseErrorEvent), null, cancel)
            .ConfigureAwait(false);
        OpenApiSchema execution = await context.GetOrCreateSchemaAsync(typeof(VideoExecutionSummary), null, cancel)
            .ConfigureAwait(false);
        complete.Properties ??= new Dictionary<string, IOpenApiSchema>(StringComparer.Ordinal);
        complete.Properties["execution"] = new OpenApiSchema
        {
            OneOf =
            [
                new OpenApiSchemaReference(nameof(VideoExecutionSummary), document),
                new OpenApiSchema { Type = JsonSchemaType.Null },
            ],
        };

        document.AddComponent(nameof(StepPreviewPayload), progress);
        document.AddComponent(nameof(NativeVideoFrameEvent), frame);
        document.AddComponent(nameof(NativeVideoAudioEvent), audio);
        document.AddComponent(nameof(NativeVideoCompleteEvent), complete);
        document.AddComponent(nameof(NativeSseErrorEvent), error);
        document.AddComponent(nameof(VideoExecutionSummary), execution);

        documentedResponse.Content ??= new Dictionary<string, OpenApiMediaType>(StringComparer.OrdinalIgnoreCase);
        documentedResponse.Content["text/event-stream"] = new OpenApiMediaType
        {
            Schema = new OpenApiSchema
            {
                Description = "A sequence of Server-Sent Events. Each message uses `event: <name>` followed by a "
                    + "JSON `data:` payload. The oneOf title is the event name; progress may repeat, frame repeats "
                    + "once per output frame, audio is optional, complete is the successful terminal event, and "
                    + "error is terminal when generation fails after streaming begins.",
                OneOf =
                [
                    EventPayload("progress", "Denoising progress and optional preview image data.",
                        new OpenApiSchemaReference(nameof(StepPreviewPayload), document)),
                    EventPayload("frame", "One generated frame encoded as a base64 PNG.",
                        new OpenApiSchemaReference(nameof(NativeVideoFrameEvent), document)),
                    EventPayload("audio", "The optional generated soundtrack encoded as base64 WAV.",
                        new OpenApiSchemaReference(nameof(NativeVideoAudioEvent), document)),
                    EventPayload("complete", "Terminal generation details and the actual execution summary.",
                        new OpenApiSchemaReference(nameof(NativeVideoCompleteEvent), document)),
                    EventPayload("error", "Terminal generation failure after the SSE response has begun.",
                        new OpenApiSchemaReference(nameof(NativeSseErrorEvent), document)),
                ],
            },
        };
        operation.Responses[StatusCodes.Status200OK.ToString()] = documentedResponse;
    }

    /// <summary>Labels one JSON payload schema with the SSE event name that carries it.</summary>
    private static OpenApiSchema EventPayload(string eventName, string description, IOpenApiSchema payload) => new()
    {
        Title = eventName,
        Description = description,
        AllOf = [payload],
    };

    /// <summary>Runs planning under the video gate, releasing it before the caller starts any SSE response.</summary>
    private static Task<VideoPlan> PlanQueuedAsync(InferenceQueue queue, IInferenceEngine engine, ModelSpec spec,
        VideoRequest request, CancellationToken cancel) =>
        queue.EnqueueAsync(() => engine.VideoPlanning.PlanAsync(spec, request, cancel), cancel);

    /// <summary>True for request/model failures that occur before a typed <see cref="VideoPlan"/> exists.</summary>
    private static bool IsPlanningInputFailure(Exception exception) => exception is
        FileNotFoundException or DirectoryNotFoundException or ArgumentException or InvalidDataException;

    /// <summary>Creates the same typed 422 envelope used by ordinary plan diagnostics without inventing a plan.</summary>
    private static NativeVideoPlanProblem PlanningFailure(Exception exception)
    {
        VideoPlanIssue issue = NativeVideoPlanResponse.SanitizeIssue(new VideoPlanIssue
        {
            Code = "video.plan.model_unresolvable",
            Severity = VideoPlanIssueSeverity.Error,
            Message = exception.Message,
            Field = nameof(NativeVideoRequest.Model),
        });
        return new NativeVideoPlanProblem { Issues = [issue] };
    }

    /// <summary>Resolves the model location and carries the additive checkpoint-profile hint into planning.</summary>
    private static ModelSpec ResolveSpec(NativeVideoRequest request)
    {
        ModelSpec spec = ModelResolver.Resolve(request.Model, request.ModelPath, Modality.Video);
        return string.IsNullOrWhiteSpace(request.ModelProfile) ? spec : spec with { ProfileId = request.ModelProfile };
    }

    /// <summary>Writes the frame sequence (and the soundtrack beside it) into a numbered directory under the output root, the same layout the CLI produces. Null when the request opted out or the write failed.</summary>
    private static string? Persist(NativeVideoRequest req, VideoGenerationResult result)
    {
        if (req.Save == false || result.Frames.Count == 0)
            return null;
        try
        {
            VideoFrame first = result.Frames[0];
            byte[][] rgb = new byte[result.Frames.Count][];
            for (int i = 0; i < result.Frames.Count; i++)
                rgb[i] = result.Frames[i].Rgb;
            return VideoOutputWriter.Write(rgb, first.Width, first.Height, OutputWriter.ResolveDir(req.OutputDir),
                req.Request.Prompt, result.Audio, result.Fps ?? 24).Directory;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[api] generated video could not be saved: {ex.Message}");
            return null;
        }
    }
}

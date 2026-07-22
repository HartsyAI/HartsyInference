using System.Text.Json;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>OpenAI-shaped <c>/v1/chat/completions</c> and <c>/v1/images/generations</c> — thin DTO mappers that
/// call the SAME native handlers <see cref="TextEndpoints"/>/<see cref="ImageEndpoints"/> use, not a parallel
/// implementation. Deliberately narrow: composition-heavy requests (LoRA/ControlNet/regional prompting, tool
/// calling, JSON-schema response format) don't fit OpenAI's schema and belong on the native routes instead.</summary>
public static class CompatEndpoints
{
    /// <summary>Maps the two OpenAI-compat routes.</summary>
    public static void MapCompatEndpoints(this WebApplication app)
    {
        app.MapPost("/v1/chat/completions", async (ChatCompletionRequest req, IInferenceEngine engine, InferenceQueue queue, HttpContext ctx, CancellationToken ct) =>
        {
            if (req.Messages.Count == 0)
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, "Field 'messages' must be non-empty.", "invalid_request_error");
            if (req.Model is null)
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, "Field 'model' is required.", "invalid_request_error");
            if (req.ResponseFormat is { Type: not "text" })
            {
                // TextRequest (the native contract) has no JSON-mode/grammar-constraint field yet — rejecting
                // rather than silently generating unconstrained text for a client that asked for JSON.
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest,
                    $"response_format.type '{req.ResponseFormat.Type}' is not supported yet — only the default 'text' is.", "invalid_request_error");
            }

            string model = req.Model; // captured into a definitely-non-null local for the streaming lambda below
            ModelSpec spec = ModelResolver.Resolve(model, modelPathArg: null, Modality.Text);
            TextRequest textRequest = ToTextRequest(req);
            string id = $"chatcmpl-{Guid.NewGuid():N}";
            long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            if (!req.Stream)
            {
                try
                {
                    TextResult result = await TextEndpoints.GenerateAsync(engine, queue, spec, textRequest, ct);
                    return Results.Ok(new ChatCompletionResponse
                    {
                        Id = id,
                        Created = created,
                        Model = model,
                        Choices = [new ChatCompletionChoice
                        {
                            Message = new ChatMessageDto { Role = "assistant", Content = result.Text },
                            FinishReason = ToFinishReason(result.Stop),
                        }],
                        Usage = new ChatUsage { PromptTokens = result.PromptTokens, CompletionTokens = result.CompletionTokens },
                    });
                }
                catch (Exception ex)
                {
                    return GenerationErrors.Map(ex);
                }
            }

            await SseHelpers.RunAsync(ctx, queue, async (writer, jsonOptions) =>
            {
                writer.TryWrite(RawDataFrame(new ChatCompletionChunk
                {
                    Id = id, Created = created, Model = model,
                    Choices = [new ChatCompletionChunkChoice { Delta = new ChatCompletionDelta { Role = "assistant" } }],
                }, jsonOptions));

                string finishReason = "stop";
                await foreach (TextChunk chunk in engine.Text.StreamAsync(spec, textRequest, ct))
                {
                    if (chunk.Kind == TextChunkKind.Chunk && chunk.Text is not null)
                    {
                        writer.TryWrite(RawDataFrame(new ChatCompletionChunk
                        {
                            Id = id, Created = created, Model = model,
                            Choices = [new ChatCompletionChunkChoice { Delta = new ChatCompletionDelta { Content = chunk.Text } }],
                        }, jsonOptions));
                    }
                    else if (chunk.Kind == TextChunkKind.StopReason && chunk.Stop is { } stop)
                    {
                        finishReason = ToFinishReason(stop);
                    }
                }

                writer.TryWrite(RawDataFrame(new ChatCompletionChunk
                {
                    Id = id, Created = created, Model = model,
                    Choices = [new ChatCompletionChunkChoice { Delta = new ChatCompletionDelta(), FinishReason = finishReason }],
                }, jsonOptions));
                writer.TryWrite("data: [DONE]\n\n");
            }, ct);
            return Results.Empty;
        });

        app.MapPost("/v1/images/generations", async (ImageGenerationRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            if (req.Model is null)
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, "Field 'model' is required.", "invalid_request_error");

            ModelSpec spec = ModelResolver.Resolve(req.Model, modelPathArg: null, Modality.Image);
            ImageRequest imageRequest = ToImageRequest(req);
            try
            {
                List<ImageData> images = [];
                int n = Math.Max(1, req.N);
                for (int i = 0; i < n; i++)
                {
                    ImageResult result = await ImageEndpoints.GenerateAsync(engine, queue, spec, imageRequest, progress: null, ct);
                    images.Add(new ImageData { B64Json = Convert.ToBase64String(PngEncoder.Encode(result.Rgb, result.Width, result.Height)) });
                }
                return Results.Ok(new ImageGenerationResponse { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = images });
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/images/generations/stream", async (ImageGenerationRequest req, IInferenceEngine engine, InferenceQueue queue, HttpContext ctx, CancellationToken ct) =>
        {
            if (req.Model is null)
            {
                await HartsyInferenceServiceExtensions.WriteErrorAsync(ctx, StatusCodes.Status400BadRequest, "Field 'model' is required.", "invalid_request_error");
                return;
            }

            ModelSpec spec = ModelResolver.Resolve(req.Model, modelPathArg: null, Modality.Image);
            ImageRequest imageRequest = ToImageRequest(req);
            await SseHelpers.RunAsync(ctx, queue, async (writer, jsonOptions) =>
            {
                Progress<StepPreview> progress = new Progress<StepPreview>(p =>
                    writer.TryWrite(SseHelpers.Event("progress", new { step = p.Step, total = p.TotalSteps }, jsonOptions)));
                // Runs inside the queue's held slot — call the service directly, not the GenerateAsync helper.
                ImageResult result = await engine.Images.GenerateAsync(spec, imageRequest, progress, ct);
                string png = Convert.ToBase64String(PngEncoder.Encode(result.Rgb, result.Width, result.Height));
                writer.TryWrite(SseHelpers.Event("complete", new { b64_json = png }, jsonOptions));
            }, ct);
        });
    }

    // Adapter: OpenAI image DTO → the engine's native ImageRequest (carries only the fields OpenAI's schema
    // expresses; LoRA/ControlNet/etc. are native-route-only, per this file's class doc).
    private static ImageRequest ToImageRequest(ImageGenerationRequest req)
    {
        (int? width, int? height) = ParseSize(req.Size);
        return new ImageRequest
        {
            Prompt = req.Prompt,
            NegativePrompt = req.NegativePrompt,
            Width = width,
            Height = height,
            Steps = req.Steps,
            CfgScale = req.CfgScale,
            Seed = req.Seed,
            ClipSkip = req.ClipSkip,
        };
    }

    // A missing or unparseable "size" leaves both null so the model family's native resolution applies.
    private static (int? width, int? height) ParseSize(string? size)
    {
        if (string.IsNullOrWhiteSpace(size)) return (null, null);
        string[] parts = size.Split('x', 'X');
        if (parts.Length == 2 && int.TryParse(parts[0], out int w) && int.TryParse(parts[1], out int h))
            return (w, h);
        return (null, null);
    }

    private static TextRequest ToTextRequest(ChatCompletionRequest req) => new TextRequest
    {
        Messages = [.. req.Messages.Select(m => new TextMessage { Role = ParseRole(m.Role), Content = m.Content })],
        Temperature = req.Temperature ?? 0.7,
        TopP = req.TopP ?? 0.95,
        TopK = req.TopK,
        MinP = req.MinP,
        RepetitionPenalty = req.RepetitionPenalty,
        MaxTokens = req.MaxTokens ?? 4096,
        Seed = req.Seed.HasValue ? (long)req.Seed.Value : -1,
        Greedy = req.Temperature is 0f,
    };

    private static TextRole ParseRole(string role) => role.ToLowerInvariant() switch
    {
        "system" => TextRole.System,
        "assistant" => TextRole.Assistant,
        "tool" => TextRole.Tool,
        _ => TextRole.User,
    };

    // OpenAI's finish_reason vocabulary has no slot for Cancelled/Error — both collapse to "stop" (best-effort
    // compat) rather than inventing a non-standard value a client SDK won't recognize.
    private static string ToFinishReason(StopReason stop) => stop switch
    {
        StopReason.Length => "length",
        StopReason.ToolCall => "tool_calls",
        _ => "stop",
    };

    // OpenAI's real wire format has no "event:" line, just "data: {...}\n\n" — unlike SseHelpers.Event's named
    // frames (a HartsyInference-native convenience for the /v1/native/* routes), so this stays a plain data frame.
    private static string RawDataFrame(object data, JsonSerializerOptions options) =>
        $"data: {JsonSerializer.Serialize(data, options)}\n\n";
}

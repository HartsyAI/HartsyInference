using System.Text.Json;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>OpenAI-shaped <c>/v1/chat/completions</c> and <c>/v1/images/generations</c> — thin DTO mappers that call the SAME native handlers <see cref="TextEndpoints"/>/<see cref="ImageEndpoints"/> use, not a parallel implementation. Deliberately narrow: composition-heavy requests (LoRA/ControlNet/regional prompting, tool calling, JSON-schema response format) don't fit OpenAI's schema and belong on the native routes instead.</summary>
public static class CompatEndpoints
{
    /// <summary>Server-process start time, reused as every <see cref="ModelEntry.Created"/> value — see that field's doc comment for why this isn't a real per-model timestamp.</summary>
    private static readonly long s_processStartUnixSeconds = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    /// <summary>Maps the OpenAI-compat routes.</summary>
    public static void MapCompatEndpoints(this WebApplication app)
    {
        app.MapGet("/v1/models", () => Results.Ok(new ModelListResponse
        {
            Data = [.. ModelCatalog.All.Select(e => new ModelEntry { Id = e.Id, Created = s_processStartUnixSeconds })],
        }));

        app.MapGet("/v1/models/{model}", (string model) =>
        {
            CatalogEntry? entry = ModelCatalog.Find(model);
            return entry is null
                ? HartsyInferenceServiceExtensions.Problem(StatusCodes.Status404NotFound, $"'{model}' is not a known model.", "invalid_request_error")
                : Results.Ok(new ModelEntry { Id = entry.Id, Created = s_processStartUnixSeconds });
        });

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
                            // OpenAI sets content null (not "") when a tool call is present.
                            Message = new ChatMessageDto
                            {
                                Role = "assistant",
                                Content = result.ToolCall is null ? result.Text : null,
                                ToolCalls = result.ToolCall is { } call ? [ToToolCallDto(call)] : null,
                            },
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
                    else if (chunk.Kind == TextChunkKind.NativeToolCall && chunk.ToolCall is { } call)
                    {
                        // Emitted as ONE complete delta (full name + arguments already assembled) rather than
                        // incrementally character-by-character like OpenAI's own servers sometimes do -- the
                        // native stream hands the whole call over in one chunk, and a single complete delta is
                        // still spec-valid (most client SDKs just concatenate whatever arrives).
                        writer.TryWrite(RawDataFrame(new ChatCompletionChunk
                        {
                            Id = id, Created = created, Model = model,
                            Choices = [new ChatCompletionChunkChoice { Delta = new ChatCompletionDelta { ToolCalls = [ToToolCallDto(call, index: 0)] } }],
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

        app.MapPost("/v1/embeddings", async (EmbeddingsRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            if (string.IsNullOrEmpty(req.Model))
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, "Field 'model' is required.", "invalid_request_error");
            if (!string.Equals(req.EncodingFormat, "float", StringComparison.OrdinalIgnoreCase))
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest,
                    $"encoding_format '{req.EncodingFormat}' is not supported yet — only 'float' is.", "invalid_request_error");
            }

            List<string> inputs;
            if (req.Input.ValueKind == JsonValueKind.String)
            {
                inputs = [req.Input.GetString() ?? ""];
            }
            else if (req.Input.ValueKind == JsonValueKind.Array)
            {
                inputs = [.. req.Input.EnumerateArray().Select(e => e.GetString() ?? "")];
            }
            else
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, "Field 'input' must be a string or an array of strings.", "invalid_request_error");
            }
            if (inputs.Count == 0)
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, "Field 'input' must be non-empty.", "invalid_request_error");

            ModelSpec spec = ModelResolver.Resolve(req.Model, modelPathArg: null, Modality.Embedding);
            EmbeddingRequest embeddingRequest = new EmbeddingRequest { Input = inputs };
            try
            {
                EmbeddingResult result = await queue.EnqueueAsync(() => engine.Embeddings.GenerateAsync(spec, embeddingRequest, ct), ct);
                if (req.Dimensions is { } dims && dims != result.Dimensions)
                {
                    return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest,
                        $"dimensions {dims} was requested but the model produces {result.Dimensions}-wide vectors — truncation isn't supported.", "invalid_request_error");
                }
                return Results.Ok(new EmbeddingsResponse
                {
                    Model = req.Model,
                    Data = [.. result.Vectors.Select((v, i) => new EmbeddingDataDto { Embedding = v, Index = i })],
                    Usage = new EmbeddingsUsage { PromptTokens = result.TotalTokens, TotalTokens = result.TotalTokens },
                });
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        app.MapPost("/v1/audio/speech", async (SpeechGenerationRequest req, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            if (string.IsNullOrEmpty(req.Model))
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, "Field 'model' is required.", "invalid_request_error");
            if (string.IsNullOrEmpty(req.Input))
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, "Field 'input' is required.", "invalid_request_error");
            if (!string.Equals(req.ResponseFormat, "wav", StringComparison.OrdinalIgnoreCase))
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest,
                    $"response_format '{req.ResponseFormat}' is not supported yet — only 'wav' is (the engine always produces a pre-encoded WAV container).", "invalid_request_error");
            }

            ModelSpec spec = ModelResolver.Resolve(req.Model, modelPathArg: null, Modality.Speech);
            SpeechRequest speechRequest = new SpeechRequest { Text = req.Input, Voice = req.Voice, Speed = req.Speed };
            try
            {
                AudioResult result = await queue.EnqueueAsync(() => engine.Speech.SynthesizeAsync(spec, speechRequest, ct), ct);
                // Raw binary, not a JSON envelope -- matches OpenAI's real behavior for this endpoint.
                return Results.Bytes(result.Data, "audio/wav");
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
        });

        // First multipart/form-data route in this codebase (every other body is base64-in-JSON). Reads the form
        // via HttpRequest.ReadFormAsync() directly rather than an [FromForm]-bound DTO parameter -- the latter
        // requires either an antiforgery service registration or an explicit DisableAntiforgery() call on
        // every such route since .NET 8 (a browser-CSRF protection this machine API doesn't need); reading the
        // form manually bypasses that binding pipeline entirely, so neither is required.
        app.MapPost("/v1/audio/transcriptions", async (HttpRequest httpReq, IInferenceEngine engine, InferenceQueue queue, CancellationToken ct) =>
        {
            if (!httpReq.HasFormContentType)
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, "Expected multipart/form-data with a 'file' field.", "invalid_request_error");

            IFormCollection form = await httpReq.ReadFormAsync(ct);
            IFormFile? file = form.Files["file"];
            if (file is null || file.Length == 0)
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, "Field 'file' is required.", "invalid_request_error");

            string? model = form["model"];
            if (string.IsNullOrEmpty(model))
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest, "Field 'model' is required.", "invalid_request_error");

            string responseFormat = form["response_format"].FirstOrDefault() ?? "json";
            if (responseFormat is not ("json" or "text"))
            {
                return HartsyInferenceServiceExtensions.Problem(StatusCodes.Status400BadRequest,
                    $"response_format '{responseFormat}' is not supported yet — only 'json' and 'text' are (srt/vtt/verbose_json need per-segment timestamp formatting this route doesn't build yet).", "invalid_request_error");
            }
            string language = form["language"].FirstOrDefault() ?? "en";

            byte[] audioBytes;
            using (MemoryStream ms = new MemoryStream())
            {
                await file.CopyToAsync(ms, ct);
                audioBytes = ms.ToArray();
            }

            ModelSpec spec = ModelResolver.Resolve(model, modelPathArg: null, Modality.Transcribe);
            // The Engine only decodes RIFF/WAVE PCM (no ffmpeg dependency) -- an mp3/m4a/webm upload fails
            // decode inside RunAsync and surfaces as a HartsyInferenceException, which GenerationErrors already
            // maps to 400 with a clear "WAV only" message, so no bespoke handling is needed here for that case.
            AudioRequest audioRequest = new AudioRequest { Audio = new AudioClip { Data = audioBytes, Format = "wav" }, Language = language };
            try
            {
                TranscriptResult result = await queue.EnqueueAsync(() => engine.Transcribe.RunAsync(spec, audioRequest, ct), ct);
                return responseFormat == "text" ? Results.Text(result.Text) : Results.Ok(new { text = result.Text });
            }
            catch (Exception ex)
            {
                return GenerationErrors.Map(ex);
            }
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
                    byte[] png = PngEncoder.Encode(result.Rgb, result.Width, result.Height);
                    ArtifactPersistence.Save(png, imageRequest.Prompt, "png");
                    images.Add(new ImageData { B64Json = Convert.ToBase64String(png) });
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
                    writer.TryWrite(SseHelpers.Event("progress", StepPreviewPayload.Create(p), jsonOptions)));
                // Runs inside the queue's held slot — call the service directly, not the GenerateAsync helper.
                ImageResult result = await engine.Images.GenerateAsync(spec, imageRequest, progress, ct);
                byte[] pngBytes = PngEncoder.Encode(result.Rgb, result.Width, result.Height);
                ArtifactPersistence.Save(pngBytes, imageRequest.Prompt, "png");
                writer.TryWrite(SseHelpers.Event("complete", new { b64_json = Convert.ToBase64String(pngBytes) }, jsonOptions));
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

    private static TextRequest ToTextRequest(ChatCompletionRequest req)
    {
        // "none" suppresses tool-calling for this turn entirely, matching OpenAI's own semantics -- everything
        // else (unset/"auto"/"required"/the named-function object form) passes Tools through when present.
        bool toolsSuppressed = req.ToolChoice is { ValueKind: JsonValueKind.String } tc && tc.GetString() == "none";
        string? forceToolId = null;
        if (req.ToolChoice is { ValueKind: JsonValueKind.Object } choice
            && choice.TryGetProperty("function", out JsonElement fn)
            && fn.TryGetProperty("name", out JsonElement nameEl))
        {
            forceToolId = nameEl.GetString();
        }

        return new TextRequest
        {
            // Native TextMessage has no tool_call_id correlation field -- a "tool" role message's content is
            // passed through as-is. Fine for the common single-pending-call case; a multi-tool-call turn can't
            // disambiguate which call a result answers, a real gap versus full OpenAI compat.
            Messages = [.. req.Messages.Select(m => new TextMessage { Role = ParseRole(m.Role), Content = m.Content ?? "" })],
            Temperature = req.Temperature ?? 0.7,
            TopP = req.TopP ?? 0.95,
            TopK = req.TopK,
            MinP = req.MinP,
            RepetitionPenalty = req.RepetitionPenalty,
            MaxTokens = req.MaxTokens ?? 4096,
            Seed = req.Seed.HasValue ? (long)req.Seed.Value : -1,
            Greedy = req.Temperature is 0f,
            Tools = (!toolsSuppressed && req.Tools is { Count: > 0 })
                ? [.. req.Tools.Select(t => new ToolDefinition
                    {
                        Name = t.Function.Name,
                        Description = t.Function.Description,
                        JsonSchema = t.Function.Parameters.ValueKind == JsonValueKind.Undefined ? "{}"
                            : t.Function.Parameters.GetRawText(),
                    })]
                : null,
            ForceToolId = toolsSuppressed ? null : forceToolId,
        };
    }

    private static TextRole ParseRole(string role) => role.ToLowerInvariant() switch
    {
        "system" => TextRole.System,
        "assistant" => TextRole.Assistant,
        "tool" => TextRole.Tool,
        _ => TextRole.User,
    };

    private static ChatToolCallDto ToToolCallDto(NativeToolCall call, int? index = null) => new ChatToolCallDto
    {
        Id = string.IsNullOrEmpty(call.Id) ? $"call_{Guid.NewGuid():N}" : call.Id,
        Index = index,
        Function = new ChatToolCallFunctionDto { Name = call.Name, Arguments = call.Arguments },
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

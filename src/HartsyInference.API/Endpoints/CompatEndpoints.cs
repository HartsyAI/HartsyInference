using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Quant;
// A bare ImageData means the API's own OpenAI DTO (the enclosing namespace wins over any using), so the native type gets another name.
using NativeImageData = HartsyInference.Engine.Requests.ImageData;

namespace HartsyInference.API.Endpoints;

/// <summary>OpenAI-shaped <c>/v1/chat/completions</c> and <c>/v1/images/generations</c> — thin DTO mappers that call the SAME native handlers <see cref="TextEndpoints"/>/<see cref="ImageEndpoints"/> use, not a parallel implementation. Deliberately narrow: composition-heavy requests (LoRA/ControlNet/regional prompting, JSON-schema response format) don't fit OpenAI's schema and belong on the native routes instead; tool definitions, <c>tool_calls</c> and <c>tool_call_id</c> map both ways.</summary>
public static class CompatEndpoints
{
    /// <summary>Largest decoded image one <c>image_url</c> part may carry. The server sets no request-body limit of its own for chat, and a host can raise Kestrel's.</summary>
    internal const int MaxImageBytes = 32 * 1024 * 1024;

    /// <summary>Largest total decoded image one chat request may carry across all its parts: four images' worth. A small PNG can declare a large canvas, so without a
    /// total a request could repeat one until it decoded to gigabytes.</summary>
    internal const long MaxRequestImageBytes = 4L * MaxImageBytes;

    /// <summary>Server-process start time, reused as every <see cref="ModelEntry.Created"/> value — see that field's doc comment for why this isn't a real per-model timestamp.</summary>
    private static readonly long s_processStartUnixSeconds = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    /// <summary>One <c>/v1/models</c> entry. Text models carry their capabilities; the other modalities do not.</summary>
    internal static ModelEntry ModelEntryFor(CatalogEntry entry) => new()
    {
        Id = entry.Id,
        Created = s_processStartUnixSeconds,
        Capabilities = entry.Modality == Modality.Text ? TextCapabilities(entry.Variants.Select(v => v.Flavor)) : null,
    };

    /// <summary>A text model's capabilities from its variants' flavors (a null flavor is GGUF). DSpark speculation is off for every format today: GGUF and MLX have no usable
    /// draft head, and the safetensors path has one that serving does not run yet. The reason names the case that applies, in the words <c>/admin/packages</c> uses.</summary>
    internal static CapabilitiesDto TextCapabilities(IEnumerable<QuantFlavor?> flavors)
    {
        List<QuantFlavor?> list = [.. flavors];
        string reason = list.Count == 0 ? TextPackageSpeculation.NoVariantReason
            : list.Any(f => f is not null && f != QuantFlavor.Mlx) ? TextPackageSpeculation.NotWiredReason
            : list.Any(f => f == QuantFlavor.Mlx) ? TextPackageSpeculation.MlxReason
            : TextPackageSpeculation.GgufReason;
        return new CapabilitiesDto { Speculation = false, SpeculationReason = reason };
    }

    /// <summary>Maps the OpenAI-compat routes.</summary>
    public static void MapCompatEndpoints(this WebApplication app)
    {
        app.MapGet("/v1/models", () => Results.Ok(new ModelListResponse
        {
            Data = [.. ModelCatalog.All.Select(ModelEntryFor)],
        }));

        app.MapGet("/v1/models/{model}", (string model) =>
        {
            CatalogEntry? entry = ModelCatalog.Find(model);
            return entry is null
                ? HartsyInferenceServiceExtensions.Problem(StatusCodes.Status404NotFound, $"'{model}' is not a known model.", "invalid_request_error")
                : Results.Ok(ModelEntryFor(entry));
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
            TextRequest textRequest;
            try
            {
                textRequest = ToTextRequest(req);
            }
            catch (Exception ex) when (ex is HartsyInferenceException or ArgumentException)
            {
                return GenerationErrors.Map(ex);
            }
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

            ChatStreamTranslator translator = new(id, created, model, includeUsage: req.StreamOptions?.IncludeUsage == true);
            await SseHelpers.RunTextAsync(ctx, queue, async (writer, jsonOptions) =>
            {
                writer.TryWrite(RawDataFrame(translator.Start(), jsonOptions));
                await foreach (TextChunk chunk in engine.Text.StreamAsync(spec, textRequest, ct))
                {
                    if (chunk.Kind == TextChunkKind.Status && chunk.Status is { } status)
                    {
                        // Queue position and prefill progress, as a named frame that OpenAI clients ignore.
                        writer.TryWrite(SseHelpers.Event("hartsy.status", HartsyStatusDto.From(status), jsonOptions));
                        continue;
                    }
                    foreach (ChatCompletionChunk frame in translator.Handle(chunk))
                        writer.TryWrite(RawDataFrame(frame, jsonOptions));
                }
                foreach (ChatCompletionChunk frame in translator.End())
                    writer.TryWrite(RawDataFrame(frame, jsonOptions));
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

    /// <summary>OpenAI chat request → native <see cref="TextRequest"/>: messages with their <c>tool_calls</c>/<c>tool_call_id</c>/<c>name</c>, sampling knobs, tools and tool choice.</summary>
    internal static TextRequest ToTextRequest(ChatCompletionRequest req)
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

        ImageBudget imageBudget = new();
        return new TextRequest
        {
            Messages = [.. req.Messages.Select(m => ToTextMessage(m, imageBudget))],
            Temperature = req.Temperature ?? 0.7,
            TopP = req.TopP ?? 0.95,
            TopK = req.TopK,
            MinP = req.MinP,
            RepetitionPenalty = req.RepetitionPenalty,
            MaxTokens = req.MaxTokens ?? 4096,
            Seed = req.Seed.HasValue ? (long)req.Seed.Value : -1,
            ReasoningEffort = ParseReasoningEffort(req.ReasoningEffort),
            EnableThinking = ParseThinking(req.Thinking),
            User = req.User,
            Priority = ParsePriority(req.Priority),
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

    /// <summary>One OpenAI message → native <see cref="TextMessage"/>, carrying an assistant turn's <c>tool_calls</c> and a tool turn's <c>tool_call_id</c>/<c>name</c>.</summary>
    /// <remarks>Array content keeps the order of its text parts, joined into one string, and of its images, in a separate list. A native message holds one text and its
    /// images apart, so where an image sat between text parts is lost.</remarks>
    internal static TextMessage ToTextMessage(ChatMessageDto m, ImageBudget? budget = null)
    {
        budget ??= new ImageBudget();
        List<NativeImageData> images = [];
        StringBuilder text = new();
        foreach (ChatContentPart part in m.Content?.Parts ?? Array.Empty<ChatContentPart>())
        {
            if (part.Type == "text")
                text.Append(part.Text ?? throw new HartsyInferenceException("A 'text' content part needs a string 'text'."));
            else if (part.Type == "image_url")
            {
                NativeImageData image = DecodeImageUrl(part.ImageUrl);
                budget.Take(image);
                images.Add(image);
            }
            else
                throw new HartsyInferenceException($"Content part type '{part.Type}' is not supported; use 'text' or 'image_url'.");
        }
        return new TextMessage
        {
            Role = ParseRole(m.Role),
            Content = m.Content?.Text ?? text.ToString(),
            Images = images.Count > 0 ? images : null,
            ToolCalls = m.ToolCalls is { Count: > 0 } ? [.. m.ToolCalls.Select(ToNativeToolCall)] : null,
            ToolCallId = m.ToolCallId,
            Name = m.Name,
            ReasoningContent = m.ReasoningContent,
        };
    }

    /// <summary>Decodes a <c>data:</c> image URI's base64 payload. A remote URL is refused, since the server does not fetch images, and so is a payload that decodes
    /// to more than <see cref="MaxImageBytes"/>, or is a PNG whose header declares more decoded RGB than that.</summary>
    internal static NativeImageData DecodeImageUrl(string? url)
    {
        const string prefix = "data:";
        if (url is null || !url.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            throw new HartsyInferenceException("Image URLs must be data: URIs; the server does not fetch remote images.");
        int comma = url.IndexOf(',');
        if (comma < 0 || !url[..comma].EndsWith(";base64", StringComparison.OrdinalIgnoreCase))
            throw new HartsyInferenceException("Image data URIs must be base64-encoded: data:<type>;base64,<data>.");
        // Every 4 base64 characters carry 3 bytes, so the size is known before anything is decoded. A line break in the payload makes this an over-estimate,
        // so only a payload right at the limit is refused early.
        long decodedBytes = (long)(url.Length - comma - 1) / 4 * 3;
        if (decodedBytes > MaxImageBytes)
            throw new HartsyInferenceException($"Image data decodes to about {decodedBytes} bytes, over the {MaxImageBytes}-byte (32 MiB) limit.");
        byte[] png;
        try
        {
            png = Convert.FromBase64String(url[(comma + 1)..].Trim());
        }
        catch (FormatException ex)
        {
            throw new HartsyInferenceException("Image data is not valid base64.", ex);
        }
        RefuseOversizedPng(png);
        try
        {
            return ImageDataCodec.Decode(png);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or InvalidDataException)
        {
            // The codec's refusals (not a PNG, a corrupt PNG) are the caller's to fix, so they are a 400.
            throw new HartsyInferenceException($"Image data could not be decoded: {ex.Message}", ex);
        }
    }

    /// <summary>Refuses a PNG whose header declares more than <see cref="MaxImageBytes"/> of decoded RGB (3 bytes a pixel), before the decoder allocates it: a small
    /// compressed file can declare a canvas far larger than its size. Bytes without a PNG header are left to the codec, which names the format it reads.</summary>
    internal static void RefuseOversizedPng(ReadOnlySpan<byte> png)
    {
        if (png.Length < 24 || !png[..8].SequenceEqual(PngSignature) || !png.Slice(12, 4).SequenceEqual("IHDR"u8))
            return;
        ulong width = BinaryPrimitives.ReadUInt32BigEndian(png[16..]);
        ulong height = BinaryPrimitives.ReadUInt32BigEndian(png[20..]);
        if (width * height > (ulong)MaxImageBytes / 3)
            throw new HartsyInferenceException($"Image is {width}x{height} pixels, which decodes to more than the {MaxImageBytes}-byte (32 MiB) limit. Send a smaller image.");
    }

    private static ReadOnlySpan<byte> PngSignature => [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    /// <summary>The decoded image bytes a chat request has taken so far. Each image is counted as it decodes, so a request decodes to at most the budget plus one image.</summary>
    internal sealed class ImageBudget
    {
        private long _taken;

        /// <summary>Counts <paramref name="image"/> against the request, or refuses it when the request's images would exceed <see cref="MaxRequestImageBytes"/>.</summary>
        public void Take(NativeImageData image)
        {
            _taken += image.Rgb.Length;
            if (_taken > MaxRequestImageBytes)
                throw new HartsyInferenceException($"The images in one request decode to more than {MaxRequestImageBytes / (1024 * 1024)} MiB together. Send fewer or smaller images.");
        }
    }

    /// <summary>OpenAI <c>reasoning_effort</c>: a name the engine defines (<c>low</c>, <c>high</c>, <c>max</c>) or an integer in [1, 100]. A string or a number is read
    /// by <see cref="EncodeOptions.ParseReasoningEffort"/>, the parser the CLI uses; anything else is the caller's error.</summary>
    internal static int? ParseReasoningEffort(JsonElement? value) => value?.ValueKind switch
    {
        null or JsonValueKind.Null => null,
        JsonValueKind.String => EncodeOptions.ParseReasoningEffort(value.Value.GetString()!),
        // The number as written: only plain digits are an integer effort, so 1.5 and 1e2 are refused.
        JsonValueKind.Number => EncodeOptions.ParseReasoningEffort(value.Value.GetRawText()),
        _ => throw new ArgumentException($"reasoning_effort must be a name (low, high, max) or an integer in [1, {EncodeOptions.MaxReasoningEffort}]."),
    };

    /// <summary>OpenAI <c>thinking</c>: <c>enabled</c> or <c>disabled</c>. Unset leaves the template's default.</summary>
    internal static bool? ParseThinking(ChatThinkingDto? thinking) => thinking?.Type switch
    {
        null => null,
        "enabled" => true,
        "disabled" => false,
        _ => throw new ArgumentException($"thinking.type must be 'enabled' or 'disabled', not '{thinking!.Type}'."),
    };

    /// <summary>OpenAI-style <c>priority</c>: <c>low</c>, <c>normal</c> or <c>high</c>. Unset is normal.</summary>
    internal static RequestPriority? ParsePriority(string? priority) => priority switch
    {
        null => null,
        "low" => RequestPriority.Low,
        "normal" => RequestPriority.Normal,
        "high" => RequestPriority.High,
        _ => throw new ArgumentException($"priority must be 'low', 'normal' or 'high', not '{priority}'."),
    };

    internal static NativeToolCall ToNativeToolCall(ChatToolCallDto call) => new NativeToolCall
    {
        Id = call.Id,
        Name = call.Function.Name ?? "",
        Arguments = string.IsNullOrEmpty(call.Function.Arguments) ? "{}" : call.Function.Arguments,
    };

    internal static ChatToolCallDto ToToolCallDto(NativeToolCall call, int? index = null) => new ChatToolCallDto
    {
        Id = string.IsNullOrEmpty(call.Id) ? $"call_{Guid.NewGuid():N}" : call.Id,
        Index = index,
        Function = new ChatToolCallFunctionDto { Name = call.Name, Arguments = call.Arguments },
    };

    // OpenAI's finish_reason vocabulary has no slot for Cancelled/Error — both collapse to "stop" (best-effort
    // compat) rather than inventing a non-standard value a client SDK won't recognize.
    internal static string ToFinishReason(StopReason stop) => stop switch
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

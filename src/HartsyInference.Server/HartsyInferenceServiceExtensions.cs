using System.Diagnostics;
using System.Threading.Channels;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Requests;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.Server.Imaging;
using Microsoft.AspNetCore.Diagnostics;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;

namespace HartsyInference.Server;

/// <summary>DI registration and endpoint mapping for the OpenAI-compatible HartsyInference server.</summary>
public static class HartsyInferenceServiceExtensions
{
    /// <summary>Registers the compute backend, model manager, inference queue, and options.</summary>
    public static IServiceCollection AddHartsyInference(this IServiceCollection services, Action<HartsyInferenceServerOptions>? configure = null)
    {
        HartsyInferenceServerOptions options = new HartsyInferenceServerOptions();
        configure?.Invoke(options);
        services.AddSingleton(options);

        services.AddSingleton<IBackend>(_ => options.Backend switch
        {
            BackendKind.Cuda => new CudaBackend(options.CudaDevice, options.PtxDirectory),
            _ => new CpuBackend(),
        });

        services.AddSingleton(new InferenceQueue(options.MaxConcurrency, options.MaxQueueDepth));
        services.AddSingleton(sp => new ModelManager(sp.GetRequiredService<IBackend>(), options, sp.GetRequiredService<InferenceQueue>()));
        return services;
    }

    /// <summary>Maps health probes, optional API-key auth, the OpenAI image endpoints, and model management.</summary>
    public static void MapHartsyInferenceEndpoints(this WebApplication app)
    {
        HartsyInferenceServerOptions options = app.Services.GetRequiredService<HartsyInferenceServerOptions>();

        // Last-resort safety net: catches any exception a route handler doesn't already handle itself
        // (chat/images already have their own try/catch for the exceptions they expect — this is for
        // whatever's left, e.g. a bug in request-shape validation that runs before those, or a future route
        // added without one) and turns it into a structured, logged 500 instead of a bare/blank response.
        // Registered first so it wraps every middleware/route below it. This does NOT and cannot catch a
        // corrupted-state exception (e.g. AccessViolationException from native/unsafe code) — the CLR
        // terminates the process for those before any handler, including this one, gets a chance to run;
        // that class of failure needs process-level supervision (see deploy/ for a restart policy), not
        // in-process handling.
        app.UseExceptionHandler(errApp =>
        {
            errApp.Run(async ctx =>
            {
                Exception? ex = ctx.Features.Get<IExceptionHandlerPathFeature>()?.Error;
                ILogger log = ctx.RequestServices.GetRequiredService<ILoggerFactory>().CreateLogger("HartsyInference.UnhandledException");
                log.LogError(ex, "Unhandled exception on {Method} {Path}", ctx.Request.Method, ctx.Request.Path);
                await WriteError(ctx, StatusCodes.Status500InternalServerError, "An unexpected server error occurred.", "server_error");
            });
        });

        // Optional API-key gate over /v1/*.
        if (!string.IsNullOrEmpty(options.ApiKey))
        {
            app.Use(async (ctx, next) =>
            {
                if (ctx.Request.Path.StartsWithSegments("/v1") && !IsAuthorized(ctx, options.ApiKey!))
                {
                    await WriteError(ctx, StatusCodes.Status401Unauthorized, "Invalid or missing API key.", "invalid_request_error");
                    return;
                }
                await next();
            });
        }

        // Liveness vs readiness, standard k8s/systemd distinction: /health only answers "is the process
        // responding to HTTP at all" (deliberately cheap and dependency-free — if it did real checks, a
        // transient backend hiccup would make an orchestrator kill+restart a process that was actually
        // fine). /ready answers "can this instance actually serve chat traffic right now" by checking every
        // loaded model's DynamicBatchScheduler loop is still alive — a model whose loop died (see that
        // class's fault-containment doc) looks identical to a healthy one on /health (the process is still
        // up) but will hang every request against it forever, which is exactly what /ready exists to catch.
        app.MapGet("/health", () => Results.Ok(new { status = "healthy" }));
        app.MapGet("/ready", (ModelManager mm) =>
        {
            IReadOnlyCollection<string> unhealthy = mm.UnhealthyChatModels;
            if (unhealthy.Count > 0)
                return Results.Json(new { status = "degraded", unhealthyModels = unhealthy }, statusCode: StatusCodes.Status503ServiceUnavailable);
            return Results.Ok(new { status = "ready" });
        });

        MapModels(app);
        MapImages(app);
        MapChat(app);
        MapUnsupportedAudio(app);
    }

    private static void MapChat(WebApplication app)
    {
        // OpenAI multiplexes streaming via the "stream" request field on the SAME route (unlike /v1/images,
        // which uses a separate /stream URL) — client SDKs expect exactly this, so we match it rather than
        // inventing a HartsyInference-specific shape.
        app.MapPost("/v1/chat/completions", async (ChatCompletionRequest req, ModelManager mm, InferenceQueue queue,
            HttpContext ctx, ILogger<Program> log, CancellationToken ct) =>
        {
            // Validation failures return a normal JSON error regardless of req.Stream — nothing has been
            // written to the response yet (SSE headers are only set once we know the request is valid), so
            // there's no reason to special-case streaming clients here; they handle a non-2xx JSON body fine.
            // Pure request-SHAPE checks (don't need any server state) run before the model-loaded check
            // (needs ModelManager state) — fail fast on a malformed request regardless of what's loaded.
            if (req.Messages.Count == 0)
                return Problem(StatusCodes.Status400BadRequest, "Field 'messages' must be non-empty.", "invalid_request_error");
            if (req.ResponseFormat is { Type: not "text" and not "json_object" })
                return Problem(StatusCodes.Status400BadRequest, $"response_format.type '{req.ResponseFormat.Type}' is not supported — only 'text' (default) and 'json_object' are; 'json_schema' (schema-constrained, not just valid-JSON) is a separate, larger feature that isn't built yet.", "invalid_request_error");
            if (req.Model is null || !mm.IsChatModel(req.Model))
                return Problem(StatusCodes.Status400BadRequest, "Field 'model' must name a loaded chat (LLM/SSM) model (POST /v1/models/load first).", "invalid_request_error");

            GenerationRequest genReq = BuildGenerationRequest(req);
            string id = $"chatcmpl-{Guid.NewGuid():N}";
            long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            Stopwatch sw = Stopwatch.StartNew();
            // Backend-contention depth across the WHOLE server (diffusion + every model's chat scheduler share
            // one InferenceQueue gate) — NOT a per-chat-request queue depth anymore: concurrently-submitted
            // chat requests against the same model batch together into shared decode rounds rather than each
            // taking their own queue slot (see DynamicBatchScheduler's "Backend exclusivity" doc).
            int gateDepthAtEntry = queue.PendingCount;

            if (!req.Stream)
            {
                try
                {
                    // No token needs to reach the client for a non-streaming response, but the callback is
                    // still the only way to timestamp when the FIRST token was produced — capturing it here
                    // is what makes time-to-first-token measurable at all for this path (previously only
                    // total elapsed time was logged, which conflates queueing/prefill latency with per-token
                    // decode throughput and can't tell you which one regressed).
                    TimeSpan? firstTokenAt = null;
                    GenerationResult result = await mm.GenerateChatAsync(req.Model!, genReq, onToken: _ => firstTokenAt ??= sw.Elapsed, ct);
                    sw.Stop();
                    LogCompletion(log, req.Model!, gateDepthAtEntry, result.PromptTokens, result.TokenIds.Count, sw.Elapsed, firstTokenAt);

                    return Results.Ok(new ChatCompletionResponse
                    {
                        Id = id,
                        Created = created,
                        Model = req.Model!,
                        Choices = [new ChatCompletionChoice
                        {
                            Message = new ChatMessageDto { Role = "assistant", Content = result.Text },
                            FinishReason = result.StoppedOnStopToken ? "stop" : "length",
                        }],
                        Usage = new ChatUsage { PromptTokens = result.PromptTokens, CompletionTokens = result.TokenIds.Count },
                    });
                }
                catch (KvPoolExhaustedException ex)
                {
                    // Distinct `type` from QueueFullException below even though both are 429: this one means
                    // "this model's KV capacity is full" (a per-model resource limit — retrying against a
                    // DIFFERENT model would succeed immediately), not "the server's overall request queue is
                    // full" (a global limit — retrying anything would hit the same wall). Same status code
                    // (429 is correct for both — "back off and retry"), different machine-readable cause.
                    return Problem(StatusCodes.Status429TooManyRequests, ex.Message, "insufficient_capacity");
                }
                catch (QueueFullException ex)
                {
                    return Problem(StatusCodes.Status429TooManyRequests, ex.Message, "rate_limit_error");
                }
                catch (OperationCanceledException)
                {
                    log.LogInformation("chat completion {Id} cancelled by client after {Elapsed}ms", id, sw.ElapsedMilliseconds);
                    return Results.StatusCode(StatusCodes.Status499ClientClosedRequest);
                }
                catch (Exception ex)
                {
                    log.LogError(ex, "chat completion {Id} failed", id);
                    return Problem(StatusCodes.Status500InternalServerError, ex.Message, "server_error");
                }
            }

            // Streaming: one SSE "chat.completion.chunk" per token, then a final chunk with finish_reason, then [DONE].
            ctx.Response.Headers.ContentType = "text/event-stream";
            Channel<string> events = Channel.CreateUnbounded<string>();
            int tokenCount = 0;
            TimeSpan? ttft = null;

            _ = Task.Run(async () =>
            {
                try
                {
                    events.Writer.TryWrite(SseChunk(new ChatCompletionChunk
                    {
                        Id = id, Created = created, Model = req.Model!,
                        Choices = [new ChatCompletionChunkChoice { Delta = new ChatCompletionDelta { Role = "assistant" } }],
                    }));
                    GenerationResult result = await mm.GenerateChatAsync(req.Model!, genReq, onToken: tokenId =>
                    {
                        ttft ??= sw.Elapsed;
                        tokenCount++;
                        // Per-token text isn't available from a raw token id without re-decoding incrementally
                        // (tokenizers aren't guaranteed to decode single ids to valid UTF-8 in isolation for
                        // BPE-merged scripts), so stream token COUNT progress and send the real text in the
                        // final chunk — still gives clients live progress without a garbled-text risk.
                        events.Writer.TryWrite($"event: token\ndata: {{\"count\":{tokenCount}}}\n\n");
                    }, ct);
                    sw.Stop();
                    LogCompletion(log, req.Model!, gateDepthAtEntry, result.PromptTokens, result.TokenIds.Count, sw.Elapsed, ttft);

                    events.Writer.TryWrite(SseChunk(new ChatCompletionChunk
                    {
                        Id = id, Created = created, Model = req.Model!,
                        Choices = [new ChatCompletionChunkChoice
                        {
                            Delta = new ChatCompletionDelta { Content = result.Text },
                            FinishReason = result.StoppedOnStopToken ? "stop" : "length",
                        }],
                    }));
                    events.Writer.TryWrite("data: [DONE]\n\n");
                }
                catch (OperationCanceledException)
                {
                    log.LogInformation("streaming chat completion {Id} cancelled by client after {Elapsed}ms", id, sw.ElapsedMilliseconds);
                }
                catch (Exception ex)
                {
                    log.LogError(ex, "streaming chat completion {Id} failed", id);
                    events.Writer.TryWrite($"event: error\ndata: {{\"message\":{System.Text.Json.JsonSerializer.Serialize(ex.Message)}}}\n\n");
                }
                finally
                {
                    events.Writer.Complete();
                }
            }, ct);

            await foreach (string evt in events.Reader.ReadAllAsync(ct))
            {
                await ctx.Response.WriteAsync(evt, ct);
                await ctx.Response.Body.FlushAsync(ct);
            }
            return Results.Empty;
        });
    }

    private static GenerationRequest BuildGenerationRequest(ChatCompletionRequest req)
    {
        SamplingOptions baseSampling = SamplingOptions.Default;
        SamplingOptions sampling = baseSampling with
        {
            Temperature = req.Temperature ?? baseSampling.Temperature,
            TopP = req.TopP ?? baseSampling.TopP,
            TopK = req.TopK ?? baseSampling.TopK,
            MinP = req.MinP ?? baseSampling.MinP,
            RepetitionPenalty = req.RepetitionPenalty ?? baseSampling.RepetitionPenalty,
            Seed = req.Seed ?? baseSampling.Seed,
            Greedy = req.Temperature is 0f,
            JsonMode = req.ResponseFormat?.Type == "json_object",
        };
        return new GenerationRequest
        {
            Messages = [.. req.Messages.Select(m => new ChatMessage(m.Role, m.Content))],
            MaxTokens = req.MaxTokens ?? 256,
            Sampling = sampling,
        };
    }

    private static string SseChunk(ChatCompletionChunk chunk) =>
        $"data: {System.Text.Json.JsonSerializer.Serialize(chunk)}\n\n";

    private static void LogCompletion(ILogger log, string model, int queueDepthAtEntry, int promptTokens, int completionTokens, TimeSpan elapsed, TimeSpan? ttft)
    {
        // tokensPerSec is computed over the FULL elapsed time (queueing + prefill + decode), so it reads low
        // for short completions dominated by a slow prefill — ttftMs isolates that: a regression in prefill
        // latency shows up as ttftMs growing with tokensPerSec unchanged, while a decode-throughput
        // regression shows up as the reverse. ttftMs is -1 when no token was produced at all (e.g. the model
        // immediately hit a stop token) rather than a misleading 0.
        double tokensPerSec = completionTokens > 0 && elapsed.TotalSeconds > 0 ? completionTokens / elapsed.TotalSeconds : 0;
        log.LogInformation(
            "chat completion: model={Model} queueDepth={QueueDepth} promptTokens={PromptTokens} completionTokens={CompletionTokens} elapsedMs={ElapsedMs} ttftMs={TtftMs} tokensPerSec={TokensPerSec:F1}",
            model, queueDepthAtEntry, promptTokens, completionTokens, elapsed.TotalMilliseconds, ttft?.TotalMilliseconds ?? -1, tokensPerSec);
    }

    private static void MapModels(WebApplication app)
    {
        app.MapGet("/v1/models", (ModelManager mm) =>
            Results.Ok(new ModelListResponse { Data = mm.LoadedModels.Select(id => new ModelEntry { Id = id }).ToArray() }));

        app.MapPost("/v1/models/load", async (ModelLoadRequest req, ModelManager mm, CancellationToken ct) =>
        {
            if (string.IsNullOrWhiteSpace(req.Model))
                return Problem(StatusCodes.Status400BadRequest, "Field 'model' is required.", "invalid_request_error");
            try
            {
                string arch = await mm.LoadAsync(req.Model, ct);
                return Results.Ok(new { loaded = req.Model, architecture = arch });
            }
            catch (Exception ex)
            {
                return Problem(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error");
            }
        });

        // pull is an alias of load (download-from-HuggingFace then construct).
        app.MapPost("/v1/models/pull", async (ModelLoadRequest req, ModelManager mm, CancellationToken ct) =>
        {
            if (string.IsNullOrWhiteSpace(req.Model))
                return Problem(StatusCodes.Status400BadRequest, "Field 'model' is required.", "invalid_request_error");
            try
            {
                string arch = await mm.LoadAsync(req.Model, ct);
                return Results.Ok(new { pulled = req.Model, architecture = arch });
            }
            catch (Exception ex)
            {
                return Problem(StatusCodes.Status400BadRequest, ex.Message, "invalid_request_error");
            }
        });

        app.MapDelete("/v1/models/{id}", (string id, ModelManager mm) =>
            mm.Unload(id) ? Results.Ok(new { unloaded = id }) : Problem(StatusCodes.Status404NotFound, $"Model '{id}' is not loaded.", "invalid_request_error"));
    }

    private static void MapImages(WebApplication app)
    {
        app.MapPost("/v1/images/generations", async (ImageGenerationRequest req, ModelManager mm, InferenceQueue queue, CancellationToken ct) =>
        {
            if (req.Model is null || !mm.IsLoaded(req.Model))
                return Problem(StatusCodes.Status400BadRequest, "Field 'model' must name a loaded model (POST /v1/models/load first).", "invalid_request_error");

            try
            {
                List<ImageData> images = await queue.EnqueueAsync(async () =>
                {
                    List<ImageData> outImages = new();
                    int n = Math.Max(1, req.N);
                    for (int i = 0; i < n; i++)
                    {
                        (byte[] rgb, int w, int h, int _) = mm.GenerateImage(req.Model!, req);
                        byte[] png = PngImageWriter.Encode(rgb, w, h);
                        outImages.Add(new ImageData { B64Json = Convert.ToBase64String(png) });
                    }
                    return await Task.FromResult(outImages);
                }, ct);

                return Results.Ok(new ImageGenerationResponse { Created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(), Data = images });
            }
            catch (QueueFullException ex)
            {
                return Problem(StatusCodes.Status429TooManyRequests, ex.Message, "rate_limit_error");
            }
            catch (NotSupportedException ex)
            {
                return Problem(StatusCodes.Status501NotImplemented, ex.Message, "invalid_request_error");
            }
            catch (Exception ex)
            {
                return Problem(StatusCodes.Status500InternalServerError, ex.Message, "server_error");
            }
        });

        // SSE streaming: emits step-progress events, then a final event carrying the base64 PNG.
        app.MapPost("/v1/images/generations/stream", async (ImageGenerationRequest req, ModelManager mm, InferenceQueue queue, HttpContext ctx, CancellationToken ct) =>
        {
            if (req.Model is null || !mm.IsLoaded(req.Model))
            {
                await WriteError(ctx, StatusCodes.Status400BadRequest, "Field 'model' must name a loaded model.", "invalid_request_error");
                return;
            }

            ctx.Response.Headers.ContentType = "text/event-stream";
            Channel<string> events = Channel.CreateUnbounded<string>();

            _ = queue.EnqueueAsync(async () =>
            {
                try
                {
                    (byte[] rgb, int w, int h, int _) = mm.GenerateImage(req.Model!, req, p =>
                        events.Writer.TryWrite($"event: progress\ndata: {{\"step\":{p.Step},\"total\":{p.TotalSteps}}}\n\n"));
                    byte[] png = PngImageWriter.Encode(rgb, w, h);
                    events.Writer.TryWrite($"event: complete\ndata: {{\"b64_json\":\"{Convert.ToBase64String(png)}\"}}\n\n");
                }
                catch (Exception ex)
                {
                    events.Writer.TryWrite($"event: error\ndata: {{\"message\":{System.Text.Json.JsonSerializer.Serialize(ex.Message)}}}\n\n");
                }
                finally
                {
                    events.Writer.Complete();
                }
                return await Task.FromResult(true);
            }, ct);

            await foreach (string evt in events.Reader.ReadAllAsync(ct))
            {
                await ctx.Response.WriteAsync(evt, ct);
                await ctx.Response.Body.FlushAsync(ct);
            }
        });

        // img2img / inpaint — pending (needs the VAE-encoder img2img request path wired through the factory).
        app.MapPost("/v1/images/edits", () =>
            Problem(StatusCodes.Status501NotImplemented, "Image edits (img2img/inpaint) are not yet exposed over HTTP.", "invalid_request_error"));
    }

    private static void MapUnsupportedAudio(WebApplication app)
    {
        // Audio endpoints are routed for API-shape completeness; wiring the Whisper/Kokoro pipelines is pending.
        app.MapPost("/v1/audio/transcriptions", () =>
            Problem(StatusCodes.Status501NotImplemented, "Audio transcription endpoint is not yet wired to the Whisper pipeline.", "invalid_request_error"));
        app.MapPost("/v1/audio/speech", () =>
            Problem(StatusCodes.Status501NotImplemented, "Audio speech endpoint is not yet wired to the Kokoro TTS pipeline.", "invalid_request_error"));
    }

    private static bool IsAuthorized(HttpContext ctx, string apiKey)
    {
        if (ctx.Request.Headers.TryGetValue("x-api-key", out Microsoft.Extensions.Primitives.StringValues k) && k == apiKey) return true;
        if (ctx.Request.Headers.Authorization.ToString() is { } auth &&
            auth.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase) &&
            auth["Bearer ".Length..] == apiKey)
        {
            return true;
        }
        return false;
    }

    private static IResult Problem(int status, string message, string type) =>
        Results.Json(OpenAiError.Make(message, type), statusCode: status);

    private static async Task WriteError(HttpContext ctx, int status, string message, string type)
    {
        ctx.Response.StatusCode = status;
        await ctx.Response.WriteAsJsonAsync(OpenAiError.Make(message, type));
    }
}

using System.Diagnostics;
using System.Threading.RateLimiting;
using HartsyInference.API.Endpoints;
using HartsyInference.Engine;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Audio.Wake;
using HartsyInference.Engine.Services;
using Microsoft.AspNetCore.Diagnostics;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.Extensions.Primitives;
using OpenTelemetry.Metrics;
using OpenTelemetry.Trace;

namespace HartsyInference.API;

/// <summary>DI registration and endpoint mapping for the HartsyInference server.</summary>
public static class HartsyInferenceServiceExtensions
{
    /// <summary>The <c>HttpContext.Items</c> key the auth middleware stashes a resolved <see cref="ApiKeyIdentity"/>
    /// under, for the rate limiter's partition key, <see cref="UsageTracker"/>'s counter key, and the request's
    /// <see cref="Activity"/> tag to share without re-parsing headers.</summary>
    internal const string ApiKeyIdentityItemKey = "HartsyInference.ApiKeyIdentity";

    /// <summary>Registers the inference engine facade and its concurrency gate.</summary>
    public static IServiceCollection AddHartsyInference(this IServiceCollection services, Action<HartsyInferenceServerOptions>? configure = null)
    {
        HartsyInferenceServerOptions options = new HartsyInferenceServerOptions();
        configure?.Invoke(options);
        services.AddSingleton(options);

        if (!string.IsNullOrWhiteSpace(options.KernelDirectory))
            BackendFactory.KernelDirOverride = options.KernelDirectory;

        // One long-lived facade for the process, exactly like the CLI's one-per-invocation InferenceEngine —
        // just scoped to app lifetime instead of one command. IInferenceEngine is NOT safely re-entrant per
        // backend, so every call site must go through the InferenceQueue below rather than calling it directly.
        services.AddSingleton<IInferenceEngine>(_ => new InferenceEngine(options.Backend));

        // Two gates: the unkeyed "fast" queue (unchanged — every route from Phases 1-4 keeps resolving it by
        // type) and a keyed "long-running" one for video generation and opening world sessions, which can each
        // run for minutes. Splitting them means one slow video job can't starve every fast request queued behind
        // it — see HartsyInferenceServerOptions.MaxLongRunningConcurrency.
        services.AddSingleton(new InferenceQueue(options.MaxConcurrency, options.MaxQueueDepth));
        services.AddKeyedSingleton(QueueKeys.LongRunning,
            new InferenceQueue(options.MaxLongRunningConcurrency, options.MaxLongRunningQueueDepth));

        services.AddSingleton(new WorldSessionRegistry(TimeSpan.FromMinutes(options.WorldSessionIdleTimeoutMinutes)));

        // The wake listener is a peer of the HTTP surface, not a route: satellites hold one long-lived TCP
        // connection each rather than making requests. It deliberately bypasses InferenceQueue — see WakeWorker.
        if (options.WakeEnabled)
        {
            services.AddSingleton(sp =>
            {
                InferenceQueue queue = sp.GetRequiredService<InferenceQueue>();
                return new WakeService(sp.GetRequiredService<IInferenceEngine>(), new WakeServiceOptions
                {
                    Port = options.WakePort,
                    ModelRoot = options.WakeModelRoot,
                    TranscribeOnDetection = options.WakeTranscribeOnDetection,
                    // Detection itself runs off-queue on its own backend, but the transcription it triggers
                    // shares the engine backend with every HTTP route, and the engine is not re-entrant there.
                    TranscribeGate = work => queue.EnqueueAsync(work, CancellationToken.None),
                });
            });
        }

        // Production-hardening surface: caller identity, per-caller usage, and the domain-specific metrics
        // ApiMetrics adds on top of ASP.NET Core's own request instrumentation (see WithMetrics below).
        services.AddSingleton(new ApiKeyStore(options));
        services.AddSingleton<UsageTracker>();
        services.AddSingleton<ApiMetrics>();

        // Per-caller rate limiting, layered ABOVE the InferenceQueue capacity gate -- a distinct failure mode
        // (quota exhausted vs. server momentarily saturated), given a distinct error `type` in OnRejected below
        // so a client can tell the two apart. Fixed-window with QueueLimit 0: reject immediately rather than
        // queueing here too, since InferenceQueue already owns queueing semantics for the actual generation work.
        services.AddRateLimiter(rl =>
        {
            rl.GlobalLimiter = PartitionedRateLimiter.Create<HttpContext, string>(ctx =>
            {
                if (IsProbePath(ctx.Request.Path))
                    return RateLimitPartition.GetNoLimiter("probe");

                ApiKeyIdentity? identity = ctx.Items.TryGetValue(ApiKeyIdentityItemKey, out object? v) ? v as ApiKeyIdentity : null;
                string partitionKey = identity?.Name ?? ctx.Connection.RemoteIpAddress?.ToString() ?? "unknown";
                int permitLimit = identity?.RateLimitPerMinute ?? options.DefaultRateLimitPerMinute;

                return RateLimitPartition.GetFixedWindowLimiter(partitionKey, _ => new FixedWindowRateLimiterOptions
                {
                    PermitLimit = permitLimit,
                    Window = TimeSpan.FromMinutes(1),
                    QueueLimit = 0,
                });
            });
            rl.OnRejected = async (context, ct) =>
            {
                context.HttpContext.Response.Headers.RetryAfter = "60";
                await WriteErrorAsync(context.HttpContext, StatusCodes.Status429TooManyRequests,
                    "Rate limit exceeded for this API key.", "rate_limit_exceeded");
            };
        });

        // Metrics: ASP.NET Core's own instrumentation (request duration/count/in-flight) plus ApiMetrics'
        // domain-specific series, scraped in Prometheus text format at GET /metrics (mapped below).
        // Tracing: auto-instruments every request as an Activity honoring the W3C traceparent header; the auth
        // middleware below tags it with the resolved caller identity so traces/logs are correlatable by caller.
        services.AddOpenTelemetry()
            .WithMetrics(b => b.AddAspNetCoreInstrumentation().AddMeter(ApiMetrics.MeterName).AddPrometheusExporter())
            .WithTracing(b => b.AddAspNetCoreInstrumentation());

        services.AddOpenApi();

        services.ConfigureHttpJsonOptions(o =>
            o.SerializerOptions.Converters.Add(new System.Text.Json.Serialization.JsonStringEnumConverter()));

        return services;
    }

    /// <summary>Maps health/settings/admin probes, API-key auth + per-key rate limiting + usage metering,
    /// observability endpoints, and native + OpenAI-compat generation endpoints — all onto <see cref="IInferenceEngine"/>.</summary>
    public static void MapHartsyInferenceEndpoints(this WebApplication app)
    {
        HartsyInferenceServerOptions options = app.Services.GetRequiredService<HartsyInferenceServerOptions>();
        ApiKeyStore apiKeyStore = app.Services.GetRequiredService<ApiKeyStore>();
        UsageTracker usageTracker = app.Services.GetRequiredService<UsageTracker>();
        ApiMetrics metrics = app.Services.GetRequiredService<ApiMetrics>();

        if (options.WakeEnabled)
        {
            WakeService wake = app.Services.GetRequiredService<WakeService>();
            // Started here rather than lazily on first use: an always-on listener that only appears once
            // something asks for it would silently miss every satellite until then.
            app.Lifetime.ApplicationStarted.Register(() =>
            {
                try { wake.Start(); }
                catch (Exception ex) { Logs.Error("[Audio][Wake] Listener failed to start; satellites cannot connect.", ex); }
            });
            app.Lifetime.ApplicationStopping.Register(wake.Dispose);
        }

        // Last-resort safety net: catches any exception a route handler doesn't already handle itself and turns
        // it into a structured, logged response instead of a bare/blank one. Registered first so it wraps every
        // middleware/route below it. This does NOT and cannot catch a corrupted-state exception (e.g.
        // AccessViolationException from native/unsafe code) — the CLR terminates the process before any handler,
        // including this one, gets a chance to run; that class of failure needs process-level supervision (see
        // deploy/ for a restart policy), not in-process handling.
        app.UseExceptionHandler(errApp =>
        {
            errApp.Run(async ctx =>
            {
                Exception? ex = ctx.Features.Get<IExceptionHandlerPathFeature>()?.Error;
                ILogger log = ctx.RequestServices.GetRequiredService<ILoggerFactory>().CreateLogger("HartsyInference.UnhandledException");
                log.LogError(ex, "Unhandled exception on {Method} {Path}", ctx.Request.Method, ctx.Request.Path);

                // A malformed/incomplete JSON body (e.g. a missing `required` field) fails minimal-API's own body
                // binding and throws BadHttpRequestException BEFORE the route handler runs — that carries the
                // real client-error status (400) in .StatusCode. Passing it through here instead of collapsing
                // everything to 500 is what makes "missing required field" behave like a 400, not a server fault.
                if (ex is Microsoft.AspNetCore.Http.BadHttpRequestException badRequest)
                {
                    await WriteErrorAsync(ctx, badRequest.StatusCode, badRequest.Message, "invalid_request_error");
                    return;
                }

                await WriteErrorAsync(ctx, StatusCodes.Status500InternalServerError, "An unexpected server error occurred.", "server_error");
            });
        });

        // Optional API-key gate over everything except the liveness/readiness/version probes — those need to
        // stay reachable for an orchestrator's health checks regardless of auth config. Resolves the caller to a
        // named identity (not just a bool) and stashes it for the rate limiter, usage tracker, and tracing below.
        if (apiKeyStore.HasAnyKeys)
        {
            app.Use(async (ctx, next) =>
            {
                if (IsProbePath(ctx.Request.Path)) { await next(); return; }

                string? presented = ResolvePresentedKey(ctx);
                if (presented is null || !apiKeyStore.TryResolve(presented, out ApiKeyIdentity identity))
                {
                    await WriteErrorAsync(ctx, StatusCodes.Status401Unauthorized, "Invalid or missing API key.", "invalid_request_error");
                    return;
                }
                ctx.Items[ApiKeyIdentityItemKey] = identity;
                Activity.Current?.SetTag("hartsyinference.caller", identity.Name);
                await next();
            });
        }

        // Usage metering: wraps rate limiting (not the other way round) so a 429 rejection still counts against
        // the caller's usage record, not just successful requests. Coarse per-top-level-route-segment modality
        // tag derived from the path rather than per-endpoint metadata — keeps this to one place instead of
        // touching all ~25 route files. Exceptions are recorded as errors and rethrown unchanged for
        // UseExceptionHandler above to turn into a response.
        app.Use(async (ctx, next) =>
        {
            bool isError = false;
            try
            {
                await next();
                isError = ctx.Response.StatusCode >= 400;
            }
            catch
            {
                isError = true;
                throw;
            }
            finally
            {
                string identityName = ctx.Items.TryGetValue(ApiKeyIdentityItemKey, out object? v) && v is ApiKeyIdentity id ? id.Name : "anonymous";
                string modality = ModalityFromPath(ctx.Request.Path);
                usageTracker.Record(identityName, modality, isError);
                metrics.RecordRequest(modality, isError ? "error" : "ok");
            }
        });

        app.UseRateLimiter();

        app.MapHealthEndpoints();
        app.MapSettingsEndpoints();
        app.MapAdminEndpoints();
        app.MapImageEndpoints();
        app.MapTextEndpoints();
        app.MapAudioEndpoints();
        app.MapEmbeddingEndpoints();
        app.MapVisionEndpoints();
        app.MapMeshEndpoints();
        app.MapVideoEndpoints();
        app.MapRestoreEndpoints();
        app.MapWorldEndpoints();
        app.MapCompatEndpoints();

        // Observability/discovery surface. Neither is a liveness/readiness probe, so both stay behind the same
        // auth gate as /admin/* when a key is configured (queue depth and route shapes are ops-sensitive, same
        // reasoning as /admin/queue already being gated).
        app.MapPrometheusScrapingEndpoint("/metrics");
        app.MapOpenApi();
    }

    private static bool IsProbePath(PathString path) =>
        path.StartsWithSegments("/health") || path.StartsWithSegments("/ready") || path.StartsWithSegments("/version");

    /// <summary>Coarse modality tag for usage/metrics, derived from the route path so this doesn't require
    /// touching every endpoint file to attach metadata: the first segment after <c>/v1/native/</c> for native
    /// routes, otherwise a fixed bucket per route group.</summary>
    private static string ModalityFromPath(PathString path)
    {
        string p = path.Value ?? "";
        const string nativePrefix = "/v1/native/";
        if (p.StartsWith(nativePrefix, StringComparison.OrdinalIgnoreCase))
        {
            string rest = p[nativePrefix.Length..];
            int slash = rest.IndexOf('/');
            return slash >= 0 ? rest[..slash] : rest;
        }
        if (p.StartsWith("/v1/", StringComparison.OrdinalIgnoreCase)) return "compat";
        if (p.StartsWith("/admin/", StringComparison.OrdinalIgnoreCase)) return "admin";
        if (IsProbePath(path) || p.StartsWith("/settings", StringComparison.OrdinalIgnoreCase)) return "probe";
        if (p.StartsWith("/metrics", StringComparison.OrdinalIgnoreCase) || p.StartsWith("/openapi", StringComparison.OrdinalIgnoreCase)) return "meta";
        return "other";
    }

    private static string? ResolvePresentedKey(HttpContext ctx)
    {
        if (ctx.Request.Headers.TryGetValue("x-api-key", out StringValues k) && !StringValues.IsNullOrEmpty(k))
            return k.ToString();
        string auth = ctx.Request.Headers.Authorization.ToString();
        if (auth.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            return auth["Bearer ".Length..];
        return null;
    }

    /// <summary>Structured error response, OpenAI-shaped for continuity with the compat routes that will land
    /// alongside generation endpoints. Internal so the <c>Endpoints/*.cs</c> route groups can share it.</summary>
    internal static IResult Problem(int status, string message, string type) =>
        Results.Json(OpenAiError.Make(message, type), statusCode: status);

    internal static async Task WriteErrorAsync(HttpContext ctx, int status, string message, string type)
    {
        ctx.Response.StatusCode = status;
        await ctx.Response.WriteAsJsonAsync(OpenAiError.Make(message, type));
    }
}

/// <summary>Keyed-DI keys for the two <see cref="InferenceQueue"/> instances.</summary>
internal static class QueueKeys
{
    /// <summary>The video/world-session queue, injected via <c>[FromKeyedServices(QueueKeys.LongRunning)]</c>.</summary>
    public const string LongRunning = "long-running";
}

using System.Text.Json;
using System.Threading.Channels;
using HartsyInference.Engine;
using Microsoft.AspNetCore.Http.Json;
using Microsoft.Extensions.Options;

namespace HartsyInference.API.Endpoints;

/// <summary>Shared SSE plumbing for every streaming generation route.</summary>
internal static class SseHelpers
{
    /// <summary>Runs <paramref name="produce"/> under <paramref name="queue"/>'s concurrency gate, draining whatever it writes to its <see cref="ChannelWriter{T}"/> straight to the response as <c>text/event-stream</c>. <paramref name="produce"/> receives the app's configured <see cref="JsonSerializerOptions"/> (camelCase, string enums) so <see cref="Event"/> calls match every non-streaming response's wire shape. Checked synchronously for an immediate <see cref="QueueFullException"/> before committing to the SSE response — enqueuing fire-and-forget without this check leaves a full-queue client hanging on events that never arrive instead of getting a proper 429.</summary>
    public static async Task RunAsync(
        HttpContext ctx, InferenceQueue queue, Func<ChannelWriter<string>, JsonSerializerOptions, Task> produce, CancellationToken ct)
    {
        JsonSerializerOptions jsonOptions = ResolveJsonOptions(ctx);
        Channel<string> events = Channel.CreateUnbounded<string>();

        Task queued = queue.EnqueueAsync(async () =>
        {
            try
            {
                await produce(events.Writer, jsonOptions);
            }
            catch (Exception ex)
            {
                events.Writer.TryWrite(ErrorEvent(ex.Message, jsonOptions));
            }
            finally
            {
                events.Writer.Complete();
            }
            return true;
        }, ct);

        // An async method that throws before its first await completes the returned task SYNCHRONOUSLY as
        // Faulted — EnqueueAsync does exactly that for QueueFullException (thrown before the first `await
        // _slots.WaitAsync`), so this check reliably catches "queue was already full" without ever touching the
        // response stream.
        if (queued.IsFaulted)
        {
            try
            {
                await queued;
            }
            catch (QueueFullException ex)
            {
                await HartsyInferenceServiceExtensions.WriteErrorAsync(ctx, StatusCodes.Status429TooManyRequests, ex.Message, "rate_limit_error");
                return;
            }
        }

        ctx.Response.Headers.ContentType = "text/event-stream";
        await foreach (string evt in events.Reader.ReadAllAsync(ct))
        {
            await ctx.Response.WriteAsync(evt, ct);
            await ctx.Response.Body.FlushAsync(ct);
        }
    }

    /// <summary>Formats one SSE frame: <c>event: &lt;name&gt;\ndata: &lt;json&gt;\n\n</c>, serialized with the app's configured JSON options so streamed payloads match non-streaming response shapes.</summary>
    public static string Event(string name, object data, JsonSerializerOptions options) =>
        $"event: {name}\ndata: {JsonSerializer.Serialize(data, options)}\n\n";

    /// <summary>The app's configured JSON options (camelCase, string enums) — shared with callers that stream SSE frames outside <see cref="RunAsync"/>'s queue-gated shape (e.g. an already-open world session).</summary>
    public static JsonSerializerOptions ResolveJsonOptions(HttpContext ctx) =>
        ctx.RequestServices.GetRequiredService<IOptions<JsonOptions>>().Value.SerializerOptions;

    internal static string ErrorEvent(string message, JsonSerializerOptions options) => Event("error", new { message }, options);
}

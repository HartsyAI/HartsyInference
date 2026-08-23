using System.Diagnostics.Metrics;
using HartsyInference.Engine;
using Microsoft.Extensions.DependencyInjection;

namespace HartsyInference.API;

/// <summary>The server's <see cref="System.Diagnostics.Metrics.Meter"/> and the counters/gauges recorded against it. Registered as a singleton and wired to <c>AddOpenTelemetry().WithMetrics(b => b.AddMeter(MeterName)...)</c> in <c>HartsyInferenceServiceExtensions.AddHartsyInference</c>; scraped in Prometheus text format at <c>GET /metrics</c>. HTTP request duration/count/in-flight come for free from <c>AddAspNetCoreInstrumentation()</c> — this class only adds domain-specific series ASP.NET Core's own instrumentation can't know about.</summary>
public sealed class ApiMetrics
{
    public const string MeterName = "HartsyInference.API";

    private readonly Meter _meter;
    private readonly Counter<long> _requestsTotal;

    public ApiMetrics(InferenceQueue fastQueue, [FromKeyedServices(QueueKeys.LongRunning)] InferenceQueue longRunningQueue)
    {
        _meter = new Meter(MeterName);

        // Named "requests_total", not "generations_total" -- this counts every request the usage-tracking
        // middleware sees (health/admin/openapi included, not just image/text/etc. generation calls), so a
        // Prometheus consumer isn't misled into thinking non-generation traffic is generation volume.
        _requestsTotal = _meter.CreateCounter<long>(
            "hartsyinference_requests_total",
            unit: "{request}",
            description: "HTTP requests handled, tagged by route group ('modality') and outcome (ok/error).");

        // Reuses InferenceQueue.PendingCount -- the same counter /admin/queue already reports -- rather than
        // tracking queue depth a second time.
        _meter.CreateObservableGauge(
            "hartsyinference_queue_pending",
            () => new[]
            {
                new Measurement<int>(fastQueue.PendingCount, new KeyValuePair<string, object?>("queue", "fast")),
                new Measurement<int>(longRunningQueue.PendingCount, new KeyValuePair<string, object?>("queue", "long_running")),
            },
            unit: "{request}",
            description: "Requests currently queued or holding a concurrency slot, by queue.");
    }

    public void RecordRequest(string modality, string outcome) =>
        _requestsTotal.Add(1,
            new KeyValuePair<string, object?>("modality", modality),
            new KeyValuePair<string, object?>("outcome", outcome));
}

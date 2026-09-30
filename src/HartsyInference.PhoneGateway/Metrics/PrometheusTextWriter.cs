using System.Globalization;
using System.Text;
using HartsyInference.Core.Numerics;

namespace HartsyInference.PhoneGateway.Metrics;

/// <summary>Renders <see cref="GatewayMetrics"/> in the Prometheus text exposition format (version 0.0.4).</summary>
public static class PrometheusTextWriter
{
    private const string Prefix = "hartsy_phone_";

    public static string Render(GatewayMetrics metrics)
    {
        ArgumentNullException.ThrowIfNull(metrics);
        StringBuilder sb = new(4096);
        Counter(sb, "calls_total", "Calls that reached the active state.", ("direction", "inbound", metrics.CallsInbound), ("direction", "outbound", metrics.CallsOutbound));
        Gauge(sb, "calls_active", "Calls in the active state.", metrics.CallsActive);
        Counter(sb, "calls_rejected_total", "INVITEs refused, by reason.",
            ("reason", "busy", metrics.CallsRejectedBusy), ("reason", "declined", metrics.CallsDeclined),
            ("reason", "host_down", metrics.CallsRejectedHostDown), ("reason", "failed", metrics.CallsFailed));
        Counter(sb, "call_seconds_total", "Seconds of active calls.", metrics.CallSecondsTotal);
        Counter(sb, "dtmf_received_total", "DTMF digits received from callers.", metrics.DtmfReceived);
        Counter(sb, "tool_requests_total", "Telephony tool requests from the host.", metrics.ToolRequests);
        Counter(sb, "tool_failures_total", "Tool requests answered Failed or Unsupported.", metrics.ToolFailures);
        Counter(sb, "outages_total", "Host outages that started during a live call.", metrics.Outages);
        Counter(sb, "outage_hangups_total", "Calls hung up because the host never came back.", metrics.OutageHangups);
        Gauge(sb, "sip_registered", "1 when registered with the registrar (or none is configured).", metrics.RegisteredProbe?.Invoke() == true ? 1 : 0);
        if (metrics.LinkProbe?.Invoke() is LinkSnapshot link)
        {
            Gauge(sb, "link_connected", "1 while the voice host link is up.", link.Connected ? 1 : 0);
            Gauge(sb, "link_outbound_rate_hz", "Host outbound sample rate from HelloAck.", link.OutboundRate);
            Counter(sb, "link_reconnects_total", "Connections established after the first.", link.Reconnects);
            Gauge(sb, "link_rtt_ms", "Round trip of the latest ping.", link.RttMs);
            Counter(sb, "link_audio_lane_dropped_total", "Inbound frames dropped from the audio lane.", link.AudioLaneDropped);
            Counter(sb, "link_inbound_dropped_down_total", "Inbound frames dropped while the link was down.", link.InboundDroppedWhileDown);
            Counter(sb, "link_stale_outbound_dropped_total", "Outbound frames dropped by the flush epoch.", link.StaleOutboundDropped);
            Counter(sb, "link_frames_sent_total", "Frames written to the host.", link.FramesSent);
            Counter(sb, "link_frames_received_total", "Frames read from the host.", link.FramesReceived);
        }
        if (metrics.MediaProbe?.Invoke() is MediaSnapshot media)
        {
            Gauge(sb, "rtp_out_frames", "RTP frames sent on the live call.", media.TickFrames);
            Gauge(sb, "rtp_out_silence_frames", "Frames padded with silence on the live call.", media.TickSilence);
            Gauge(sb, "rtp_out_catchup_frames", "Catch-up frames on the live call.", media.TickCatchUp);
            Gauge(sb, "rtp_out_resyncs", "Clock resyncs on the live call.", media.TickResyncs);
            Gauge(sb, "rtp_fifo", "1 when the tick thread runs under SCHED_FIFO.", media.TickFifo ? 1 : 0);
            Header(sb, "tick_late_us", "gauge", "Tick lateness quantiles in microseconds (bucket upper bounds).");
            Line(sb, "tick_late_us", "quantile=\"0.5\"", media.Lateness.P50Us);
            Line(sb, "tick_late_us", "quantile=\"0.99\"", media.Lateness.P99Us);
            Line(sb, "tick_late_us", "quantile=\"max\"", media.Lateness.MaxUs);
            Header(sb, "tick_late", "histogram", "Tick lateness histogram in microseconds.");
            long cumulative = 0;
            ReadOnlySpan<long> bounds = LatencyHistogram.UpperBoundsUs;
            for (int i = 0; i < bounds.Length; i++)
            {
                cumulative += media.LatenessBuckets[i];
                Line(sb, "tick_late_bucket", "le=\"" + bounds[i].ToString(CultureInfo.InvariantCulture) + "\"", cumulative);
            }
            cumulative += media.LatenessBuckets[LatencyHistogram.BucketCount - 1];
            Line(sb, "tick_late_bucket", "le=\"+Inf\"", cumulative);
            Line(sb, "tick_late_count", null, media.Lateness.Count);
            Line(sb, "tick_late_sum", null, media.Lateness.MeanUs * media.Lateness.Count);
            Gauge(sb, "rtp_in_received", "RTP packets received on the live call.", media.JitterReceived);
            Gauge(sb, "rtp_in_late", "Packets that arrived after their slot.", media.JitterLate);
            Gauge(sb, "rtp_in_lost", "Playout slots with no packet.", media.JitterLost);
            Gauge(sb, "rtp_in_duplicate", "Duplicate packets.", media.JitterDuplicate);
            Gauge(sb, "rtp_in_reordered", "Reordered packets.", media.JitterReordered);
            Gauge(sb, "rtp_in_resets", "Jitter buffer resets.", media.JitterResets);
            Gauge(sb, "rtp_in_depth_ms", "Jitter buffer depth.", media.JitterDepthMs);
            Gauge(sb, "pump_frames", "Frames the pump handed to the link.", media.PumpFrames);
            Gauge(sb, "pump_concealed_frames", "Frames filled by concealment.", media.PumpConcealed);
            Gauge(sb, "pump_dropped_by_link", "Frames the link refused (down).", media.PumpDroppedByLink);
            Gauge(sb, "rtp_out_ring_dropped_samples", "Outbound samples refused by a full ring.", media.OutboundDroppedSamples);
        }
        return sb.ToString();
    }

    private static void Counter(StringBuilder sb, string name, string help, long value)
    {
        Header(sb, name, "counter", help);
        Line(sb, name, null, value);
    }

    private static void Counter(StringBuilder sb, string name, string help, params (string Label, string Value, long Count)[] series)
    {
        Header(sb, name, "counter", help);
        foreach ((string label, string value, long count) in series)
        {
            Line(sb, name, label + "=\"" + value + "\"", count);
        }
    }

    private static void Gauge(StringBuilder sb, string name, string help, double value)
    {
        Header(sb, name, "gauge", help);
        Line(sb, name, null, value);
    }

    private static void Header(StringBuilder sb, string name, string type, string help)
    {
        sb.Append("# HELP ").Append(Prefix).Append(name).Append(' ').Append(help).Append('\n');
        sb.Append("# TYPE ").Append(Prefix).Append(name).Append(' ').Append(type).Append('\n');
    }

    private static void Line(StringBuilder sb, string name, string? labels, double value)
    {
        sb.Append(Prefix).Append(name);
        if (labels is not null)
        {
            sb.Append('{').Append(labels).Append('}');
        }
        sb.Append(' ').Append(value.ToString("R", CultureInfo.InvariantCulture)).Append('\n');
    }
}

using HartsyInference.Core.Numerics;

namespace HartsyInference.PhoneGateway.Metrics;

/// <summary>The live call's media counters at one moment, read by <c>/metrics</c>. <see cref="LatenessBuckets"/> has
/// <see cref="LatencyHistogram.BucketCount"/> entries in the histogram's bucket order.</summary>
public sealed record MediaSnapshot(
    long TickFrames, long TickSilence, long TickCatchUp, long TickResyncs, bool TickFifo,
    LatencyHistogram.Summary Lateness, long[] LatenessBuckets,
    long JitterReceived, long JitterLate, long JitterLost, long JitterDuplicate, long JitterReordered, long JitterResets, int JitterDepthMs,
    long PumpFrames, long PumpConcealed, long PumpDroppedByLink, long OutboundDroppedSamples);

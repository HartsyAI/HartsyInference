using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe.Telemetry;
using Xunit;

namespace HartsyInference.Core.Tests.Moe.Telemetry;

/// <summary>Routing telemetry: per-layer counters, frequency, the step ring, bounds checks and zero allocation.</summary>
public sealed class RoutingTelemetryTests
{
    [Fact]
    public void Record_AggregatesPerLayerCountersAndFrequency()
    {
        RoutingTelemetry telemetry = new(layerCount: 2, expertsPerLayer: 4, ringCapacity: 8);

        telemetry.Record(new ExpertKey(0, 1), hit: true, bytesMoved: 0, ranOnGpu: false);
        telemetry.Record(new ExpertKey(0, 1), hit: false, bytesMoved: 100, ranOnGpu: true);
        telemetry.Record(new ExpertKey(1, 2), hit: false, bytesMoved: 50, ranOnGpu: false);

        LayerRoutingCounters layer0 = telemetry.GetLayer(0);
        Assert.Equal(2, layer0.Routed);
        Assert.Equal(1, layer0.Hits);
        Assert.Equal(1, layer0.Misses);
        Assert.Equal(100, layer0.BytesMoved);
        Assert.Equal(1, layer0.CpuRuns);
        Assert.Equal(1, layer0.GpuRuns);
        Assert.Equal(0.5, layer0.HitRate);

        Assert.Equal(1, telemetry.GetLayer(1).Misses);
        Assert.Equal(2, telemetry.Frequency(0, 1));
        Assert.Equal(1, telemetry.Frequency(1, 2));
        Assert.Equal(0, telemetry.Frequency(0, 0));
    }

    [Fact]
    public void Ring_RetainsTheNewestRecordsInOrder()
    {
        RoutingTelemetry telemetry = new(layerCount: 1, expertsPerLayer: 8, ringCapacity: 3);
        for (int i = 0; i < 5; i++)
        {
            telemetry.BeginStep();
            telemetry.Record(new ExpertKey(0, i), hit: i % 2 == 0, bytesMoved: i, ranOnGpu: false);
        }

        Assert.Equal(3, telemetry.RingCount);
        for (int i = 0; i < 3; i++)
        {
            Assert.True(telemetry.TryGetRecent(i, out RoutingStepRecord record));
            Assert.Equal(new ExpertKey(0, i + 2), record.Key);
            Assert.Equal(i + 3, record.Step);
            Assert.Equal(i + 2, record.BytesMoved);
        }

        Assert.False(telemetry.TryGetRecent(3, out _));
    }

    [Fact]
    public void Record_RejectsOutOfRangeAccesses()
    {
        RoutingTelemetry telemetry = new(layerCount: 2, expertsPerLayer: 4, ringCapacity: 4);

        Assert.Throws<ArgumentOutOfRangeException>(() => telemetry.Record(new ExpertKey(2, 0), true, 0, false));
        Assert.Throws<ArgumentOutOfRangeException>(() => telemetry.Record(new ExpertKey(0, 4), true, 0, false));
        Assert.Throws<ArgumentOutOfRangeException>(() => telemetry.Record(new ExpertKey(0, 0), false, -1, false));
    }

    [Fact]
    public void Record_AllocatesNothingAfterConstruction()
    {
        RoutingTelemetry telemetry = new(layerCount: 4, expertsPerLayer: 16, ringCapacity: 64);
        for (int i = 0; i < 1000; i++)
        {
            telemetry.Record(new ExpertKey(i % 4, i % 16), (i & 1) == 0, i & 7, (i & 2) != 0);
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10000; i++)
        {
            telemetry.BeginStep();
            telemetry.Record(new ExpertKey(i % 4, i % 16), (i & 1) == 0, i & 7, (i & 2) != 0);
        }

        long after = GC.GetAllocatedBytesForCurrentThread();
        Assert.Equal(0L, after - before);
    }
}

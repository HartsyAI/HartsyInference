using HartsyInference.Core.Configuration;
using HartsyInference.Cuda;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The device-sized query tile (<c>numerics.tileFromDevice</c>): the same shape and device give the same tile whatever the free memory was,
/// and the knob is off unless set.</summary>
[Collection(GlobalKnobStore.Name)]
public sealed class SdpaTileFromDeviceTests
{
    private const long Heads = 12, Tokens = 32528;
    private const ulong Card24Gb = 24UL << 30, Card8Gb = 8UL << 30;

    [Fact]
    public void TheKnobIsOffUnlessSet()
    {
        // A whole-run lane that names the knob (HARTSY_GPU_TEST_KNOBS) turns it on for every test in the run, so then the knob is on by design.
        if (Environment.GetEnvironmentVariable(GpuRunKnobs.EnvVar) is { } run && run.Contains("numerics.tileFromDevice", StringComparison.Ordinal))
        {
            Assert.Equal("run profile", KnobStore.SourceOf("numerics.tileFromDevice"));
            return;
        }
        Assert.Null(EngineKnobs.TileFromDevice.Value);
    }

    [Fact]
    public void TheRowsFollowTheDeviceRungAndNotFreeMemory()
    {
        long perRow = Heads * Tokens * sizeof(float);
        Assert.Equal((1L << 30) / perRow, CudaBackend.TileRowsFromDevice(Card24Gb, Heads, Tokens, Tokens, forced: null));
        Assert.Equal((512L << 20) / perRow, CudaBackend.TileRowsFromDevice(Card8Gb, Heads, Tokens, Tokens, forced: null));
    }

    [Fact]
    public void AForcedHeightAndTheQueryRowsStillBound()
    {
        Assert.Equal(256, CudaBackend.TileRowsFromDevice(Card24Gb, Heads, Tokens, Tokens, forced: 256));
        Assert.Equal(100, CudaBackend.TileRowsFromDevice(Card24Gb, Heads, 100, Tokens, forced: null));
    }

    [Fact]
    public void TheOneGateBudgetIsHalfTheTotalOrTheFreeWhenThatIsLess()
    {
        // Ample free memory: the gate reads the device total only, so the path is the same on every run of one card.
        Assert.Equal(Card24Gb / 2, CudaBackend.DeviceGateBudget(Card24Gb, freeBytes: 20UL << 30));
        // Free below half the total: the free clamp applies (a buffer the card does not have free is not chosen).
        Assert.Equal(6UL << 30, CudaBackend.DeviceGateBudget(Card24Gb, freeBytes: 6UL << 30));
    }

    [Fact]
    public void TheGateBudgetIsNotTheTileRung()
    {
        // A 1 GiB rung on a 24 GB card: a 2 GiB score matrix is under the gate, so it stays materialized.
        Assert.Equal(1L << 30, CudaBackend.TileRungBytes(Card24Gb));
        Assert.True(2UL << 30 <= CudaBackend.DeviceGateBudget(Card24Gb, freeBytes: 20UL << 30));
    }

    [Fact]
    public void TheFreeLogCountsAHalvedCallAndNamesItsHeights()
    {
        using IDisposable isolated = TileFreeLog.Isolate();
        TileFreeLog.Record(6UL << 30, 24UL << 30, plannedRows: 100, finalRows: 50);
        TileFreeLog.Record(10UL << 30, 24UL << 30, plannedRows: 100, finalRows: 100);

        // Two calls: one halved from 100 to 50 rows, one at 100. Free ranged 6 to 10 GiB; total 24 GiB.
        Assert.Equal("tileFromDevice: calls=2 halved=1 totalMiB=24576 freeMiB min=6144 max=10240 tileRows=[50,100]", TileFreeLog.Summary());
        using (TileFreeLog.Isolate())
        {
            Assert.Equal("tileFromDevice: no tiled calls, gate decisions or executed rows this process.", TileFreeLog.Summary());
        }
    }

    [Fact]
    public void TheFreeLogNamesTheGateAndWhetherFreeSetItsBudget()
    {
        using IDisposable isolated = TileFreeLog.Isolate();
        // 24 GiB total: half is 12 GiB. Free of 6 GiB is below half, so free set the budget; free of 20 GiB is not.
        TileFreeLog.RecordGate("bf16cast", 6UL << 30, 24UL << 30, over: true);
        TileFreeLog.RecordGate("bf16cast", 20UL << 30, 24UL << 30, over: false);
        TileFreeLog.RecordGate("causal", 20UL << 30, 24UL << 30, over: true);

        Assert.Equal("tileFromDevice gate=bf16cast decisions=2 over=1 freeLimited=1\ntileFromDevice gate=causal decisions=1 over=1 freeLimited=0",
            TileFreeLog.Summary());
    }

    [Fact]
    public void TheFreeLogAppendsOneBlockPerProcessToItsFile()
    {
        using IDisposable isolated = TileFreeLog.Isolate();
        TileFreeLog.Record(6UL << 30, 24UL << 30, plannedRows: 100, finalRows: 100);
        string path = Path.Combine(Path.GetTempPath(), "tile-free-log-" + Guid.NewGuid().ToString("N") + ".log");
        try
        {
            TileFreeLog.AppendTo(path);
            string text = File.ReadAllText(path);
            Assert.Contains("# pid=", text, StringComparison.Ordinal);
            Assert.Contains("tileFromDevice: calls=1 halved=0 totalMiB=24576 freeMiB min=6144 max=6144 tileRows=[100]", text, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TheRowLinesPrintBeforeTheTileAndGateLines()
    {
        using IDisposable isolated = TileFreeLog.Isolate();
        TileFreeLog.RecordRow("tiled.f32");
        TileFreeLog.RecordRow("tiled.f32");
        TileFreeLog.Record(6UL << 30, 24UL << 30, plannedRows: 100, finalRows: 100);
        TileFreeLog.RecordGate("causal", 20UL << 30, 24UL << 30, over: false);

        string[] lines = TileFreeLog.Summary().Split('\n');
        Assert.Equal("tileFromDevice row=tiled.f32 executed=2", lines[0]);
        Assert.StartsWith("tileFromDevice: calls=1 ", lines[1], StringComparison.Ordinal);
        Assert.StartsWith("tileFromDevice gate=causal ", lines[2], StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(20UL << 30, 1L << 30)]
    [InlineData((20UL << 30) - 1, 512L << 20)]
    [InlineData(18_000_000_000UL, 512L << 20)]
    public void TheRungBoundaryIs20GiB(ulong total, long expected)
    {
        Assert.Equal(expected, CudaBackend.TileRungBytes(total));
    }
}

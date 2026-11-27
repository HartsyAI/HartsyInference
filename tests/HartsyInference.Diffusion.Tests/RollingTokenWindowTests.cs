using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Streaming;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary><see cref="RollingTokenWindow"/> bookkeeping: trimming by whole chunks behind a sink, trimming by units with a partial
/// cut of the oldest chunk that keeps its newest units, staging that only joins at commit, and refusal of malformed chunks.
/// Upstream's own softmax KV accumulation is checked in <see cref="SanaWmStreamingUpstreamTests"/>.</summary>
public sealed class RollingTokenWindowTests
{
    [Fact]
    public void RecentChunks_KeepsTheSinkAndTheNewestChunks()
    {
        using CpuBackend cpu = new();
        using RollingTokenWindow window = new(cpu, TokenWindowPolicy.RecentChunks(2, sink: true), streams: 2, tokenAxis: 2);
        for (int chunk = 0; chunk < 5; chunk++)
        {
            int units = chunk == 0 ? 4 : 3;
            window.Stage([Filled([1, 2, units * 2, 3], chunk), Filled([1, 2, units * 2, 3], 100 + chunk)], units);
            window.Commit();
        }

        Assert.Equal(2, window.PartCount);
        Assert.Equal(4, window.SinkUnits);
        Assert.Equal(3, window.HistoryUnits);
        Assert.Equal(14, window.Tokens);
        Tensor[] parts = new Tensor[2];
        Assert.Equal(2, window.CopyParts(1, parts));
        Assert.Equal(100f, parts[0].AsReadOnlySpan<float>()[0]);
        Assert.Equal(104f, parts[1].AsReadOnlySpan<float>()[0]);
    }

    [Fact]
    public void RecentUnits_CutsTheOldestChunkToItsNewestUnits()
    {
        using CpuBackend cpu = new();
        using RollingTokenWindow window = new(cpu, TokenWindowPolicy.RecentUnits(10), streams: 1, tokenAxis: 1);
        for (int chunk = 0; chunk < 6; chunk++)
        {
            // Each unit carries its global index, so the kept units are readable after the cut.
            Tensor tokens = new(new TensorShape(1, 3 * 2, 1), DType.F32);
            Span<float> values = tokens.AsSpan<float>();
            for (int i = 0; i < values.Length; i++)
                values[i] = chunk * 3 + i / 2;
            window.Stage([tokens], 3);
            window.Commit();
        }

        Assert.Equal(10, window.HistoryUnits);
        Assert.Equal(20, window.Tokens);
        Tensor[] parts = new Tensor[window.PartCount];
        int count = window.CopyParts(0, parts);
        float[] kept = [.. parts.Take(count).SelectMany(p => p.AsReadOnlySpan<float>().ToArray()).Where((_, i) => i % 2 == 0)];
        Assert.Equal(Enumerable.Range(8, 10).Select(u => (float)u).ToArray(), kept);
    }

    [Fact]
    public void StagedChunk_JoinsOnlyAtCommit_AndDiscardDropsIt()
    {
        using CpuBackend cpu = new();
        using RollingTokenWindow window = new(cpu, TokenWindowPolicy.RecentChunks(1, sink: false), streams: 1, tokenAxis: 0);
        window.Stage([Filled([4], 1)], 2);
        Assert.True(window.IsEmpty);
        window.Discard();
        window.Commit();
        Assert.True(window.IsEmpty);

        window.Stage([Filled([4], 2)], 2);
        window.Stage([Filled([6], 3)], 3);
        window.Commit();
        Assert.Equal(3, window.HistoryUnits);
        window.Reset();
        Assert.True(window.IsEmpty);
    }

    [Fact]
    public void Stage_RefusesMalformedChunks()
    {
        using CpuBackend cpu = new();
        using RollingTokenWindow window = new(cpu, TokenWindowPolicy.RecentChunks(1, sink: false), streams: 2, tokenAxis: 0);
        Assert.Throws<ArgumentException>(() => window.Stage([Filled([4], 0)], 2));
        Assert.Throws<ArgumentException>(() => window.Stage([Filled([4], 0), Filled([6], 0)], 2));
        Assert.Throws<ArgumentException>(() => window.Stage([Filled([5], 0), Filled([5], 0)], 2));
        Assert.True(window.IsEmpty);
    }

    private static Tensor Filled(long[] dims, float value)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32, DeviceKind.Cpu);
        tensor.AsSpan<float>().Fill(value);
        return tensor;
    }
}

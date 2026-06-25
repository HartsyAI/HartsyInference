using HartsyInference.Server;
using HartsyInference.Server.Imaging;
using HartsyInference.Vision.Codec;
using Xunit;

namespace HartsyInference.Server.Tests;

/// <summary>Tests for the server's self-contained pieces: the PNG encoder (round-trips through the
/// engine's PngDecoder) and the inference queue's capacity/429 behaviour.</summary>
public sealed class ServerTests
{
    [Fact]
    public void PngWriter_RoundTrips_ThroughDecoder()
    {
        const int w = 3, h = 2;
        byte[] rgb = new byte[w * h * 3];
        for (int i = 0; i < rgb.Length; i++) rgb[i] = (byte)(i * 7 % 256);

        byte[] png = PngImageWriter.Encode(rgb, w, h);

        // Valid PNG signature.
        Assert.Equal(0x89, png[0]);
        Assert.Equal((byte)'P', png[1]);

        (byte[] decoded, int dw, int dh) = PngDecoder.Decode(png);
        Assert.Equal(w, dw);
        Assert.Equal(h, dh);
        Assert.Equal(rgb, decoded);
    }

    [Fact]
    public async Task InferenceQueue_RunsWork()
    {
        using InferenceQueue queue = new InferenceQueue(maxConcurrency: 1, maxQueueDepth: 4);
        int result = await queue.EnqueueAsync(() => Task.FromResult(42), CancellationToken.None);
        Assert.Equal(42, result);
    }

    [Fact]
    public async Task InferenceQueue_RejectsWhenFull()
    {
        using InferenceQueue queue = new InferenceQueue(maxConcurrency: 1, maxQueueDepth: 0);

        // Occupy the single slot with a task that blocks until released.
        TaskCompletionSource gate = new TaskCompletionSource();
        Task<bool> running = queue.EnqueueAsync(async () => { await gate.Task; return true; }, CancellationToken.None);

        // Give the first task time to acquire the slot.
        await Task.Delay(50);

        // Capacity is concurrency(1) + depth(0) = 1; a second enqueue must be rejected.
        await Assert.ThrowsAsync<QueueFullException>(() =>
            queue.EnqueueAsync(() => Task.FromResult(true), CancellationToken.None));

        gate.SetResult();
        Assert.True(await running);
    }

    [Fact]
    public void Options_Defaults()
    {
        HartsyInferenceServerOptions o = new HartsyInferenceServerOptions();
        Assert.Equal(BackendKind.Cpu, o.Backend);
        Assert.Equal(1, o.MaxConcurrency);
        Assert.Equal(16, o.MaxQueueDepth);
    }
}

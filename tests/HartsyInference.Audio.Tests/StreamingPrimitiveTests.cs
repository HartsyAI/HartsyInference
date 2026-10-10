using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests for the streaming primitives: <see cref="AudioRingBuffer"/>,
/// <see cref="AudioStreamer"/>, <see cref="StreamingKvCache"/>. These have no model
/// weights and run instantly, so they're the cheapest part of the suite to keep
/// healthy.</summary>
public sealed class StreamingPrimitiveTests
{
    // ── AudioRingBuffer ─────────────────────────────────────────────────

    [Fact]
    public void AudioRingBuffer_WrapsAroundCleanly()
    {
        AudioRingBuffer ring = new(capacity: 8);
        ring.Write([1f, 2f, 3f, 4f, 5f]);
        float[] firstRead = new float[3];
        ring.Read(firstRead);
        Assert.Equal(new[] { 1f, 2f, 3f }, firstRead);

        // Write past the original boundary — should wrap around.
        ring.Write([6f, 7f, 8f, 9f]);
        Assert.Equal(6, ring.Available);     // 2 left from first write + 4 new

        float[] secondRead = new float[6];
        ring.Read(secondRead);
        Assert.Equal(new[] { 4f, 5f, 6f, 7f, 8f, 9f }, secondRead);
    }

    [Fact]
    public void AudioRingBuffer_OverflowDropsOldest()
    {
        AudioRingBuffer ring = new(capacity: 4);
        ring.Write([1f, 2f, 3f, 4f]);
        int dropped = ring.Write([5f, 6f]);
        Assert.Equal(2, dropped);
        Assert.Equal(2, ring.SamplesDropped);
        Assert.Equal(4, ring.Available);

        float[] output = new float[4];
        ring.Read(output);
        // 1, 2 dropped; what remains is 3, 4, 5, 6.
        Assert.Equal(new[] { 3f, 4f, 5f, 6f }, output);
    }

    [Fact]
    public void AudioRingBuffer_PeekDoesNotAdvance()
    {
        AudioRingBuffer ring = new(capacity: 8);
        ring.Write([10f, 20f, 30f]);
        float[] peek1 = new float[3];
        ring.Peek(peek1);
        float[] peek2 = new float[3];
        ring.Peek(peek2);
        Assert.Equal(peek1, peek2);
        Assert.Equal(3, ring.Available);     // Peek doesn't consume.
    }

    // ── AudioStreamer ───────────────────────────────────────────────────

    [Fact]
    public async Task AudioStreamer_ProducerConsumer_DeliversChunks()
    {
        using AudioStreamer streamer = new(capacity: 8);

        Task producer = Task.Run(async () =>
        {
            for (int i = 0; i < 3; i++)
            {
                AudioChunk chunk = new(new float[100], 24_000, 1, StartSampleOffset: i * 100);
                await streamer.Put(chunk);
            }
            streamer.Complete();
        });

        List<AudioChunk> received = new();
        await foreach (AudioChunk c in streamer.ReadAllAsync())
            received.Add(c);

        await producer;
        Assert.Equal(3, received.Count);
        Assert.Equal(0, received[0].StartSampleOffset);
        Assert.Equal(100, received[1].StartSampleOffset);
        Assert.Equal(200, received[2].StartSampleOffset);
        Assert.All(received, c =>
        {
            Assert.Equal(24_000, c.SampleRate);
            Assert.Equal(1, c.Channels);
            Assert.Equal(100, c.FrameCount);
        });
    }

    // ── AudioChunk ──────────────────────────────────────────────────────

    // ── StreamingKvCache ────────────────────────────────────────────────

    [Fact]
    public void StreamingKvCache_AppendAndAdvance_UpdatesLength()
    {
        using StreamingKvCache cache = new(
            numLayers: 2, batch: 1, numKvHeads: 2, maxSequenceLength: 32, headDim: 64);
        Assert.Equal(0, cache.CurrentLength);
        Assert.Equal(32, cache.FreeSlots);

        Tensor k = new(new TensorShape(1, 2, 4, 64), DType.F32);
        Tensor v = new(new TensorShape(1, 2, 4, 64), DType.F32);
        try
        {
            cache.Append(layerIndex: 0, k, v);
            cache.Append(layerIndex: 1, k, v);
            cache.AdvanceLength(by: 4);
            Assert.Equal(4, cache.CurrentLength);
            Assert.Equal(28, cache.FreeSlots);
        }
        finally
        {
            k.Dispose();
            v.Dispose();
        }
    }

    [Fact]
    public void StreamingKvCache_OverflowThrows()
    {
        using StreamingKvCache cache = new(
            numLayers: 1, batch: 1, numKvHeads: 1, maxSequenceLength: 4, headDim: 8);
        Tensor k = new(new TensorShape(1, 1, 5, 8), DType.F32);
        Tensor v = new(new TensorShape(1, 1, 5, 8), DType.F32);
        try
        {
            Assert.Throws<InvalidOperationException>(() => cache.Append(0, k, v));
        }
        finally
        {
            k.Dispose();
            v.Dispose();
        }
    }

    [Fact]
    public void StreamingKvCache_DtypeMismatchThrows()
    {
        using StreamingKvCache cache = new(
            numLayers: 1, batch: 1, numKvHeads: 1, maxSequenceLength: 4, headDim: 4);
        // Cache defaults to F32; appending F16 should be rejected.
        Tensor k16 = new(new TensorShape(1, 1, 1, 4), DType.F16);
        Tensor v16 = new(new TensorShape(1, 1, 1, 4), DType.F16);
        try
        {
            Assert.Throws<ArgumentException>(() => cache.Append(0, k16, v16));
        }
        finally
        {
            k16.Dispose();
            v16.Dispose();
        }
    }
}

using System.Text;
using System.Text.Json;
using HartsyInference.Engine.Audio.Wake;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>A reply that is synthesized in pieces has to arrive as one reply.
///
/// <para>The device resets its playback ring whenever a reply starts — it has to, or a cancelled reply's tail
/// plays in front of the next one. So the thing that says "a reply is starting" must be said exactly once per
/// reply, no matter how many pieces the server produced it in. Sending each sentence as its own reply is the
/// failure this guards: every sentence after the first would reset the ring and cut off the one before it, and
/// a three-sentence answer would be heard as most of the last sentence.</para></summary>
public sealed class WakeAudioStreamTests
{
    private const int Rate = 16000;

    private static byte[] Tone(int bytes)
    {
        byte[] pcm = new byte[bytes];
        for (int i = 0; i < bytes; i++)
        {
            pcm[i] = (byte)(i * 7 % 251);
        }
        return pcm;
    }

    /// <summary>Walks the raw bytes the way the satellite does — header line, then exactly the payload it
    /// declares — rather than through the codec's reader, because what is being tested is the header the
    /// device parses by hand, and the reader does not surface <c>final</c>.</summary>
    private static List<(JsonElement Data, byte[] Payload)> ReadBack(MemoryStream raw)
    {
        byte[] all = raw.ToArray();
        List<(JsonElement, byte[])> frames = [];
        int at = 0;
        while (at < all.Length)
        {
            int nl = Array.IndexOf(all, (byte)'\n', at);
            Assert.True(nl >= 0, "a header with no newline — the stream is desynced");
            JsonElement header = JsonDocument.Parse(Encoding.UTF8.GetString(all, at, nl - at)).RootElement;
            Assert.Equal("audio", header.GetProperty("type").GetString());
            int length = header.GetProperty("payload_length").GetInt32();
            at = nl + 1;
            Assert.True(at + length <= all.Length, "a header promised bytes that are not there");
            frames.Add((header.GetProperty("data"), all[at..(at + length)]));
            at += length;
        }
        return frames;
    }

    [Fact]
    public async Task ThreeSentences_AreOneReply()
    {
        using MemoryStream raw = new();
        // leadMs 0: this is about numbering, not pacing, and a test should not sleep for it.
        WakeAudioStream stream = new(new WakeFrameCodec(raw), "pico-1", Rate, leadMs: 0);
        byte[] a = Tone(1280), b = Tone(2560), c = Tone(640);
        await stream.WriteAsync(a);
        await stream.WriteAsync(b);
        await stream.WriteAsync(c);
        await stream.CompleteAsync();

        List<(JsonElement Data, byte[] Payload)> frames = ReadBack(raw);

        // One continuous numbering across all three pieces — a restart at zero is what the device reads as a
        // new reply.
        Assert.Equal(Enumerable.Range(0, frames.Count), frames.Select(f => f.Data.GetProperty("seq").GetInt32()));
        // Exactly one final frame, and it is the last one.
        Assert.Single(frames, f => f.Data.GetProperty("final").GetBoolean());
        Assert.True(frames[^1].Data.GetProperty("final").GetBoolean());
        // Every byte, in order, once.
        Assert.Equal([.. a, .. b, .. c], frames.SelectMany(f => f.Payload).ToArray());
        Assert.Equal(a.Length + b.Length + c.Length, stream.BytesSent);
    }

    [Fact]
    public async Task PiecesThatDoNotDivideEvenly_NeverSendAnOddFrame()
    {
        // A short frame is legal, but an odd byte count would put half a sample into the device's ring and
        // every sample after it would be assembled from two different ones.
        using MemoryStream raw = new();
        WakeAudioStream stream = new(new WakeFrameCodec(raw), "pico-1", Rate, leadMs: 0);
        await stream.WriteAsync(Tone(101));
        await stream.WriteAsync(Tone(37));
        await stream.WriteAsync(Tone(1500));
        await stream.CompleteAsync();

        List<(JsonElement Data, byte[] Payload)> frames = ReadBack(raw);
        foreach ((JsonElement data, byte[] payload) in frames.Take(frames.Count - 1))
        {
            Assert.Equal(Rate / 25 * 2, payload.Length);
        }
        Assert.Equal(101 + 37 + 1500, frames.Sum(f => f.Payload.Length));
    }

    [Fact]
    public async Task TheReplyIsAlwaysClosed_EvenWithNothingLeftToSay()
    {
        // The final frame is what tells the device the turn is over. Skipping it because the tail happened to
        // be empty leaves the device holding the turn until its own watchdog gives up on it.
        using MemoryStream raw = new();
        WakeAudioStream stream = new(new WakeFrameCodec(raw), "pico-1", Rate, leadMs: 0);
        await stream.WriteAsync(Tone(1280));
        await stream.CompleteAsync();

        List<(JsonElement Data, byte[] Payload)> frames = ReadBack(raw);
        Assert.True(frames[^1].Data.GetProperty("final").GetBoolean());
        Assert.Empty(frames[^1].Payload);
        Assert.True(stream.IsComplete);
    }

    [Fact]
    public async Task AnAbandonedReply_IsClosedOnTheWayOut()
    {
        // Barge-in: the turn is cancelled between sentences. Nobody calls CompleteAsync, and the device would
        // be left waiting on a reply that never ends.
        using MemoryStream raw = new();
        await using (WakeAudioStream stream = new(new WakeFrameCodec(raw), "pico-1", Rate, leadMs: 0))
        {
            await stream.WriteAsync(Tone(1280));
        }
        List<(JsonElement Data, byte[] Payload)> frames = ReadBack(raw);
        Assert.True(frames[^1].Data.GetProperty("final").GetBoolean());
    }

    [Fact]
    public async Task WritingAfterTheEnd_IsRefused()
    {
        using MemoryStream raw = new();
        WakeAudioStream stream = new(new WakeFrameCodec(raw), "pico-1", Rate, leadMs: 0);
        await stream.CompleteAsync();
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await stream.WriteAsync(Tone(64)));
    }
}

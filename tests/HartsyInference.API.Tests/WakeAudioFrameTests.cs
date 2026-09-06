using System.Text;
using HartsyInference.Engine.Audio.Wake;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Frames that carry bytes after their header, written by the server for once.
///
/// <para>The satellite has sent audio this way since the protocol was written; this is the same shape in the
/// other direction, so that spoken replies can travel on the socket the device already holds open instead of
/// costing it a second connection and a second protocol.</para>
///
/// <para>The failure this guards against is specific and has happened before, in the other direction: a header
/// promising bytes that do not arrive desynchronizes the stream permanently, because the reader then takes the
/// next frame's header as payload and everything after it is garbage. So the round trip is tested rather than
/// the bytes alone.</para></summary>
public sealed class WakeAudioFrameTests
{
    [Fact]
    public async Task AFrameWithAPayload_ReadsBackExactly()
    {
        byte[] pcm = new byte[2560];
        for (int i = 0; i < pcm.Length; i++)
        {
            pcm[i] = (byte)(i * 7 % 251);
        }

        using MemoryStream stream = new();
        WakeFrameCodec writer = new(stream);
        await writer.WriteAsync("audio", "{\"seq\":0,\"final\":false}", pcm, CancellationToken.None);

        stream.Position = 0;
        WakeFrameCodec reader = new(stream);
        WakeFrame? maybe = await reader.ReadAsync(CancellationToken.None);

        WakeFrame frame = Assert.IsType<WakeFrame>(maybe);
        Assert.Equal("audio", frame.Type);
        Assert.Equal(pcm.Length, frame.PayloadLength);
        Assert.Equal(pcm, frame.Payload);
    }

    [Fact]
    public async Task BackToBackFrames_DoNotBleedIntoEachOther()
    {
        // The desync failure only shows up on the second frame: if the payload length is wrong by even one
        // byte, the reader takes part of the next header as audio and never recovers.
        byte[] first = Encoding.UTF8.GetBytes("first payload, deliberately an odd length");
        byte[] second = Encoding.UTF8.GetBytes("second");

        using MemoryStream stream = new();
        WakeFrameCodec writer = new(stream);
        await writer.WriteAsync("audio", "{\"seq\":0,\"final\":false}", first, CancellationToken.None);
        await writer.WriteAsync("audio", "{\"seq\":1,\"final\":true}", second, CancellationToken.None);
        await writer.WriteAsync("status", WakeStatus.Data(WakeStatus.Done), CancellationToken.None);

        stream.Position = 0;
        WakeFrameCodec reader = new(stream);
        WakeFrame a = Assert.IsType<WakeFrame>(await reader.ReadAsync(CancellationToken.None));
        WakeFrame b = Assert.IsType<WakeFrame>(await reader.ReadAsync(CancellationToken.None));
        WakeFrame c = Assert.IsType<WakeFrame>(await reader.ReadAsync(CancellationToken.None));

        Assert.Equal(first, a.Payload);
        Assert.Equal(second, b.Payload);
        Assert.Equal("status", c.Type);
        Assert.Null(c.Payload);
        Assert.Null(await reader.ReadAsync(CancellationToken.None));
    }

    [Fact]
    public async Task AnEmptyPayload_StillDeclaresItsLength()
    {
        using MemoryStream stream = new();
        WakeFrameCodec writer = new(stream);
        await writer.WriteAsync("audio", "{\"seq\":9,\"final\":true}", ReadOnlyMemory<byte>.Empty,
            CancellationToken.None);

        string header = Encoding.UTF8.GetString(stream.ToArray());
        Assert.Equal("{\"type\":\"audio\",\"data\":{\"seq\":9,\"final\":true},\"payload_length\":0}\n", header);
    }
}

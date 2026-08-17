using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using HartsyInference.Engine.Audio.Wake.Wyoming;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The Wyoming compatibility endpoint's silent-failure surface.
///
/// <para>Three things here break without an error anywhere: framing, because the reference library writes
/// <c>data</c> as a separate length-prefixed block and a reader that assumes inline <c>data</c> desyncs on the
/// first request instead of throwing; segment boundaries, because a codec that only works when a frame arrives in
/// one TCP read passes every local test and fails on a real network; and the <c>info</c> manifest, because Wyoming
/// drops unknown keys but refuses a service missing a required one, so Home Assistant just lists nothing.</para>
///
/// <para>No weights and no engine — <c>describe</c> and <c>ping</c> are answered before any model is touched.</para></summary>
public sealed class WyomingProtocolTests
{
    [Fact]
    public async Task Describe_ProducesInfoWithTheKeysHomeAssistantRequires()
    {
        using WyomingListener listener = new(null, new WyomingOptions { Port = 0 });
        listener.Start();

        using WyomingEvent info = await RoundTripAsync(listener.Port, "describe", null);
        Assert.Equal("info", info.Type);
        JsonElement data = info.Data;

        foreach (string domain in new[] { "asr", "tts", "wake", "handle", "intent", "mic", "snd" })
            Assert.Equal(JsonValueKind.Array, data.GetProperty(domain).ValueKind);

        JsonElement asr = Assert.Single(data.GetProperty("asr").EnumerateArray().ToArray());
        AssertArtifact(asr);
        JsonElement asrModel = Assert.Single(asr.GetProperty("models").EnumerateArray().ToArray());
        AssertArtifact(asrModel);
        Assert.Equal("whisper", asrModel.GetProperty("name").GetString());
        Assert.Contains("en", asrModel.GetProperty("languages").EnumerateArray().Select(static l => l.GetString()));

        JsonElement tts = Assert.Single(data.GetProperty("tts").EnumerateArray().ToArray());
        AssertArtifact(tts);
        JsonElement voice = Assert.Single(tts.GetProperty("voices").EnumerateArray().ToArray());
        AssertArtifact(voice);
        Assert.Equal("kokoro", voice.GetProperty("name").GetString());
        Assert.Equal(JsonValueKind.Array, voice.GetProperty("languages").ValueKind);

        // Wake is advertised only when a detector is wired; a listed service that can never fire is worse than
        // one Home Assistant never sees.
        Assert.Empty(data.GetProperty("wake").EnumerateArray().ToArray());
    }

    [Fact]
    public async Task Describe_AdvertisesWakeOnlyWithADetectorWired()
    {
        WyomingOptions options = new()
        {
            Port = 0,
            WakeModels = [new WyomingArtifact { Name = "hey_jarvis", Phrase = "hey jarvis" }],
            WakeDetectorFactory = static _ => new SilentDetector(),
        };
        using WyomingListener listener = new(null, options);
        listener.Start();

        using WyomingEvent info = await RoundTripAsync(listener.Port, "describe", null);
        JsonElement wake = Assert.Single(info.Data.GetProperty("wake").EnumerateArray().ToArray());
        AssertArtifact(wake);
        JsonElement model = Assert.Single(wake.GetProperty("models").EnumerateArray().ToArray());
        AssertArtifact(model);
        Assert.Equal("hey_jarvis", model.GetProperty("name").GetString());
        Assert.Equal("hey jarvis", model.GetProperty("phrase").GetString());
    }

    [Fact]
    public async Task Ping_IsAnsweredWithPongEchoingItsText()
    {
        using WyomingListener listener = new(null, new WyomingOptions { Port = 0 });
        listener.Start();

        using WyomingEvent pong = await RoundTripAsync(listener.Port, "ping", "{\"text\":\"are you there\"}");
        Assert.Equal("pong", pong.Type);
        Assert.Equal("are you there", pong.GetString("text"));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(64)]
    public async Task Codec_ReassemblesFramesSplitAtAnySegmentBoundary(int bytesPerRead)
    {
        byte[] payload = new byte[640];
        for (int i = 0; i < payload.Length; i++) payload[i] = (byte)(i * 7);

        using MemoryStream wire = new();
        WriteReferenceFrame(wire, "transcribe", "{\"name\":\"whisper\",\"language\":\"en\"}", []);
        WriteReferenceFrame(wire, "audio-chunk", "{\"rate\":16000,\"width\":2,\"channels\":1}", payload);
        WriteReferenceFrame(wire, "audio-stop", null, []);

        // A stream that hands back a few bytes at a time splits every frame mid-header, mid-data-block and
        // mid-payload across the run, which is what a real socket does under load.
        using DribbleStream dribble = new(wire.ToArray(), bytesPerRead);
        WyomingFrameCodec codec = new(dribble);

        using (WyomingEvent? first = await codec.ReadAsync(CancellationToken.None))
        {
            Assert.NotNull(first);
            Assert.Equal("transcribe", first.Type);
            Assert.Equal("whisper", first.GetString("name"));
            Assert.Equal("en", first.GetString("language"));
        }
        using (WyomingEvent? second = await codec.ReadAsync(CancellationToken.None))
        {
            Assert.NotNull(second);
            Assert.Equal("audio-chunk", second.Type);
            Assert.Equal(16_000, second.GetInt32("rate"));
            Assert.Equal(payload.Length, second.PayloadLength);
            Assert.Equal(payload, second.Payload);
        }
        using (WyomingEvent? third = await codec.ReadAsync(CancellationToken.None))
        {
            Assert.NotNull(third);
            Assert.Equal("audio-stop", third.Type);
        }
        Assert.Null(await codec.ReadAsync(CancellationToken.None));
    }

    [Fact]
    public async Task Codec_AcceptsInlineDataAndLetsTheBlockWin()
    {
        using MemoryStream wire = new();
        // The satellite protocol writes `data` inline; the reference reader merges an out-of-line block on top
        // of it, so a peer that sends both must see the block's value.
        byte[] block = Encoding.UTF8.GetBytes("{\"language\":\"fr\"}");
        byte[] header = Encoding.UTF8.GetBytes(
            $"{{\"type\":\"transcribe\",\"data\":{{\"name\":\"whisper\",\"language\":\"en\"}},\"data_length\":{block.Length}}}\n");
        wire.Write(header);
        wire.Write(block);
        wire.Position = 0;

        WyomingFrameCodec codec = new(wire);
        using WyomingEvent? evt = await codec.ReadAsync(CancellationToken.None);
        Assert.NotNull(evt);
        Assert.Equal("whisper", evt.GetString("name"));
        Assert.Equal("fr", evt.GetString("language"));
    }

    [Fact]
    public async Task Writer_EmitsDataOutOfLineLikeTheReferenceLibrary()
    {
        using MemoryStream wire = new();
        WyomingFrameCodec codec = new(wire);
        await codec.WriteAsync("transcript", WyomingFrameCodec.BuildData(static writer => writer.WriteString("text", "hello")), CancellationToken.None);

        byte[] bytes = wire.ToArray();
        int newline = Array.IndexOf(bytes, (byte)'\n');
        Assert.True(newline > 0);
        using JsonDocument header = JsonDocument.Parse(bytes.AsMemory(0, newline));
        Assert.Equal("transcript", header.RootElement.GetProperty("type").GetString());
        Assert.Equal(WyomingFrameCodec.ProtocolVersion, header.RootElement.GetProperty("version").GetString());
        // Inline `data` in the header is exactly what the reference reader does NOT expect from a writer; if this
        // ever flips back, Home Assistant still parses it but the two sides stop agreeing on the wire shape.
        Assert.False(header.RootElement.TryGetProperty("data", out _));
        int dataLength = header.RootElement.GetProperty("data_length").GetInt32();
        Assert.Equal(bytes.Length - newline - 1, dataLength);
        Assert.Equal("{\"text\":\"hello\"}", Encoding.UTF8.GetString(bytes, newline + 1, dataLength));
    }

    private static void AssertArtifact(JsonElement artifact)
    {
        // Wyoming fills only its Optional fields in from a missing key, so every one of these must be present or
        // the whole manifest fails to decode and the service silently never appears.
        Assert.Equal(JsonValueKind.String, artifact.GetProperty("name").ValueKind);
        Assert.True(artifact.GetProperty("installed").GetBoolean());
        Assert.True(artifact.TryGetProperty("description", out _));
        Assert.True(artifact.TryGetProperty("version", out _));
        JsonElement attribution = artifact.GetProperty("attribution");
        Assert.False(string.IsNullOrEmpty(attribution.GetProperty("name").GetString()));
        Assert.False(string.IsNullOrEmpty(attribution.GetProperty("url").GetString()));
    }

    private static async Task<WyomingEvent> RoundTripAsync(int port, string type, string? dataJson)
    {
        using TcpClient client = new();
        await client.ConnectAsync("127.0.0.1", port);
        using NetworkStream stream = client.GetStream();
        WyomingFrameCodec codec = new(stream);
        await codec.WriteAsync(type, dataJson is null ? ReadOnlyMemory<byte>.Empty : Encoding.UTF8.GetBytes(dataJson), CancellationToken.None);
        WyomingEvent? reply = await codec.ReadAsync(CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(15));
        Assert.NotNull(reply);
        return reply;
    }

    /// <summary>Writes a frame the way the reference library's writer does: header line, out-of-line data block,
    /// then the payload.</summary>
    private static void WriteReferenceFrame(Stream stream, string type, string? dataJson, ReadOnlySpan<byte> payload)
    {
        byte[] data = dataJson is null ? [] : Encoding.UTF8.GetBytes(dataJson);
        StringBuilder header = new($"{{\"type\":\"{type}\",\"version\":\"1.10.0\"");
        if (data.Length > 0) header.Append(",\"data_length\":").Append(data.Length);
        if (payload.Length > 0) header.Append(",\"payload_length\":").Append(payload.Length);
        header.Append("}\n");
        stream.Write(Encoding.UTF8.GetBytes(header.ToString()));
        stream.Write(data);
        stream.Write(payload);
    }

    /// <summary>A read-only stream that never returns more than <c>bytesPerRead</c> bytes at a time.</summary>
    private sealed class DribbleStream(byte[] data, int bytesPerRead) : Stream
    {
        private int _position;

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => data.Length;
        public override long Position { get => _position; set => throw new NotSupportedException(); }

        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

        public override int Read(Span<byte> buffer)
        {
            int available = Math.Min(Math.Min(bytesPerRead, buffer.Length), data.Length - _position);
            if (available <= 0) return 0;
            data.AsSpan(_position, available).CopyTo(buffer);
            _position += available;
            return available;
        }

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) =>
            ValueTask.FromResult(Read(buffer.Span));

        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    /// <summary>A detector that never fires, so the manifest can advertise wake without any weights on disk.</summary>
    private sealed class SilentDetector : IWyomingWakeDetector
    {
        public WyomingWakeHit? Push(ReadOnlySpan<float> samples) => null;

        public void Reset() { }

        public void Dispose() { }
    }
}

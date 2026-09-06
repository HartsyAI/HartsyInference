using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.Engine.Audio.Wake;

/// <summary>One protocol frame: a small JSON header and an optional binary payload.
///
/// <para>The framing is Wyoming's (Home Assistant / Rhasspy) — a single UTF-8 JSON line terminated by
/// <c>\n</c>, optionally followed by exactly <c>payload_length</c> raw bytes:</para>
/// <code>
/// {"type":"audio-chunk","data":{"seq":41},"payload_length":2560}\n
/// &lt;2560 bytes of little-endian signed 16-bit mono PCM&gt;
/// </code>
///
/// <para>It is used here because it is trivial to implement on a microcontroller (one line, then a byte
/// count — no handshake, no frame masking, no varint protobuf), and because speaking it already means a
/// Home Assistant compatibility endpoint is mostly a matter of mapping event names.</para>
///
/// <para>Only the fields this engine needs are parsed, so unknown fields from a newer or third-party client
/// are ignored rather than rejected.</para></summary>
public readonly record struct WakeFrame(string Type, WakeFrameData Data, byte[]? Payload, int PayloadLength)
{
    /// <summary>Payload interpreted as 16-bit PCM samples, widened to the int16-scaled floats the wake models expect (±32768, NOT normalized to ±1 — normalizing silently mis-scores).</summary>
    public int ReadPcm(Span<float> destination)
    {
        if (Payload is null || PayloadLength == 0) return 0;
        int samples = PayloadLength / 2;
        if (destination.Length < samples)
            throw new ArgumentException($"PCM destination holds {destination.Length} samples, frame carries {samples}.", nameof(destination));
        ReadOnlySpan<byte> bytes = Payload.AsSpan(0, samples * 2);
        for (int i = 0; i < samples; i++)
            destination[i] = (short)(bytes[i * 2] | (bytes[i * 2 + 1] << 8));
        return samples;
    }
}

/// <summary>The header fields the engine reads. Everything is optional: which ones are meaningful depends on <see cref="WakeFrame.Type"/>.</summary>
public readonly record struct WakeFrameData(string? DeviceId, long Sequence, int Rate, int Width, int Channels, string? Text, string? Name, string? Token);

/// <summary>Reads and writes <see cref="WakeFrame"/>s on a stream. One instance per connection; not thread-safe for concurrent reads, and writes are serialized internally so the worker thread and the connection loop can both send.</summary>
public sealed class WakeFrameCodec(Stream stream, int maxPayloadBytes = 1 << 20)
{
    private const byte Newline = (byte)'\n';
    private const int MaxHeaderBytes = 8 * 1024;

    private readonly Stream _stream = stream ?? throw new ArgumentNullException(nameof(stream));
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private readonly byte[] _headerBuffer = new byte[MaxHeaderBytes];
    private int _headerLength;

    /// <summary>Reads the next frame, or null at end of stream.</summary>
    public async Task<WakeFrame?> ReadAsync(CancellationToken cancel)
    {
        int lineEnd = await ReadHeaderLineAsync(cancel).ConfigureAwait(false);
        if (lineEnd < 0) return null;

        (string type, WakeFrameData data, int payloadLength) = ParseHeader(_headerBuffer.AsSpan(0, lineEnd));
        if (payloadLength < 0 || payloadLength > maxPayloadBytes)
            throw new HartsyInferenceException($"Wake frame declares a {payloadLength}-byte payload; the limit is {maxPayloadBytes}.");

        // Carry over whatever of the next frame already landed in the header buffer.
        int carriedOver = _headerLength - (lineEnd + 1);
        byte[]? payload = null;
        if (payloadLength > 0)
        {
            payload = new byte[payloadLength];
            int fromBuffer = Math.Min(carriedOver, payloadLength);
            if (fromBuffer > 0)
            {
                Array.Copy(_headerBuffer, lineEnd + 1, payload, 0, fromBuffer);
                Array.Copy(_headerBuffer, lineEnd + 1 + fromBuffer, _headerBuffer, 0, carriedOver - fromBuffer);
                _headerLength = carriedOver - fromBuffer;
            }
            else
            {
                Array.Copy(_headerBuffer, lineEnd + 1, _headerBuffer, 0, carriedOver);
                _headerLength = carriedOver;
            }
            int remaining = payloadLength - fromBuffer;
            if (remaining > 0)
                await _stream.ReadExactlyAsync(payload.AsMemory(fromBuffer, remaining), cancel).ConfigureAwait(false);
        }
        else
        {
            Array.Copy(_headerBuffer, lineEnd + 1, _headerBuffer, 0, carriedOver);
            _headerLength = carriedOver;
        }

        return new WakeFrame(type, data, payload, payloadLength);
    }

    private async Task<int> ReadHeaderLineAsync(CancellationToken cancel)
    {
        while (true)
        {
            int newline = Array.IndexOf(_headerBuffer, Newline, 0, _headerLength);
            if (newline >= 0) return newline;
            if (_headerLength == _headerBuffer.Length)
                throw new HartsyInferenceException($"Wake frame header exceeded {MaxHeaderBytes} bytes with no newline; the peer is not speaking this protocol.");

            int read = await _stream.ReadAsync(_headerBuffer.AsMemory(_headerLength), cancel).ConfigureAwait(false);
            if (read == 0) return -1;
            _headerLength += read;
        }
    }

    private static (string Type, WakeFrameData Data, int PayloadLength) ParseHeader(ReadOnlySpan<byte> json)
    {
        string type = "";
        int payloadLength = 0;
        string? deviceId = null, text = null, name = null, token = null;
        long sequence = 0;
        int rate = 0, width = 0, channels = 0;

        Utf8JsonReader reader = new(json);
        int depth = 0;
        string property = "";
        bool inData = false;
        while (reader.Read())
        {
            switch (reader.TokenType)
            {
                case JsonTokenType.StartObject:
                    depth++;
                    if (depth == 2 && property == "data") inData = true;
                    break;
                case JsonTokenType.EndObject:
                    if (depth == 2 && inData) inData = false;
                    depth--;
                    break;
                case JsonTokenType.PropertyName:
                    property = reader.GetString() ?? "";
                    break;
                case JsonTokenType.String:
                    if (depth == 1 && property == "type") type = reader.GetString() ?? "";
                    else if (inData)
                    {
                        if (property is "device_id") deviceId = reader.GetString();
                        else if (property == "text") text = reader.GetString();
                        else if (property == "name") name = reader.GetString();
                        else if (property == "token") token = reader.GetString();
                    }
                    break;
                case JsonTokenType.Number:
                    if (depth == 1 && property == "payload_length") payloadLength = reader.GetInt32();
                    else if (inData)
                    {
                        if (property is "seq" or "sequence") sequence = reader.GetInt64();
                        else if (property == "rate") rate = reader.GetInt32();
                        else if (property == "width") width = reader.GetInt32();
                        else if (property == "channels") channels = reader.GetInt32();
                    }
                    break;
            }
        }

        if (type.Length == 0)
            throw new HartsyInferenceException("Wake frame header has no 'type'.");
        return (type, new WakeFrameData(deviceId, sequence, rate, width, channels, text, name, token), payloadLength);
    }

    /// <summary>Writes a header-only frame whose <c>data</c> object is <paramref name="dataJson"/> (already serialized, or null for none).</summary>
    public async Task WriteAsync(string type, string? dataJson, CancellationToken cancel)
    {
        StringBuilder sb = new();
        sb.Append("{\"type\":\"").Append(type).Append('"');
        if (dataJson is not null) sb.Append(",\"data\":").Append(dataJson);
        sb.Append("}\n");
        byte[] bytes = Encoding.UTF8.GetBytes(sb.ToString());

        await _writeLock.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            await _stream.WriteAsync(bytes, cancel).ConfigureAwait(false);
            await _stream.FlushAsync(cancel).ConfigureAwait(false);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    /// <summary>Writes a frame carrying raw bytes after its header — the same shape a satellite already sends
    /// audio in, now in the other direction.
    ///
    /// <para>Speech reaches the device over HTTP today, which costs it a second connection per turn and a
    /// second protocol to keep working. The socket the device is already holding open can carry it, and the
    /// wire format needs nothing new: a header line with <c>payload_length</c>, then that many bytes. The
    /// reader on the far side has parsed exactly this since the beginning.</para>
    ///
    /// <para>Header and payload go out under one lock and in one pair of writes, because a header promising
    /// bytes that never arrive desynchronizes the stream permanently — the reader takes the next frame's header
    /// as payload and every frame after it is garbage. That failure has been seen on this protocol in the other
    /// direction and it cost a night to find.</para></summary>
    public async Task WriteAsync(string type, string? dataJson, ReadOnlyMemory<byte> payload, CancellationToken cancel)
    {
        StringBuilder sb = new();
        sb.Append("{\"type\":\"").Append(type).Append('"');
        if (dataJson is not null) sb.Append(",\"data\":").Append(dataJson);
        sb.Append(",\"payload_length\":").Append(payload.Length.ToString(CultureInfo.InvariantCulture));
        sb.Append("}\n");
        byte[] header = Encoding.UTF8.GetBytes(sb.ToString());

        await _writeLock.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            await _stream.WriteAsync(header, cancel).ConfigureAwait(false);
            if (payload.Length > 0)
            {
                await _stream.WriteAsync(payload, cancel).ConfigureAwait(false);
            }
            await _stream.FlushAsync(cancel).ConfigureAwait(false);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    /// <summary>Escapes a string for embedding in a JSON header value.</summary>
    public static string Escape(string value)
    {
        ArrayBufferWriter<byte> buffer = new();
        using (Utf8JsonWriter writer = new(buffer))
        {
            writer.WriteStringValue(value);
        }
        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }
}

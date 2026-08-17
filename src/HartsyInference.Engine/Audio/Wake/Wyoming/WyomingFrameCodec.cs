using System.Buffers;
using System.Text.Json;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.Engine.Audio.Wake.Wyoming;

/// <summary>Reads and writes real Wyoming frames. One instance per connection; reads are single-threaded, writes
/// are serialized internally so a synthesis loop and the request loop can both send.
///
/// <para>This is deliberately NOT <see cref="WakeFrameCodec"/>. The satellite protocol uses the <em>subset</em> of
/// Wyoming where <c>data</c> is inline in the header line, but the reference library's writer — the one Home
/// Assistant uses — pops <c>data</c> out of the header and emits it as a separate length-prefixed block:</para>
/// <code>
/// {"type":"transcribe","version":"1.10.0","data_length":31,"payload_length":0}\n
/// {"name":"whisper","language":"en"}
/// &lt;payload_length bytes&gt;
/// </code>
/// <para>A reader that only understands inline <c>data</c> would treat that block as the start of the next frame
/// and desync on the first request Home Assistant sends. <see cref="WakeFrameCodec"/> also cannot write a frame
/// with a payload at all, which Wyoming text-to-speech needs. Both halves are handled here; the reference reader
/// accepts inline <c>data</c>, an out-of-line block, or both (block wins per key), so all three are supported.</para></summary>
public sealed class WyomingFrameCodec(Stream stream, int maxPayloadBytes = 1 << 20)
{
    /// <summary>Reference-library version advertised in every header we write. Peers ignore it; it is emitted
    /// because the reference writer always does, and an absent key is a cheap way to look like a stranger.</summary>
    public const string ProtocolVersion = "1.10.0";

    private const byte Newline = (byte)'\n';
    private const int MaxHeaderBytes = 8 * 1024;

    private static readonly byte[] NewlineBytes = [Newline];

    private readonly Stream _stream = stream ?? throw new ArgumentNullException(nameof(stream));
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private readonly byte[] _buffer = new byte[MaxHeaderBytes];
    private int _buffered;

    /// <summary>Reads the next event, or null at end of stream.</summary>
    public async Task<WyomingEvent?> ReadAsync(CancellationToken cancel)
    {
        int lineEnd = await ReadLineAsync(cancel).ConfigureAwait(false);
        if (lineEnd < 0) return null;

        // Copied out before consuming: JsonDocument keeps referencing the memory it parsed, and the read
        // buffer is compacted in place.
        byte[] headerBytes = _buffer.AsSpan(0, lineEnd).ToArray();
        Consume(lineEnd + 1);

        string type;
        int dataLength, payloadLength;
        JsonDocument? data = null;
        byte[]? payload = null;
        using (JsonDocument header = ParseHeader(headerBytes))
        {
            (type, dataLength, payloadLength) = ReadHeaderFields(header);
            // Bounded by the payload limit, not by the 8 KB header line: the block is read into its own array,
            // and a `synthesize` carrying a long assistant reply routinely exceeds a header's worth of text.
            if (dataLength < 0 || dataLength > maxPayloadBytes)
                throw new HartsyInferenceException($"Wyoming frame declares a {dataLength}-byte data block; the limit is {maxPayloadBytes}.");
            if (payloadLength < 0 || payloadLength > maxPayloadBytes)
                throw new HartsyInferenceException($"Wyoming frame declares a {payloadLength}-byte payload; the limit is {maxPayloadBytes}.");

            JsonDocument? block = null;
            if (dataLength > 0)
            {
                byte[] dataBytes = new byte[dataLength];
                await FillAsync(dataBytes, dataLength, cancel).ConfigureAwait(false);
                block = JsonDocument.Parse(dataBytes);
            }
            header.RootElement.TryGetProperty("data", out JsonElement inline);
            data = MergeData(inline, block);
        }

        if (payloadLength > 0)
        {
            payload = new byte[payloadLength];
            await FillAsync(payload, payloadLength, cancel).ConfigureAwait(false);
        }
        return new WyomingEvent(type, data, payload, payloadLength);
    }

    /// <summary>Writes one event: header line, then the <c>data</c> block, then the payload.</summary>
    public async Task WriteAsync(string type, ReadOnlyMemory<byte> dataJson, ReadOnlyMemory<byte> payload, CancellationToken cancel)
    {
        ArrayBufferWriter<byte> header = new(128);
        using (Utf8JsonWriter writer = new(header))
        {
            writer.WriteStartObject();
            writer.WriteString("type", type);
            writer.WriteString("version", ProtocolVersion);
            if (dataJson.Length > 0) writer.WriteNumber("data_length", dataJson.Length);
            if (payload.Length > 0) writer.WriteNumber("payload_length", payload.Length);
            writer.WriteEndObject();
        }

        await _writeLock.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            await _stream.WriteAsync(header.WrittenMemory, cancel).ConfigureAwait(false);
            await _stream.WriteAsync(NewlineBytes, cancel).ConfigureAwait(false);
            if (dataJson.Length > 0) await _stream.WriteAsync(dataJson, cancel).ConfigureAwait(false);
            if (payload.Length > 0) await _stream.WriteAsync(payload, cancel).ConfigureAwait(false);
            await _stream.FlushAsync(cancel).ConfigureAwait(false);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    /// <summary>Writes an event with no binary payload.</summary>
    public Task WriteAsync(string type, ReadOnlyMemory<byte> dataJson, CancellationToken cancel) =>
        WriteAsync(type, dataJson, ReadOnlyMemory<byte>.Empty, cancel);

    /// <summary>Serializes a <c>data</c> object; <paramref name="write"/> emits the properties, the braces are
    /// supplied here.</summary>
    public static byte[] BuildData(Action<Utf8JsonWriter> write)
    {
        ArrayBufferWriter<byte> buffer = new(128);
        using (Utf8JsonWriter writer = new(buffer))
        {
            writer.WriteStartObject();
            write(writer);
            writer.WriteEndObject();
        }
        return buffer.WrittenSpan.ToArray();
    }

    private static JsonDocument ParseHeader(ReadOnlyMemory<byte> json)
    {
        try
        {
            return JsonDocument.Parse(json);
        }
        catch (JsonException ex)
        {
            throw new HartsyInferenceException("Wyoming frame header is not valid JSON; the peer is not speaking this protocol.", ex);
        }
    }

    private static (string Type, int DataLength, int PayloadLength) ReadHeaderFields(JsonDocument header)
    {
        JsonElement root = header.RootElement;
        if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("type", out JsonElement typeElement)
            || typeElement.ValueKind != JsonValueKind.String)
            throw new HartsyInferenceException("Wyoming frame header has no 'type'.");
        return (typeElement.GetString()!, ReadLength(root, "data_length"), ReadLength(root, "payload_length"));
    }

    private static int ReadLength(JsonElement root, string key) =>
        root.TryGetProperty(key, out JsonElement value) && value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out int length) ? length : 0;

    /// <summary>Combines the inline <c>data</c> object with the out-of-line block, the block winning per key —
    /// the reference reader's <c>dict.update</c> order.</summary>
    private static JsonDocument? MergeData(JsonElement inline, JsonDocument? block)
    {
        bool hasInline = inline.ValueKind == JsonValueKind.Object;
        bool hasBlock = block is not null && block.RootElement.ValueKind == JsonValueKind.Object;
        if (!hasInline)
        {
            if (hasBlock) return block;
            block?.Dispose();
            return null;
        }
        if (!hasBlock)
        {
            block?.Dispose();
            return CloneObject(inline, default);
        }
        JsonDocument merged = CloneObject(inline, block!.RootElement);
        block.Dispose();
        return merged;
    }

    private static JsonDocument CloneObject(JsonElement primary, JsonElement overrides)
    {
        ArrayBufferWriter<byte> buffer = new(256);
        using (Utf8JsonWriter writer = new(buffer))
        {
            writer.WriteStartObject();
            foreach (JsonProperty property in primary.EnumerateObject())
            {
                if (overrides.ValueKind == JsonValueKind.Object && overrides.TryGetProperty(property.Name, out _)) continue;
                property.WriteTo(writer);
            }
            if (overrides.ValueKind == JsonValueKind.Object)
            {
                foreach (JsonProperty property in overrides.EnumerateObject()) property.WriteTo(writer);
            }
            writer.WriteEndObject();
        }
        return JsonDocument.Parse(buffer.WrittenSpan.ToArray());
    }

    /// <summary>Index of the newline ending the header line, or -1 at end of stream.</summary>
    private async Task<int> ReadLineAsync(CancellationToken cancel)
    {
        while (true)
        {
            int newline = Array.IndexOf(_buffer, Newline, 0, _buffered);
            if (newline >= 0) return newline;
            if (_buffered == _buffer.Length)
                throw new HartsyInferenceException($"Wyoming frame header exceeded {MaxHeaderBytes} bytes with no newline; the peer is not speaking this protocol.");

            int read = await _stream.ReadAsync(_buffer.AsMemory(_buffered), cancel).ConfigureAwait(false);
            if (read == 0) return -1;
            _buffered += read;
        }
    }

    /// <summary>Fills <paramref name="destination"/> from whatever the header read buffered ahead, then the stream.</summary>
    private async Task FillAsync(byte[] destination, int count, CancellationToken cancel)
    {
        int fromBuffer = Math.Min(_buffered, count);
        if (fromBuffer > 0)
        {
            Array.Copy(_buffer, 0, destination, 0, fromBuffer);
            Consume(fromBuffer);
        }
        if (count > fromBuffer)
            await _stream.ReadExactlyAsync(destination.AsMemory(fromBuffer, count - fromBuffer), cancel).ConfigureAwait(false);
    }

    private void Consume(int count)
    {
        _buffered -= count;
        if (_buffered > 0) Array.Copy(_buffer, count, _buffer, 0, _buffered);
    }
}

namespace HartsyInference.ModelAssets.Onnx;

/// <summary>Minimal protobuf wire-format cursor over a byte buffer — just enough to walk ONNX
/// <c>ModelProto</c> / <c>GraphProto</c> / <c>NodeProto</c> / <c>TensorProto</c> without a generated descriptor
/// or the <c>Google.Protobuf</c> dependency (matching the engine's hand-rolled pickle / safetensors parsers).
/// Positions are absolute indices into the shared buffer so length-delimited sub-messages can be re-read by
/// constructing a bounded cursor over the same span — and a <c>raw_data</c> blob's absolute offset is recovered
/// for a zero-intermediate-copy materialization.</summary>
internal ref struct ProtoReader
{
    public const int WireVarint = 0;
    public const int WireFixed64 = 1;
    public const int WireLengthDelimited = 2;
    public const int WireFixed32 = 5;

    private readonly ReadOnlySpan<byte> _buf;
    private int _pos;
    private readonly int _end;

    public ProtoReader(ReadOnlySpan<byte> buf) : this(buf, 0, buf.Length) { }

    public ProtoReader(ReadOnlySpan<byte> buf, int start, int end)
    {
        _buf = buf;
        _pos = start;
        _end = end;
    }

    public readonly bool HasMore => _pos < _end;
    public readonly int Position => _pos;

    /// <summary>Reads a field tag, splitting it into field number and wire type. Returns false at the end.</summary>
    public bool TryReadTag(out int fieldNumber, out int wireType)
    {
        if (_pos >= _end) { fieldNumber = 0; wireType = 0; return false; }
        ulong tag = ReadVarint();
        fieldNumber = (int)(tag >> 3);
        wireType = (int)(tag & 0x7);
        return true;
    }

    public ulong ReadVarint()
    {
        ulong result = 0;
        int shift = 0;
        while (_pos < _end)
        {
            byte b = _buf[_pos++];
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
            shift += 7;
            if (shift > 63) throw new InvalidDataException("Protobuf varint overflow.");
        }
        throw new InvalidDataException("Truncated protobuf varint.");
    }

    /// <summary>Reads a length-delimited field, returning its absolute offset/length into the shared buffer
    /// and advancing past it.</summary>
    public (int Offset, int Length) ReadLengthDelimited()
    {
        int len = (int)ReadVarint();
        int off = _pos;
        if (off + len > _end) throw new InvalidDataException("Length-delimited field exceeds buffer.");
        _pos += len;
        return (off, len);
    }

    public string ReadString()
    {
        (int off, int len) = ReadLengthDelimited();
        return System.Text.Encoding.UTF8.GetString(_buf.Slice(off, len));
    }

    /// <summary>Skips a field of the given wire type (used to ignore the many ONNX fields we don't need).</summary>
    public void Skip(int wireType)
    {
        switch (wireType)
        {
            case WireVarint: ReadVarint(); break;
            case WireFixed64: _pos += 8; break;
            case WireLengthDelimited: ReadLengthDelimited(); break;
            case WireFixed32: _pos += 4; break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wireType}.");
        }
        if (_pos > _end) throw new InvalidDataException("Protobuf skip ran past buffer.");
    }
}

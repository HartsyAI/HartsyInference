using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf.Codecs;

namespace HartsyInference.ModelAssets.Gguf;

/// <summary>Writes a GGUF v3 file. Inverse of <see cref="GgufLoader"/> — same on-disk format. Two-pass design: the caller registers tensors first (each gets a relative offset assigned in registration order), then <see cref="Finalize"/> writes the header + descriptor table + zero-padded data section.
///
/// <para><b>Usage</b>:</para>
/// <code>
/// using GgufWriter w = new(outputPath);
/// w.SetMetadata("general.architecture", "flux");
/// w.SetMetadata("general.alignment", (ulong)32);
/// w.AddTensor("blk.0.attn_q.weight", quantizedTensor);  // tensor stays untouched until Flush
/// w.AddTensor("blk.0.attn_k.weight", quantizedTensor2);
/// w.Flush();
/// </code>
///
/// <para>The writer does not quantize. Pass already-quantized tensors. Use <see cref="GgufQuantizer"/> for the safetensors → GGUF orchestration that handles quantization + writing in one pass.</para></summary>
public sealed class GgufWriter : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF" bytes 47 47 55 46 read as little-endian uint32 — matches real city96 / unsloth files.
    private const uint GgufVersion = 3;
    private const long DefaultAlignment = 32;

    private readonly string _outputPath;
    private readonly Dictionary<string, object> _metadata = new();
    private readonly List<TensorEntry> _tensors = new();
    private int _disposed;

    private sealed class TensorEntry
    {
        public required string Name { get; init; }
        public required Tensor Tensor { get; init; }
        public required uint GgufTypeId { get; init; }
        public long RelativeOffset { get; set; }
    }

    /// <summary>Creates a writer that will produce a GGUF file at <paramref name="outputPath"/> on <see cref="Finalize"/>.</summary>
    public GgufWriter(string outputPath)
    {
        _outputPath = outputPath ?? throw new ArgumentNullException(nameof(outputPath));
        SetMetadata("general.alignment", (ulong)DefaultAlignment);
    }

    /// <summary>Sets a metadata key. Value must be one of the supported GGUF types: bool / sbyte / byte / short / ushort / int / uint / long / ulong / float / double / string / string[].</summary>
    public void SetMetadata(string key, object value)
    {
        ThrowIfDisposed();
        _metadata[key] = value ?? throw new ArgumentNullException(nameof(value));
    }

    /// <summary>Registers a tensor. The tensor's data is read at <see cref="Finalize"/> — caller must keep it alive until then. Throws if the dtype isn't a valid ggml type (i.e. has no canonical type ID).</summary>
    public void AddTensor(string name, Tensor tensor)
    {
        ThrowIfDisposed();
        if (string.IsNullOrEmpty(name)) throw new ArgumentException("Tensor name cannot be empty.", nameof(name));
        if (_tensors.Any(t => t.Name == name))
            throw new ArgumentException($"Duplicate tensor name '{name}'.", nameof(name));

        uint ggufTypeId = MapDTypeToGgufId(tensor.DType);
        _tensors.Add(new TensorEntry
        {
            Name = name,
            Tensor = tensor,
            GgufTypeId = ggufTypeId,
        });
    }

    /// <summary>Number of tensors registered so far.</summary>
    public int TensorCount => _tensors.Count;

    /// <summary>Writes the GGUF file. All registered tensors are emitted in registration order.</summary>
    public void Flush()
    {
        ThrowIfDisposed();

        long alignment = (long)(ulong)_metadata["general.alignment"];
        AssignTensorOffsets(alignment);

        using FileStream fs = new(_outputPath, FileMode.Create, FileAccess.Write, FileShare.None);
        using BinaryWriter w = new(fs, Encoding.UTF8, leaveOpen: false);

        w.Write(GgufMagic);
        w.Write(GgufVersion);
        w.Write((ulong)_tensors.Count);
        w.Write((ulong)_metadata.Count);

        foreach (KeyValuePair<string, object> kv in _metadata)
        {
            WriteString(w, kv.Key);
            WriteMetadataValue(w, kv.Value);
        }

        foreach (TensorEntry t in _tensors)
        {
            WriteTensorInfo(w, t);
        }

        long currentPos = fs.Position;
        long padding = AlignUp(currentPos, alignment) - currentPos;
        for (long i = 0; i < padding; i++) w.Write((byte)0);

        long dataStart = fs.Position;
        foreach (TensorEntry t in _tensors)
        {
            long expectedAbsolute = dataStart + t.RelativeOffset;
            long padToHere = expectedAbsolute - fs.Position;
            for (long i = 0; i < padToHere; i++) w.Write((byte)0);
            WriteTensorData(w, t.Tensor);
        }
    }

    private void AssignTensorOffsets(long alignment)
    {
        long offset = 0;
        foreach (TensorEntry t in _tensors)
        {
            offset = AlignUp(offset, alignment);
            t.RelativeOffset = offset;
            offset += t.Tensor.DType.ComputeByteCount(t.Tensor.Shape.ElementCount);
        }
    }

    private static void WriteString(BinaryWriter w, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static void WriteMetadataValue(BinaryWriter w, object value)
    {
        switch (value)
        {
            case byte b: w.Write((uint)0); w.Write(b); break;
            case sbyte sb: w.Write((uint)1); w.Write(sb); break;
            case ushort us: w.Write((uint)2); w.Write(us); break;
            case short s: w.Write((uint)3); w.Write(s); break;
            case uint u: w.Write((uint)4); w.Write(u); break;
            case int i: w.Write((uint)5); w.Write(i); break;
            case float f: w.Write((uint)6); w.Write(f); break;
            case bool b: w.Write((uint)7); w.Write(b ? (byte)1 : (byte)0); break;
            case string str: w.Write((uint)8); WriteString(w, str); break;
            case string[] arr:
                w.Write((uint)9);
                w.Write((uint)8);
                w.Write((ulong)arr.Length);
                foreach (string s in arr) WriteString(w, s);
                break;
            case ulong ul: w.Write((uint)10); w.Write(ul); break;
            case long l: w.Write((uint)11); w.Write(l); break;
            case double d: w.Write((uint)12); w.Write(d); break;
            default:
                throw new ArgumentException($"Unsupported metadata value type: {value.GetType()}");
        }
    }

    private static void WriteTensorInfo(BinaryWriter w, TensorEntry t)
    {
        WriteString(w, t.Name);
        int rank = t.Tensor.Shape.Rank;
        w.Write((uint)rank);
        for (int i = 0; i < rank; i++)
        {
            w.Write((ulong)t.Tensor.Shape[i]);
        }
        w.Write(t.GgufTypeId);
        w.Write((ulong)t.RelativeOffset);
    }

    private static unsafe void WriteTensorData(BinaryWriter w, Tensor tensor)
    {
        long byteCount = tensor.DType.ComputeByteCount(tensor.Shape.ElementCount);
        ReadOnlySpan<byte> data = new ReadOnlySpan<byte>((void*)tensor.DataPointer, (int)byteCount);
        w.Write(data);
    }

    private static long AlignUp(long value, long alignment) => (value + alignment - 1) & ~(alignment - 1);

    /// <summary>Reverse of <see cref="GgufLoader.MapGgufType"/>. Maps HartsyInference DType → ggml type ID per `enum ggml_type` in ggml.h.</summary>
    public static uint MapDTypeToGgufId(DType dtype)
    {
        if (dtype == DType.F32) return 0;
        if (dtype == DType.F16) return 1;
        if (dtype == DType.Q4_0) return 2;
        if (dtype == DType.Q4_1) return 3;
        if (dtype == DType.Q5_0) return 6;
        if (dtype == DType.Q5_1) return 7;
        if (dtype == DType.Q8_0) return 8;
        if (dtype == DType.Q8_1) return 9;
        if (dtype == DType.Q2_K) return 10;
        if (dtype == DType.Q3_K) return 11;
        if (dtype == DType.Q4_K) return 12;
        if (dtype == DType.Q5_K) return 13;
        if (dtype == DType.Q6_K) return 14;
        if (dtype == DType.Q8_K) return 15;
        if (dtype == DType.IQ2_XXS) return 16;
        if (dtype == DType.IQ2_XS) return 17;
        if (dtype == DType.IQ3_XXS) return 18;
        if (dtype == DType.IQ1_S) return 19;
        if (dtype == DType.IQ4_NL) return 20;
        if (dtype == DType.IQ3_S) return 21;
        if (dtype == DType.IQ2_S) return 22;
        if (dtype == DType.IQ4_XS) return 23;
        if (dtype == DType.IQ1_M) return 29;
        if (dtype == DType.BF16) return 30;
        if (dtype == DType.TQ1_0) return 31;
        if (dtype == DType.TQ2_0) return 32;
        if (dtype == DType.MXFP4) return 39;
        throw new ArgumentException($"DType {dtype} has no canonical ggml type ID.", nameof(dtype));
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _tensors.Clear();
        _metadata.Clear();
    }
}

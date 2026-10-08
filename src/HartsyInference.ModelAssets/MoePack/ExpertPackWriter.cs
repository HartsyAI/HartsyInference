using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.ModelAssets.MoePack;

/// <summary>
/// Writes an expert pack: each expert's three projections quantized with the existing GGUF codecs, appended as 4 KiB-aligned
/// records, then a manifest and a <c>COMPLETE</c> marker. Nothing is visible to a reader until <see cref="Finish"/> succeeds:
/// data goes to <c>experts.bin.partial</c> and is renamed only after the manifest is in place.
/// </summary>
public sealed class ExpertPackWriter : IDisposable
{
    /// <summary>Current pack format version.</summary>
    public const int FormatVersion = 1;

    /// <summary>Largest matrix a pack will quantize in one piece; keeps every element count an <c>int</c>.</summary>
    public const long MaxMatrixElements = 1L << 28;

    /// <summary>Records start on this boundary so the reader can issue aligned reads.</summary>
    public const int RecordAlignment = 4096;

    private readonly string _directory;
    private readonly string _fingerprint;
    private readonly int _hidden;
    private readonly int _intermediate;
    private readonly DType _dtype;
    private readonly HashSet<ExpertKey> _expected;
    private readonly FileStream _stream;
    private readonly FileStream _lock;
    private readonly string _lockPath;
    private readonly List<ExpertPackRecord> _records = [];
    private readonly HashSet<ExpertKey> _written = [];
    private long _position;
    private bool _finished;

    /// <summary>Starts a pack in <paramref name="directory"/>.</summary>
    /// <param name="directory">Created if missing. Refused when it already holds a completed pack.</param>
    /// <param name="topologyFingerprint">Fingerprint of the topology these experts belong to.</param>
    /// <param name="hidden">Model width H; the quantized row length of gate and up.</param>
    /// <param name="intermediate">Expert inner width I; the quantized row length of down.</param>
    /// <param name="dtype">Quant dtype of every projection: Q8_0, Q4_K, Q5_K or Q6_K.</param>
    /// <param name="expectedExperts">The exact experts the pack must hold. Only these can be added, and <see
    /// cref="Finish"/> publishes only when all are present.</param>
    /// <exception cref="ArgumentException">The dtype is not a pack dtype, a dimension is not a multiple of its block
    /// size, or the expected set is empty.</exception>
    /// <exception cref="InvalidOperationException">The directory already holds a completed pack.</exception>
    public ExpertPackWriter(string directory, string topologyFingerprint, int hidden, int intermediate, DType dtype,
        IEnumerable<ExpertKey> expectedExperts)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        ArgumentException.ThrowIfNullOrWhiteSpace(topologyFingerprint);
        ArgumentNullException.ThrowIfNull(expectedExperts);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hidden);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(intermediate);
        // The same bounds the reader enforces, and element counts that cannot overflow the arithmetic that sizes a matrix.
        if (hidden > ExpertPackReader.MaxDimension || intermediate > ExpertPackReader.MaxDimension)
            throw new ArgumentOutOfRangeException(nameof(hidden), $"Dimensions must be at most {ExpertPackReader.MaxDimension}.");
        if ((long)hidden * intermediate > MaxMatrixElements)
            throw new ArgumentOutOfRangeException(nameof(intermediate), $"A {hidden} x {intermediate} matrix is too large to pack.");
        if (!ExpertPackDTypes.IsPackDType(dtype))
            throw new ArgumentException($"{dtype.Name} is not a pack dtype; packs store Q8_0, Q4_K, Q5_K or Q6_K.", nameof(dtype));
        if (hidden % dtype.BlockElementCount != 0 || intermediate % dtype.BlockElementCount != 0)
            throw new ArgumentException(
                    $"Hidden {hidden} and intermediate {intermediate} must be multiples of {dtype.Name}'s block size {dtype.BlockElementCount}.");
        _expected = [.. expectedExperts];
        if (_expected.Count == 0) throw new ArgumentException("A pack needs at least one expected expert.", nameof(expectedExperts));
        _directory = directory;
        _fingerprint = topologyFingerprint;
        _hidden = hidden;
        _intermediate = intermediate;
        _dtype = dtype;
        Directory.CreateDirectory(directory);
        // One writer per directory. Ownership is the open handle, not the file's existence: FileShare.None makes a second
        // open fail while a writer is alive, and a WRITING file left by a crashed writer (no handle) opens and is taken over.
        _lockPath = Path.Combine(directory, "WRITING");
        try
        {
            _lock = new FileStream(_lockPath, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None);
        }
        catch (IOException)
        {
            throw new InvalidOperationException($"'{directory}' is already being written; a second writer would replace the pack.");
        }
        try
        {
            // Checked under the lock: a writer that published between an earlier check and now must not be overwritten.
            if (File.Exists(Path.Combine(directory, "COMPLETE")))
                throw new InvalidOperationException($"'{directory}' already holds a completed expert pack.");
            _stream = new FileStream(Path.Combine(directory, "experts.bin.partial"), FileMode.Create, FileAccess.Write, FileShare.None);
        }
        catch
        {
            // Setup failed after the lock was taken; release it so a later writer is not refused by a stale marker.
            ReleaseLock();
            throw;
        }
    }

    /// <summary>Quantizes and appends one expert. Gate and up are <c>[I, H]</c> row-major; down is <c>[H, I]</c>.</summary>
    /// <remarks>A call that throws leaves the expert unwritten, so the caller can retry it with corrected input.</remarks>
    /// <exception cref="ArgumentException">The expert is not in the expected set, is already written, or an array does
    /// not hold its matrix.</exception>
    public ExpertPackRecord AddExpert(int layer, int expert, float[] gate, float[] up, float[] down, ushort bank = 0)
    {
        ObjectDisposedException.ThrowIf(_finished, this);
        ExpertKey key = new(layer, expert, bank);
        if (!_expected.Contains(key)) throw new ArgumentException($"{key} is not in the pack's expected set.");
        if (_written.Contains(key)) throw new ArgumentException($"{key} is already in the pack.");
        ValidateLength(gate, _intermediate * _hidden, nameof(gate));
        ValidateLength(up, _intermediate * _hidden, nameof(up));
        ValidateLength(down, _hidden * _intermediate, nameof(down));

        byte[] record = Concat(
            Quantize(gate, _intermediate, _hidden),
            Quantize(up, _intermediate, _hidden),
            Quantize(down, _hidden, _intermediate));

        long offset = AlignUp(_position);
        if (offset > _position) _stream.Write(new byte[offset - _position]);
        _stream.Write(record);
        _position = offset + record.Length;

        ExpertPackRecord written = new(layer, expert, bank, offset, record.Length, Convert.ToHexString(SHA256.HashData(record)).ToLowerInvariant());
        _records.Add(written);
        _written.Add(key);
        return written;
    }

    /// <summary>Publishes the pack: data, then manifest, then the <c>COMPLETE</c> marker. Idempotent after success.</summary>
    /// <exception cref="InvalidOperationException">Some expected expert has not been added.</exception>
    public void Finish()
    {
        if (_finished) return;
        if (!_written.SetEquals(_expected))
            throw new InvalidOperationException($"The pack expects {_expected.Count} experts but holds {_written.Count}; it is not published.");
        _stream.Flush(flushToDisk: true);
        _stream.Dispose();
        string partial = Path.Combine(_directory, "experts.bin.partial");
        string data = Path.Combine(_directory, "experts.bin");
        File.Move(partial, data, overwrite: true);

        ExpertPackManifest manifest = new(FormatVersion, _fingerprint, _hidden, _intermediate, _dtype.Name, _records, _expected.Count);
        string manifestPath = Path.Combine(_directory, "manifest.json");
        File.WriteAllText(manifestPath + ".partial", JsonSerializer.Serialize(manifest, ExpertPackJsonContext.Default.ExpertPackManifest));
        File.Move(manifestPath + ".partial", manifestPath, overwrite: true);

        File.WriteAllText(Path.Combine(_directory, "COMPLETE"), FormatVersion.ToString());
        _finished = true;
        ReleaseLock();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_finished) return;
        _stream.Dispose();
        ReleaseLock();
    }

    private void ReleaseLock()
    {
        _lock.Dispose();
        File.Delete(_lockPath);
    }

    private byte[] Quantize(float[] values, int rows, int cols)
    {
        using Tensor source = new(new TensorShape(rows, cols), DType.F32);
        unsafe
        {
            float* dst = (float*)source.DataPointer;
            for (int i = 0; i < values.Length; i++) dst[i] = values[i];
        }
        using Tensor quant = GgufQuantizer.Quantize(source, _dtype);
        long bytes = _dtype.ComputeByteCount(quant.ElementCount);
        byte[] result = new byte[bytes];
        unsafe
        {
            fixed (byte* dst = result) Buffer.MemoryCopy((void*)quant.DataPointer, dst, bytes, bytes);
        }
        return result;
    }

    private long AlignUp(long value) => (value + RecordAlignment - 1) / RecordAlignment * RecordAlignment;

    private static void ValidateLength(float[] values, int expected, string name)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Length != expected) throw new ArgumentException($"{name} must hold {expected} values; it holds {values.Length}.", name);
    }

    private static byte[] Concat(byte[] a, byte[] b, byte[] c)
    {
        byte[] result = new byte[a.Length + b.Length + c.Length];
        a.CopyTo(result, 0);
        b.CopyTo(result, a.Length);
        c.CopyTo(result, a.Length + b.Length);
        return result;
    }
}

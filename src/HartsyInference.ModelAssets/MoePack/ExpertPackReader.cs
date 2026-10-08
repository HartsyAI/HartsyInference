using System.Security.Cryptography;
using Microsoft.Win32.SafeHandles;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.MoePack;

/// <summary>
/// Reads a completed expert pack as an <see cref="IExpertSource"/>: each <see cref="Resolve"/> reads one record, checks its
/// SHA-256 and returns quantized weights ready for the cache. Refuses incomplete packs, other format versions, and a topology
/// fingerprint it was not built for.
/// </summary>
public sealed class ExpertPackReader : IExpertSource, IDisposable
{
    private readonly SafeFileHandle _handle;
    private readonly Dictionary<ExpertKey, ExpertPackRecord> _records = [];
    private readonly DType _dtype;
    private readonly bool _verifyChecksums;

    private ExpertPackReader(string directory, ExpertPackManifest manifest, bool verifyChecksums)
    {
        Directory = directory;
        TopologyFingerprint = manifest.TopologyFingerprint;
        Hidden = manifest.Hidden;
        Intermediate = manifest.Intermediate;
        _dtype = ExpertPackDTypes.Resolve(manifest.DType);
        _verifyChecksums = verifyChecksums;
        foreach (ExpertPackRecord record in manifest.Records)
            _records.Add(new ExpertKey(record.Layer, record.Expert, record.Bank), record);
        _handle = File.OpenHandle(Path.Combine(directory, "experts.bin"), FileMode.Open, FileAccess.Read, FileShare.Read);
    }

    /// <summary>The directory the pack was opened from.</summary>
    public string Directory { get; }

    /// <summary>The topology fingerprint the pack was built for.</summary>
    public string TopologyFingerprint { get; }

    /// <summary>Model width H.</summary>
    public int Hidden { get; }

    /// <summary>Expert inner width I.</summary>
    public int Intermediate { get; }

    /// <summary>Number of experts in the pack.</summary>
    public int Count => _records.Count;

    /// <summary>Every expert in the pack, in no particular order.</summary>
    public IEnumerable<ExpertKey> Keys => _records.Keys;

    /// <summary>Every record, as written in the manifest.</summary>
    public IEnumerable<ExpertPackRecord> Records => _records.Values;

    /// <summary>Quantized bytes stored (records only, excluding alignment padding and the manifest).</summary>
    public long PayloadBytes => _records.Values.Sum(static record => record.Length);

    /// <inheritdoc/>
    public ExpertBacking Backing => ExpertBacking.Pack;

    /// <summary>Opens a completed pack.</summary>
    /// <param name="directory">The pack directory.</param>
    /// <param name="expectedFingerprint">When set, the pack must have been built for this topology fingerprint.</param>
    /// <param name="verifyChecksums">Check each record's SHA-256 as it is read.</param>
    /// <exception cref="InvalidDataException">The pack is incomplete, another version, or built for a different topology.</exception>
    public static ExpertPackReader Open(string directory, string? expectedFingerprint = null, bool verifyChecksums = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        if (!File.Exists(Path.Combine(directory, "COMPLETE")))
            throw new InvalidDataException($"'{directory}' is not a completed expert pack (no COMPLETE marker).");
        ExpertPackManifest manifest = JsonSerializer.Deserialize(
            File.ReadAllText(Path.Combine(directory, "manifest.json")), ExpertPackJsonContext.Default.ExpertPackManifest)
            ?? throw new InvalidDataException("The pack manifest is empty.");
        if (manifest.Format != ExpertPackWriter.FormatVersion)
            throw new InvalidDataException($"Expert pack format {manifest.Format} is not version {ExpertPackWriter.FormatVersion}.");
        if (expectedFingerprint is not null && manifest.TopologyFingerprint != expectedFingerprint)
            throw new InvalidDataException("The expert pack was built for a different topology fingerprint.");
        return new ExpertPackReader(directory, manifest, verifyChecksums);
    }

    /// <inheritdoc/>
    /// <exception cref="KeyNotFoundException">The pack has no record for <paramref name="key"/>.</exception>
    /// <exception cref="InvalidDataException">The record's checksum does not match its bytes.</exception>
    public ExpertWeights Resolve(ExpertKey key)
    {
        if (!_records.TryGetValue(key, out ExpertPackRecord? record)) throw new KeyNotFoundException($"The expert pack has no record for {key}.");
        byte[] bytes = new byte[record.Length];
        int read = RandomAccess.Read(_handle, bytes, record.Offset);
        if (read != bytes.Length) throw new InvalidDataException($"Short read for {key}: {read} of {bytes.Length} bytes.");
        if (_verifyChecksums && Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant() != record.Sha256)
            throw new InvalidDataException($"Checksum mismatch for {key}; the pack is corrupt.");

        long gateBytes = _dtype.ComputeByteCount((long)Intermediate * Hidden);
        long downBytes = _dtype.ComputeByteCount((long)Hidden * Intermediate);
        Tensor gate = Load(bytes, 0, new TensorShape(Intermediate, Hidden), gateBytes);
        Tensor up = Load(bytes, gateBytes, new TensorShape(Intermediate, Hidden), gateBytes);
        Tensor down = Load(bytes, 2 * gateBytes, new TensorShape(Hidden, Intermediate), downBytes);
        return new ExpertWeights(key, new ExpertMatrix(gate), new ExpertMatrix(down), new ExpertMatrix(up));
    }

    /// <inheritdoc/>
    public void Dispose() => _handle.Dispose();

    private Tensor Load(byte[] bytes, long offset, TensorShape shape, long length)
    {
        Tensor tensor = new(shape, _dtype);
        unsafe
        {
            fixed (byte* src = bytes) Buffer.MemoryCopy(src + offset, (void*)tensor.DataPointer, length, length);
        }
        return tensor;
    }
}

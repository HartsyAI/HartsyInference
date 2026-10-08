using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Microsoft.Win32.SafeHandles;

namespace HartsyInference.ModelAssets.MoePack;

/// <summary>
/// Reads a completed expert pack as an <see cref="IExpertSource"/>: each <see cref="Resolve"/> reads one record straight into
/// native tensor memory and checks its SHA-256 incrementally. Refuses incomplete packs, other format versions, a topology
/// fingerprint it was not built for, and any record whose length does not match its three projections.
/// </summary>
public sealed class ExpertPackReader : IExpertSource, IDisposable
{
    private readonly SafeFileHandle _handle;
    private readonly Dictionary<ExpertKey, ExpertPackRecord> _records = [];
    private readonly DType _dtype;
    private readonly bool _verifyChecksums;
    private readonly long _gateBytes;
    private readonly long _downBytes;

    private ExpertPackReader(string directory, ExpertPackManifest manifest, bool verifyChecksums)
    {
        Directory = directory;
        TopologyFingerprint = manifest.TopologyFingerprint;
        Hidden = manifest.Hidden;
        Intermediate = manifest.Intermediate;
        _dtype = ExpertPackDTypes.Resolve(manifest.DType);
        _verifyChecksums = verifyChecksums;
        _gateBytes = _dtype.ComputeByteCount((long)Intermediate * Hidden);
        _downBytes = _dtype.ComputeByteCount((long)Hidden * Intermediate);
        long expectedLength = 2 * _gateBytes + _downBytes;
        foreach (ExpertPackRecord record in manifest.Records)
        {
            if (record.Length != expectedLength)
                throw new InvalidDataException($"Record for {new ExpertKey(record.Layer, record.Expert, record.Bank)} is {record.Length} bytes; its three projections need {expectedLength}.");
            if (!_records.TryAdd(new ExpertKey(record.Layer, record.Expert, record.Bank), record))
                throw new InvalidDataException($"The manifest lists {new ExpertKey(record.Layer, record.Expert, record.Bank)} twice.");
        }
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
    /// <exception cref="InvalidDataException">The pack is incomplete, another version, built for a different topology, or has a malformed record.</exception>
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
        if (manifest.Records.Count != manifest.ExpertCount)
            throw new InvalidDataException($"The manifest lists {manifest.Records.Count} experts; it declares {manifest.ExpertCount}.");
        return new ExpertPackReader(directory, manifest, verifyChecksums);
    }

    /// <inheritdoc/>
    /// <exception cref="KeyNotFoundException">The pack has no record for <paramref name="key"/>.</exception>
    /// <exception cref="InvalidDataException">The record's checksum does not match its bytes.</exception>
    public ExpertWeights Resolve(ExpertKey key)
    {
        if (!_records.TryGetValue(key, out ExpertPackRecord? record)) throw new KeyNotFoundException($"The expert pack has no record for {key}.");
        Tensor gate = new(new TensorShape(Intermediate, Hidden), _dtype);
        Tensor up = new(new TensorShape(Intermediate, Hidden), _dtype);
        Tensor down = new(new TensorShape(Hidden, Intermediate), _dtype);
        try
        {
            using IncrementalHash? hash = _verifyChecksums ? IncrementalHash.CreateHash(HashAlgorithmName.SHA256) : null;
            ReadProjection(record.Offset, gate, _gateBytes, hash, key);
            ReadProjection(record.Offset + _gateBytes, up, _gateBytes, hash, key);
            ReadProjection(record.Offset + 2 * _gateBytes, down, _downBytes, hash, key);
            if (hash is not null && Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant() != record.Sha256)
                throw new InvalidDataException($"Checksum mismatch for {key}; the pack is corrupt.");
        }
        catch
        {
            gate.Dispose();
            up.Dispose();
            down.Dispose();
            throw;
        }
        return new ExpertWeights(key, new ExpertMatrix(gate), new ExpertMatrix(down), new ExpertMatrix(up));
    }

    /// <inheritdoc/>
    public void Dispose() => _handle.Dispose();

    private unsafe void ReadProjection(long offset, Tensor destination, long length, IncrementalHash? hash, ExpertKey key)
    {
        Span<byte> span = new((void*)destination.DataPointer, checked((int)length));
        int read = RandomAccess.Read(_handle, span, offset);
        if (read != span.Length) throw new InvalidDataException($"Short read for {key}: {read} of {span.Length} bytes.");
        hash?.AppendData(span);
    }
}

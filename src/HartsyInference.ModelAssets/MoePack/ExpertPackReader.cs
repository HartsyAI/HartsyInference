using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using System.IO.MemoryMappedFiles;

namespace HartsyInference.ModelAssets.MoePack;

/// <summary>
/// Reads a completed expert pack as an <see cref="IExpertSource"/>: each <see cref="Resolve"/> reads one record straight into
/// native tensor memory and checks its SHA-256 incrementally. Refuses incomplete packs, other format versions, a topology
/// fingerprint it was not built for, and any record whose length does not match its three projections.
/// </summary>
public sealed class ExpertPackReader : IExpertSource, IDisposable
{
    private MemoryMappedFile? _mapping;
    private MemoryMappedViewAccessor? _view;
    private unsafe byte* _base;
    private int _released;
    private readonly Dictionary<ExpertKey, ExpertPackRecord> _records = [];
    private readonly DType _dtype;
    private readonly bool _verifyChecksums;
    private readonly long _gateBytes;
    private readonly long _downBytes;

    private unsafe ExpertPackReader(string directory, ExpertPackManifest manifest, bool verifyChecksums)
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
        long fileLength = new FileInfo(Path.Combine(directory, "experts.bin")).Length;
        foreach (ExpertPackRecord record in manifest.Records)
        {
            ExpertKey recordKey = new(record.Layer, record.Expert, record.Bank);
            if (record.Length != expectedLength)
                throw new InvalidDataException(
                    $"Record for {recordKey} is {record.Length} bytes; its three projections need {expectedLength}.");
            if (record.Offset < 0 || record.Offset > fileLength - record.Length)
                throw new InvalidDataException($"Record for {recordKey} lies outside experts.bin.");
            if (!_records.TryAdd(recordKey, record))
                throw new InvalidDataException($"The manifest lists {recordKey} twice.");
        }
        _mapping = MemoryMappedFile.CreateFromFile(Path.Combine(directory, "experts.bin"), FileMode.Open, null, 0,
            MemoryMappedFileAccess.Read);
        _view = _mapping.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* pointer = null;
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
        _base = pointer;
    }

    /// <summary>The directory the pack was opened from.</summary>
    public string Directory { get; }

    /// <summary>The topology fingerprint the pack was built for.</summary>
    public string TopologyFingerprint { get; }

    /// <summary>Model width H.</summary>
    public int Hidden { get; }

    /// <summary>Expert inner width I.</summary>
    public int Intermediate { get; }

    /// <summary>True when each <see cref="Resolve"/> checks the record's SHA-256 before handing out its bytes.</summary>
    public bool VerifiesChecksums => _verifyChecksums;

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

    /// <summary>Largest manifest the reader will read into memory; a record is a few hundred bytes of JSON.</summary>
    public const long MaxManifestBytes = 256L << 20;

    /// <summary>Largest width a manifest may declare; far above any real expert and keeps the size arithmetic safe.</summary>
    public const int MaxDimension = 1 << 20;

    /// <summary>Opens a completed pack.</summary>
    /// <param name="directory">The pack directory.</param>
    /// <param name="expectedFingerprint">When set, the pack must have been built for this topology fingerprint.</param>
    /// <param name="verifyChecksums">Check each record's SHA-256 as it is read.</param>
    /// <exception cref="InvalidDataException">The pack is incomplete, another version, built for a different topology,
    /// or has a malformed record.</exception>
    public static ExpertPackReader Open(string directory, string? expectedFingerprint = null, bool verifyChecksums = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        if (!File.Exists(Path.Combine(directory, "COMPLETE")))
            throw new InvalidDataException($"'{directory}' is not a completed expert pack (no COMPLETE marker).");
        string manifestPath = Path.Combine(directory, "manifest.json");
        // Bound the size before reading: a tampered manifest must not be materialized in full to find out it is invalid.
        if (new FileInfo(manifestPath).Length > MaxManifestBytes)
            throw new InvalidDataException($"The pack manifest exceeds {MaxManifestBytes} bytes.");
        ExpertPackManifest manifest = JsonSerializer.Deserialize(
            File.ReadAllText(manifestPath), ExpertPackJsonContext.Default.ExpertPackManifest)
            ?? throw new InvalidDataException("The pack manifest is empty.");
        if (manifest.Format != ExpertPackWriter.FormatVersion)
            throw new InvalidDataException($"Expert pack format {manifest.Format} is not version {ExpertPackWriter.FormatVersion}.");
        if (expectedFingerprint is not null && manifest.TopologyFingerprint != expectedFingerprint)
            throw new InvalidDataException("The expert pack was built for a different topology fingerprint.");
        if (manifest.Hidden <= 0 || manifest.Intermediate <= 0 || manifest.Hidden > MaxDimension || manifest.Intermediate > MaxDimension)
            throw new InvalidDataException(
                $"The manifest declares dimensions {manifest.Hidden} x {manifest.Intermediate}; both must be in [1, {MaxDimension}].");
        if ((long)manifest.Hidden * manifest.Intermediate > ExpertPackWriter.MaxMatrixElements)
            throw new InvalidDataException(
                $"The manifest declares a {manifest.Hidden} x {manifest.Intermediate} matrix; " +
                $"the pack limit is {ExpertPackWriter.MaxMatrixElements} elements.");
        if (manifest.ExpertCount <= 0) throw new InvalidDataException("The manifest declares no experts.");
        if (manifest.Records.Count != manifest.ExpertCount)
            throw new InvalidDataException($"The manifest lists {manifest.Records.Count} experts; it declares {manifest.ExpertCount}.");
        DType dtype = ExpertPackDTypes.Resolve(manifest.DType);
        // Each width must be a whole number of quant blocks, not just their product: a 16 x 40 Q8_0 matrix has 640 elements
        // (20 blocks) but its rows split a block.
        if (manifest.Hidden % dtype.BlockElementCount != 0 || manifest.Intermediate % dtype.BlockElementCount != 0)
            throw new InvalidDataException(
                $"Dimensions {manifest.Hidden} x {manifest.Intermediate} are not whole {dtype.Name} blocks of {dtype.BlockElementCount}.");
        return new ExpertPackReader(directory, manifest, verifyChecksums);
    }

    /// <inheritdoc/>
    /// <exception cref="KeyNotFoundException">The pack has no record for <paramref name="key"/>.</exception>
    /// <exception cref="InvalidDataException">The record's checksum does not match its bytes.</exception>
    public unsafe ExpertWeights Resolve(ExpertKey key)
    {
        ObjectDisposedException.ThrowIf(_released != 0, this);
        if (!_records.TryGetValue(key, out ExpertPackRecord? record)) throw new KeyNotFoundException($"The expert pack has no record for {key}.");
        byte* start = _base + record.Offset;
        if (_verifyChecksums)
        {
            using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            hash.AppendData(new ReadOnlySpan<byte>(start, checked((int)record.Length)));
            if (Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant() != record.Sha256)
                throw new InvalidDataException($"Checksum mismatch for {key}; the pack is corrupt.");
        }
        Tensor gate = Borrow(start, new TensorShape(Intermediate, Hidden));
        Tensor up = Borrow(start + _gateBytes, new TensorShape(Intermediate, Hidden));
        Tensor down = Borrow(start + 2 * _gateBytes, new TensorShape(Hidden, Intermediate));
        return new ExpertWeights(key, new ExpertMatrix(gate), new ExpertMatrix(down), new ExpertMatrix(up));
    }

    /// <summary>Views of the mapping stay valid only while the reader is alive; each view keeps the reader rooted.</summary>
    private unsafe Tensor Borrow(byte* pointer, TensorShape shape)
    {
        Tensor view = new(pointer, shape, _dtype);
        view.SetKeepAlive(this);
        return view;
    }

    /// <summary>
    /// Releases the mapping. Weights already resolved are views into it and must not be used afterwards; a reader that is
    /// garbage-collected while views exist stays rooted by them, so the mapping is never released under a live view.
    /// Safe to call more than once; a finalizer releases the mapping if the caller never disposed the reader.
    /// </summary>
    public void Dispose()
    {
        Release();
        GC.SuppressFinalize(this);
    }

    /// <summary>Releases the mapping once, whether called from <see cref="Dispose"/> or the finalizer.</summary>
    ~ExpertPackReader() => Release();

    private unsafe void Release()
    {
        // A constructor that threw never assigned the mapping, and the finalizer still runs on it, so every field is optional here.
        if (Interlocked.Exchange(ref _released, 1) != 0) return;
        if (_base is not null && _view is not null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _base = null;
        }
        _view?.Dispose();
        _mapping?.Dispose();
    }

}

using System.Buffers.Binary;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.LLM.DeepSeekV41.Engram;

/// <summary>
/// The Engram hash constants of DeepSeek-V4.1-Flash: the compressed token map, per-layer hash multipliers, bucket primes and
/// cumulative column offsets. They are loaded from fixtures embedded in this assembly, never re-derived here: they come out of the
/// unmodified upstream <c>engram.py</c> (see tests/python-reference/deepseek_v41/dump_engram_constants.py), and the token map depends on a
/// Rust tokenizer normalizer chain that has no C# equivalent.
/// </summary>
/// <remarks>Each file's SHA-256 is checked against the embedded manifest on load, and the manifest pins the checkpoint revision the values were dumped for.</remarks>
public sealed class EngramConstants
{
    /// <summary>Marks a position that takes part in no n-gram (an image span) in the history cache.</summary>
    public const int Dead = -1;

    /// <summary>Hash columns per layer: (max n-gram size - 1) n-gram orders times the head count.</summary>
    public const int ColumnsPerLayer = 24;

    private const string ResourcePrefix = "HartsyInference.LLM.Engram.";
    private static readonly Lazy<EngramConstants> Pinned = new(Load);

    private readonly int[] _tokenMap;
    private readonly long[] _multipliers;
    private readonly long[] _offsets;
    private readonly long[] _primes;

    private EngramConstants(string checkpoint, string revision, int[] layerIds, int maxNgramSize, int nHeads, int compressedVocab,
        int padCompressedId, long[] tableRows, int[] tokenMap, long[] multipliers, long[] offsets, long[] primes)
    {
        Checkpoint = checkpoint;
        Revision = revision;
        LayerIds = layerIds;
        MaxNgramSize = maxNgramSize;
        HeadCount = nHeads;
        CompressedVocab = compressedVocab;
        PadCompressedId = padCompressedId;
        TableRows = tableRows;
        _tokenMap = tokenMap;
        _multipliers = multipliers;
        _offsets = offsets;
        _primes = primes;
    }

    /// <summary>The constants pinned for the checkpoint revision in <see cref="Revision"/>.</summary>
    public static EngramConstants Default => Pinned.Value;

    /// <summary>Hub repo the values were dumped for.</summary>
    public string Checkpoint { get; }

    /// <summary>Checkpoint revision (commit) the values were dumped for.</summary>
    public string Revision { get; }

    /// <summary>Backbone layer ids that own an Engram table, in table order.</summary>
    public IReadOnlyList<int> LayerIds { get; }

    /// <summary>Longest n-gram order (4): orders 2 to this are hashed.</summary>
    public int MaxNgramSize { get; }

    /// <summary>Hash heads per n-gram order.</summary>
    public int HeadCount { get; }

    /// <summary>Distinct ids in the compressed token space.</summary>
    public int CompressedVocab { get; }

    /// <summary>Compressed id of the pad token; dead and out-of-sequence lookbacks hash as this.</summary>
    public int PadCompressedId { get; }

    /// <summary>Rows in each layer's table (the sum of that layer's primes).</summary>
    public IReadOnlyList<long> TableRows { get; }

    /// <summary>Maps each tokenizer id to its compressed id.</summary>
    public ReadOnlySpan<int> TokenMap => _tokenMap;

    /// <summary>Layer count times <see cref="MaxNgramSize"/> multipliers, layer-major.</summary>
    public ReadOnlySpan<long> Multipliers => _multipliers;

    /// <summary>Layer count times <see cref="ColumnsPerLayer"/> cumulative column offsets, layer-major.</summary>
    public ReadOnlySpan<long> Offsets => _offsets;

    /// <summary>Layer count times (<see cref="MaxNgramSize"/> - 1) times <see cref="HeadCount"/> bucket primes, layer then order then head.</summary>
    public ReadOnlySpan<long> Primes => _primes;

    /// <summary>Tokenizer vocabulary size the token map covers.</summary>
    public int TokenizerVocab => _tokenMap.Length;

    /// <summary>Index of <paramref name="layer"/> in the per-layer arrays, or -1 when it has no Engram table.</summary>
    public int IndexOfLayer(int layer)
    {
        for (int i = 0; i < LayerIds.Count; i++)
        {
            if (LayerIds[i] == layer)
                return i;
        }
        return -1;
    }

    private static EngramConstants Load()
    {
        Assembly assembly = typeof(EngramConstants).Assembly;
        using JsonDocument manifest = JsonDocument.Parse(ReadResource(assembly, "manifest.json"));
        JsonElement root = manifest.RootElement;
        int[] layerIds = root.GetProperty("layer_ids").EnumerateArray().Select(static e => e.GetInt32()).ToArray();
        int layers = layerIds.Length;
        int maxNgram = root.GetProperty("max_ngram_size").GetInt32();
        int heads = root.GetProperty("n_heads").GetInt32();
        int columns = (maxNgram - 1) * heads;
        if (columns != ColumnsPerLayer)
            throw new HartsyInferenceException($"Engram fixtures describe {columns} hash columns per layer, expected {ColumnsPerLayer}.");
        JsonElement files = root.GetProperty("files");

        int[] tokenMap = ReadInt32s(assembly, files, "token_map.i32.bin", root.GetProperty("tokenizer_vocab").GetInt32());
        long[] multipliers = ReadInt64s(assembly, files, "multipliers.i64.bin", layers * maxNgram);
        long[] offsets = ReadInt64s(assembly, files, "offsets.i64.bin", layers * columns);
        long[] primes = ReadInt64s(assembly, files, "primes.i64.bin", layers * columns);
        long[] tableRows = root.GetProperty("table_rows").EnumerateArray().Select(static e => e.GetInt64()).ToArray();

        int compressed = root.GetProperty("compressed_vocab").GetInt32();
        foreach (int id in tokenMap)
        {
            if (id < 0 || id >= compressed)
                throw new HartsyInferenceException($"Engram token map holds id {id}, outside the compressed vocab of {compressed}.");
        }
        // The hasher uses C# '%', which is Python's only for a non-negative dividend: every id * multiplier product must be
        // non-negative and fit in int64 (then their XOR is non-negative too), and every prime a positive divisor.
        foreach (long multiplier in multipliers)
        {
            if (multiplier <= 0 || multiplier > long.MaxValue / Math.Max(compressed, 1))
                throw new HartsyInferenceException($"Engram multiplier {multiplier} can overflow int64 against {compressed} compressed ids.");
        }
        foreach (long prime in primes)
        {
            if (prime <= 0)
                throw new HartsyInferenceException($"Engram prime {prime} is not a positive divisor.");
        }
        for (int layer = 0; layer < layers; layer++)
        {
            long sum = 0;
            for (int c = 0; c < columns; c++)
            {
                if (offsets[layer * columns + c] != sum)
                    throw new HartsyInferenceException($"Engram offsets of layer {layerIds[layer]} are not the running sum of its primes at column {c}.");
                sum += primes[layer * columns + c];
            }
            if (sum != tableRows[layer])
                throw new HartsyInferenceException($"Engram primes of layer {layerIds[layer]} sum to {sum}, not its {tableRows[layer]} table rows.");
        }

        return new EngramConstants(root.GetProperty("checkpoint").GetString()!, root.GetProperty("checkpoint_revision").GetString()!,
            layerIds, maxNgram, heads, compressed, root.GetProperty("pad_compressed_id").GetInt32(), tableRows, tokenMap,
            multipliers, offsets, primes);
    }

    private static int[] ReadInt32s(Assembly assembly, JsonElement files, string name, int expected)
    {
        byte[] bytes = ReadVerified(assembly, files, name);
        if (bytes.Length != expected * sizeof(int))
            throw new HartsyInferenceException($"Engram fixture '{name}' is {bytes.Length} bytes, expected {expected * sizeof(int)}.");
        int[] values = new int[expected];
        for (int i = 0; i < values.Length; i++)
            values[i] = BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(i * sizeof(int)));
        return values;
    }

    private static long[] ReadInt64s(Assembly assembly, JsonElement files, string name, int expected)
    {
        byte[] bytes = ReadVerified(assembly, files, name);
        if (bytes.Length != expected * sizeof(long))
            throw new HartsyInferenceException($"Engram fixture '{name}' is {bytes.Length} bytes, expected {expected * sizeof(long)}.");
        long[] values = new long[expected];
        for (int i = 0; i < values.Length; i++)
            values[i] = BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(i * sizeof(long)));
        return values;
    }

    private static byte[] ReadVerified(Assembly assembly, JsonElement files, string name)
    {
        byte[] bytes = ReadResource(assembly, name);
        string actual = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        string pinned = files.GetProperty(name).GetProperty("sha256").GetString()!;
        if (!string.Equals(actual, pinned, StringComparison.Ordinal))
            throw new HartsyInferenceException($"Engram fixture '{name}' has SHA-256 {actual}, but its manifest pins {pinned}.");
        return bytes;
    }

    private static byte[] ReadResource(Assembly assembly, string name)
    {
        using Stream? stream = assembly.GetManifestResourceStream(ResourcePrefix + name);
        if (stream is null)
            throw new HartsyInferenceException($"Engram fixture '{name}' is not embedded in {assembly.GetName().Name}.");
        using MemoryStream copy = new();
        stream.CopyTo(copy);
        return copy.ToArray();
    }
}

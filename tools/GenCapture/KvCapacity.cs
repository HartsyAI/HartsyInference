using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.GenCapture;

/// <summary>KV-cache geometry of the model whose KV pool a batched run used.</summary>
internal sealed record KvGeometry(int Layers, int KvHeads, int HeadDim, string Architecture, string ModelSha256);

/// <summary>Batch KV capacity. A page figure exists only when the geometry comes from an authoritative, identifiable
/// source: the model's GGUF header, identified by its SHA-256 and read with the keys the engine's GGUF loader uses
/// (<c>{arch}.block_count</c>, <c>{arch}.attention.head_count_kv</c>, <c>{arch}.attention.key_length</c>). Pages come from the
/// engine's own formula, <see cref="PagedKvPool.PageCountForBudget"/>. Observed payload events are never a source: the
/// <c>paged.keys</c> events carry no KV geometry (KvHeads and QkHeadDim are 0). When the geometry cannot be established the
/// report states the reason and writes no page figure, neither zero nor negative.</summary>
internal sealed class KvCapacity
{
    private KvCapacity(bool available, string? reason, KvGeometry? geometry, long poolBudgetBytes, int pageSize,
        int poolPages, int pagesPerRequest)
    {
        Available = available;
        UnavailableReason = reason;
        Geometry = geometry;
        PoolBudgetBytes = poolBudgetBytes;
        PageSize = pageSize;
        PoolPages = poolPages;
        PagesPerRequest = pagesPerRequest;
    }

    public bool Available { get; }
    public string? UnavailableReason { get; }
    public KvGeometry? Geometry { get; }
    public long PoolBudgetBytes { get; }
    public int PageSize { get; }
    public int PoolPages { get; }
    public int PagesPerRequest { get; }
    public int PagesTwoRequests => 2 * PagesPerRequest;
    public int SparePagesAfterTwoRequests => PoolPages - PagesTwoRequests;

    /// <summary>Capacity that cannot be established. It carries the reason and no page figure.</summary>
    public static KvCapacity Unavailable(string reason, long poolBudgetBytes, int pageSize) =>
        new(false, reason, null, poolBudgetBytes, pageSize, 0, 0);

    /// <summary>Capacity for known geometry. Uses the engine's page formula and the same request-page arithmetic as the run.</summary>
    public static KvCapacity Compute(KvGeometry geometry, long poolBudgetBytes, int pageSize, long promptTokens, int generatedTokens)
    {
        if (geometry.Layers <= 0 || geometry.KvHeads <= 0 || geometry.HeadDim <= 0)
            return Unavailable("KV geometry is not positive", poolBudgetBytes, pageSize);
        if (poolBudgetBytes <= 0 || pageSize <= 0)
            return Unavailable("pool budget or page size is not positive", poolBudgetBytes, pageSize);
        int[] headDimPerLayer = Enumerable.Repeat(geometry.HeadDim, geometry.Layers).ToArray();
        int pool = PagedKvPool.PageCountForBudget(geometry.KvHeads, headDimPerLayer, pageSize, poolBudgetBytes);
        long perRequest = (promptTokens + generatedTokens + 1 + pageSize - 1) / pageSize;
        return new KvCapacity(true, null, geometry, poolBudgetBytes, pageSize, pool, checked((int)perRequest));
    }

    /// <summary>Why observed payload events are not a geometry source. Reported with the capacity, never used as one.</summary>
    public static string ObservedEventsReason(AttentionDecodeProof proof) =>
        "paged.keys payload events carry KvHeads=" + proof.PagedKvHeads.ToString(System.Globalization.CultureInfo.InvariantCulture)
        + " and QkHeadDim=" + proof.PagedHeadDim.ToString(System.Globalization.CultureInfo.InvariantCulture)
        + "; payload events are not a geometry source";

    /// <summary>Reads the KV geometry from the model's GGUF header. The file must match <paramref name="expectedSha256"/>.
    /// Mirrors the keys in <c>GgufLanguageModel</c>: head_count defaults the KV head count, key_length defaults to
    /// embedding_length / head_count.</summary>
    public static (KvGeometry? Geometry, string? Reason) ReadGgufGeometry(string modelPath, string expectedSha256)
    {
        if (!File.Exists(modelPath)) return (null, "model file is not present at the recorded path");
        if (Sha256Hex(modelPath) != expectedSha256) return (null, "model file SHA-256 differs from the recorded model_sha256");
        using GgufLoader loader = new();
        loader.Load(modelPath);
        GgufMetadata meta = loader.Metadata;
        string? arch = meta.GetString("general.architecture");
        if (string.IsNullOrEmpty(arch)) return (null, "GGUF header has no general.architecture");
        int layers = meta.GetInteger(arch + ".block_count");
        if (layers <= 0) return (null, "GGUF header has no positive " + arch + ".block_count");
        int heads = meta.GetInteger(arch + ".attention.head_count");
        int kvHeads = meta.GetInteger(arch + ".attention.head_count_kv", heads);
        if (kvHeads <= 0) return (null, "GGUF header gives no KV head count");
        int headDim = meta.GetInteger(arch + ".attention.key_length");
        if (headDim <= 0 && heads > 0) headDim = meta.GetInteger(arch + ".embedding_length") / heads;
        if (headDim <= 0) return (null, "GGUF header gives no key length");
        return (new KvGeometry(layers, kvHeads, headDim, arch, expectedSha256), null);
    }

    /// <summary>Creates <paramref name="path"/> and refuses an existing file. Derived reports are write-once.</summary>
    public static FileStream CreateFreshOutput(string path) =>
        new(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);

    public static string Sha256Hex(string path)
    {
        using FileStream input = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(input)).ToLowerInvariant();
    }

    /// <summary>Writes the capacity object. The caller writes the property name.</summary>
    public void WriteJson(Utf8JsonWriter writer)
    {
        writer.WriteStartObject();
        writer.WriteString("status", Available ? "available" : "unavailable");
        writer.WriteNumber("pool_budget_bytes", PoolBudgetBytes);
        writer.WriteNumber("page_size", PageSize);
        if (!Available)
        {
            writer.WriteString("reason", UnavailableReason ?? "unknown");
        }
        else if (Geometry is { } geometry)
        {
            writer.WriteStartObject("geometry");
            writer.WriteNumber("layers", geometry.Layers);
            writer.WriteNumber("kv_heads", geometry.KvHeads);
            writer.WriteNumber("head_dim", geometry.HeadDim);
            writer.WriteNumber("page_element_bytes", sizeof(float));
            writer.WriteEndObject();
            writer.WriteStartObject("source");
            writer.WriteString("kind", "gguf-header");
            writer.WriteString("architecture", geometry.Architecture);
            writer.WriteString("model_sha256", geometry.ModelSha256);
            writer.WriteString("keys", geometry.Architecture + ".block_count, " + geometry.Architecture
                + ".attention.head_count_kv, " + geometry.Architecture + ".attention.key_length");
            writer.WriteString("page_count_formula", "HartsyInference.LLM.Transformer.PagedKvPool.PageCountForBudget");
            writer.WriteEndObject();
            writer.WriteNumber("pool_pages", PoolPages);
            writer.WriteNumber("pages_per_request", PagesPerRequest);
            writer.WriteNumber("pages_two_requests", PagesTwoRequests);
            writer.WriteNumber("spare_pages_after_two_requests", SparePagesAfterTwoRequests);
        }
        writer.WriteEndObject();
    }
}

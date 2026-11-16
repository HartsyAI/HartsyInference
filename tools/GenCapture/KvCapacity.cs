using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.GenCapture;

/// <summary>KV-cache geometry of the model whose KV pool a batched run used: the head dimension of each decoder layer, as the
/// engine sizes its GGUF pool (<c>CreateGgufScheduler</c> reads <c>HeadDimFor</c> per layer), and the KV head count.</summary>
internal sealed record KvGeometry(int[] HeadDimPerLayer, int KvHeads, string Architecture, string ModelSha256)
{
    /// <summary>Decoder layers, counted as the engine counts them: next-token-prediction layers are excluded.</summary>
    public int Layers => HeadDimPerLayer.Length;

    /// <summary>True when every layer has the same head dimension.</summary>
    public bool UniformHeadDim => HeadDimPerLayer.Length > 0 && HeadDimPerLayer.All(dim => dim == HeadDimPerLayer[0]);

    /// <summary>Geometry with one head dimension on every layer.</summary>
    public static KvGeometry Uniform(int layers, int kvHeads, int headDim, string architecture, string modelSha256) =>
        new(Enumerable.Repeat(headDim, Math.Max(layers, 0)).ToArray(), kvHeads, architecture, modelSha256);
}

/// <summary>Batch KV capacity. A page figure exists only when the geometry comes from the model's GGUF header, read by the
/// engine's own config reader (<c>GgufConfigFactory.FromGguf</c>) for an architecture this derive has been verified against.
/// Pages come from the engine's own formula, <see cref="PagedKvPool.PageCountForBudget"/>. Observed payload events are never a
/// source: the <c>paged.keys</c> events carry no KV geometry (KvHeads and QkHeadDim are 0). When the geometry cannot be
/// established the report states the reason and writes no page figure, neither zero nor negative.</summary>
internal sealed class KvCapacity
{
    /// <summary>Architectures whose GGUF layout has been checked against a recorded run. Any other is reported unavailable.</summary>
    private static readonly string[] VerifiedArchitectures = ["qwen3"];

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

    /// <summary>Capacity for known geometry. Uses the engine's page formula, per layer, and the same request-page arithmetic as the run.</summary>
    public static KvCapacity Compute(KvGeometry geometry, long poolBudgetBytes, int pageSize, long promptTokens, int generatedTokens)
    {
        if (geometry.Layers <= 0 || geometry.KvHeads <= 0 || geometry.HeadDimPerLayer.Any(dim => dim <= 0))
            return Unavailable("KV geometry is not positive", poolBudgetBytes, pageSize);
        if (poolBudgetBytes <= 0 || pageSize <= 0)
            return Unavailable("pool budget or page size is not positive", poolBudgetBytes, pageSize);
        int pool = PagedKvPool.PageCountForBudget(geometry.KvHeads, geometry.HeadDimPerLayer, pageSize, poolBudgetBytes);
        long perRequest = (promptTokens + generatedTokens + 1 + pageSize - 1) / pageSize;
        return new KvCapacity(true, null, geometry, poolBudgetBytes, pageSize, pool, checked((int)perRequest));
    }

    /// <summary>Why observed payload events are not a geometry source. Reported with the capacity, never used as one.</summary>
    public static string ObservedEventsReason(AttentionDecodeProof proof) =>
        "paged.keys payload events carry KvHeads=" + proof.PagedKvHeads.ToString(System.Globalization.CultureInfo.InvariantCulture)
        + " and QkHeadDim=" + proof.PagedHeadDim.ToString(System.Globalization.CultureInfo.InvariantCulture)
        + "; payload events are not a geometry source";

    /// <summary>Reads the KV geometry from the model's GGUF header through the engine's own config reader. The file must match
    /// <paramref name="expectedSha256"/>. The architecture is checked before the reader runs, so no other model class's
    /// loader path is exercised. Nothing throws: any failure is returned as the reason, so a caller's write-once output is
    /// always completed.</summary>
    public static (KvGeometry? Geometry, string? Reason) ReadGgufGeometry(string modelPath, string expectedSha256)
    {
        try
        {
            if (!File.Exists(modelPath)) return (null, "model file is not present at the recorded path");
            if (Sha256Hex(modelPath) != expectedSha256) return (null, "model file SHA-256 differs from the recorded model_sha256");
            using GgufModelLoader.LoadedGgufModel model = GgufModelLoader.Load(modelPath);
            string? declared = model.Metadata.GetString("general.architecture");
            if (declared is null || Array.IndexOf(VerifiedArchitectures, declared) < 0)
                return (null, $"architecture '{declared ?? "(none)"}' has no verified KV capacity reader (verified: {string.Join(", ", VerifiedArchitectures)})");
            TransformerConfig config = GgufConfigFactory.FromGguf(model.Metadata, model.Weights);
            return FromEngineConfig(config, declared, expectedSha256);
        }
        catch (Exception error)
        {
            return (null, $"engine GGUF reader refused the header ({error.GetType().Name}): {error.Message}");
        }
    }

    /// <summary>The geometry the engine's pool is sized from, read from its config: one head dimension per decoder layer and the
    /// KV head count. A layout whose cache is not a plain per-layer K/V pool is reported with its reason, not sized.</summary>
    public static (KvGeometry? Geometry, string? Reason) FromEngineConfig(TransformerConfig config, string architecture, string modelSha256)
    {
        if (config.Mla is not null)
            return (null, "multi-head latent attention keeps a latent KV cache, which this derive does not model");
        if (config.CrossAttnLayers is { Count: > 0 })
            return (null, "cross-attention layers keep their own cache, which this derive does not model");
        if (config.NumLayers <= 0) return (null, "engine config reports no decoder layers");
        if (config.NumKvHeads <= 0) return (null, "engine config reports no KV heads");
        int[] headDims = config.HeadDimsPerLayer();
        if (headDims.Any(dim => dim <= 0)) return (null, "engine config reports a non-positive head dimension");
        return (new KvGeometry(headDims, config.NumKvHeads, architecture, modelSha256), null);
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
            if (geometry.UniformHeadDim)
            {
                writer.WriteNumber("head_dim", geometry.HeadDimPerLayer[0]);
            }
            else
            {
                writer.WriteStartArray("head_dim_per_layer");
                foreach (int dim in geometry.HeadDimPerLayer) writer.WriteNumberValue(dim);
                writer.WriteEndArray();
            }
            writer.WriteNumber("page_element_bytes", sizeof(float));
            writer.WriteEndObject();
            writer.WriteStartObject("source");
            writer.WriteString("kind", "gguf-header");
            writer.WriteString("reader", "HartsyInference.LLM.Transformer.GgufConfigFactory.FromGguf");
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

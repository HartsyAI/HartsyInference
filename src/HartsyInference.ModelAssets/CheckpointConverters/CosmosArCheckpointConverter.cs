using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads Cosmos-Predict1 Video2World <c>model.pt</c> (a BF16 PyTorch pickle; the 5B is 9.17 GB, the 13B
/// 26.6 GB — no safetensors ship) into the state-dict names
/// <see cref="HartsyInference.Video.Models.Cosmos.CosmosArTransformer"/> expects: <c>tok_embeddings.weight</c>,
/// <c>layers.{i}.attention.{wq,wk,wv,wo,q_norm,k_norm}.weight</c>,
/// <c>layers.{i}.cross_attention.{wq,wk,wv,wo}.weight</c>, <c>layers.{i}.cross_attention_norm.weight</c>,
/// <c>layers.{i}.attention_norm.weight</c>, <c>layers.{i}.ffn_norm.weight</c>,
/// <c>layers.{i}.feed_forward.{w1,w2,w3}.weight</c>, <c>norm.weight</c>, <c>output.weight</c>, and (V2W)
/// <c>pos_emb.weight</c>.
///
/// <para>The expected keys already match Cosmos's own module hierarchy (research-doc §Data Layouts), so conversion
/// is a prefix strip plus a small rename table for the variations a real dump may show. The exact source keys are
/// research-doc Open Q §1 (the pickle is too large for the HF web viewer) — run the key-dump snippet in that
/// section against a real <c>model.pt</c> and extend <see cref="Renames"/> if any key falls outside the strip.</para>
///
/// <para>Reuses <see cref="PytorchPickleLoader"/> for the <c>.pt</c> pickle — no off-ship Python conversion.</para></summary>
public static class CosmosArCheckpointConverter
{
    // Leading module wrappers a real dump may carry before the transformer's own keys.
    private static readonly string[] StripPrefixes =
    [
        "model.model.", "model.", "module.", "net.", "network.",
    ];

    // Key-tail renames for likely naming variations (extend once the real dump is inspected — Open Q §1).
    private static readonly (string From, string To)[] Renames =
    [
        ("tok_embeddings.weight", "tok_embeddings.weight"),
        ("embeddings.weight", "tok_embeddings.weight"),
        ("token_embedding.weight", "tok_embeddings.weight"),
        (".attention.wqkv.", ".attention.wqkv."),   // fused (fuse_qkv) — left as-is; split is an integration step
    ];

    /// <summary>Reads <paramref name="path"/> (<c>.pt</c>/<c>.pth</c>/<c>.bin</c> pickle or <c>.safetensors</c>) and
    /// returns the Cosmos-named weights plus the backing loader (dispose after preload). <paramref name="castToF32"/>
    /// forces every tensor to F32 (CPU / structural runs); leave false to keep BF16 for GPU residency.</summary>
    public static (Dictionary<string, Tensor> Weights, IDisposable Loader) LoadAndConvert(string path, bool castToF32 = false)
    {
        (Dictionary<string, Tensor> raw, IDisposable loader) = ReadRaw(path);
        Dictionary<string, Tensor> mapped = new(raw.Count);
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            string key = StripPrefix(kv.Key);
            key = ApplyRenames(key);
            Tensor value = kv.Value;
            if (castToF32 && value.DType != DType.F32 && !value.DType.IsQuantized)
                value = value.CastTo(DType.F32);
            mapped[key] = value;
        }
        return (mapped, loader);
    }

    /// <summary>Maps a single source key to its Cosmos name (prefix strip + rename table). Exposed for tests.</summary>
    public static string MapKey(string key) => ApplyRenames(StripPrefix(key));

    private static string StripPrefix(string key)
    {
        foreach (string p in StripPrefixes)
            if (key.StartsWith(p, StringComparison.Ordinal))
                return key.Substring(p.Length);
        return key;
    }

    private static string ApplyRenames(string key)
    {
        foreach ((string from, string to) in Renames)
            if (from != to && key.EndsWith(from, StringComparison.Ordinal))
                return key.Substring(0, key.Length - from.Length) + to;
        // Whole-key renames (embedding aliases).
        foreach ((string from, string to) in Renames)
            if (from != to && key == from)
                return to;
        return key;
    }

    private static (Dictionary<string, Tensor> Raw, IDisposable Loader) ReadRaw(string path)
    {
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            SafeTensorsLoader st = new();
            st.Load(path);
            return (st.GetAllTensors(), st);
        }
        PytorchPickleLoader pt = new();
        pt.Load(path);
        return (pt.GetAllTensors(), pt);
    }
}

using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Looped;

/// <summary>Builds a <see cref="LoopedTransformerConfig"/> from the metadata llama.cpp's Dory converter writes (<c>dory.*</c> keys: the three block
/// sizes, <c>num_loops</c>, the <c>S*-</c> layer pattern, a per-layer RoPE profile selecting one of two base frequencies, the cache mode and the
/// nGPT init constants).</summary>
public static class LoopedConfigFactory
{
    /// <summary>GGUF architectures this factory reads.</summary>
    public static readonly IReadOnlySet<string> Architectures = new HashSet<string>(StringComparer.Ordinal) { "dory" };

    /// <summary>Reads <paramref name="metadata"/> (and the embedding table in <paramref name="weights"/>, for the vocabulary) into a validated config.</summary>
    public static LoopedTransformerConfig FromGguf(GgufMetadata metadata, IReadOnlyDictionary<string, Tensor> weights, bool lowVramQuant = false)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        ArgumentNullException.ThrowIfNull(weights);
        string arch = metadata.GetString("general.architecture") ?? "";
        if (!Architectures.Contains(arch))
            throw new ArgumentException($"'{arch}' is not a looped (latent-recurrent) architecture this engine reads.", nameof(metadata));

        int layers = RequireUInt(metadata, $"{arch}.block_count");
        int hidden = RequireUInt(metadata, $"{arch}.embedding_length");
        int heads = RequireUInt(metadata, $"{arch}.attention.head_count");
        int kvHeads = (int)metadata.GetUInt32($"{arch}.attention.head_count_kv", (uint)heads);
        int intermediate = RequireUInt(metadata, $"{arch}.feed_forward_length");
        int headDim = RequireUInt(metadata, $"{arch}.attention.key_length");
        int valueDim = (int)metadata.GetUInt32($"{arch}.attention.value_length", (uint)headDim);
        int ropeDim = (int)metadata.GetUInt32($"{arch}.rope.dimension_count", (uint)headDim);
        if (valueDim != headDim || ropeDim != headDim)
            throw new NotSupportedException($"'{arch}' needs value_length and rope.dimension_count equal to key_length ({headDim}); got {valueDim} and {ropeDim}.");
        int maxPos = (int)metadata.GetUInt32($"{arch}.context_length", 8192u);
        float theta = metadata.GetFloat32($"{arch}.rope.freq_base", 10_000f);
        float theta2 = metadata.GetFloat32($"{arch}.rope.freq_base_2", theta);
        int window = (int)metadata.GetUInt32($"{arch}.attention.sliding_window", 0u);
        int inputLayers = RequireUInt(metadata, $"{arch}.input_layer_count");
        int recurrentLayers = RequireUInt(metadata, $"{arch}.recurrent_layer_count");
        int outputLayers = RequireUInt(metadata, $"{arch}.output_layer_count");
        int loops = RequireUInt(metadata, $"{arch}.num_loops");
        string pattern = metadata.GetString($"{arch}.layer_pattern")
            ?? throw new ArgumentException($"GGUF metadata lacks {arch}.layer_pattern.", nameof(metadata));
        int[] profile = metadata.GetIntArray($"{arch}.rope.profile")
            ?? throw new ArgumentException($"GGUF metadata lacks {arch}.rope.profile.", nameof(metadata));
        string mode = metadata.GetString($"{arch}.recurrent_kv_cache_mode") ?? "per_loop";
        LoopedKvCacheMode kvMode = mode switch
        {
            "per_loop" => LoopedKvCacheMode.PerLoop,
            "last_loop" => LoopedKvCacheMode.LastLoop,
            _ => throw new NotSupportedException($"{arch}.recurrent_kv_cache_mode '{mode}' is not per_loop or last_loop."),
        };
        float initStd = metadata.GetFloat32($"{arch}.init_std", 0.02f);
        float alphaInit = metadata.GetFloat32($"{arch}.alpha_init", 0.05f);
        float gateInit = metadata.GetFloat32($"{arch}.gate_init", 0f);

        if (inputLayers + recurrentLayers + outputLayers != layers)
            throw new ArgumentException($"{arch} layer blocks {inputLayers}+{recurrentLayers}+{outputLayers} do not sum to block_count {layers}.", nameof(metadata));
        if (pattern.Length != layers || profile.Length != layers)
            throw new ArgumentException($"{arch}.layer_pattern ({pattern.Length}) and rope.profile ({profile.Length}) must have one entry per layer ({layers}).", nameof(metadata));
        LoopedLayerKind[] kinds = new LoopedLayerKind[layers];
        float[] thetas = new float[layers];
        for (int p = 0; p < layers; p++)
        {
            kinds[p] = pattern[p] switch
            {
                'S' => LoopedLayerKind.SlidingAttention,
                '*' => LoopedLayerKind.FullAttention,
                '-' => LoopedLayerKind.Mlp,
                _ => throw new ArgumentException($"{arch}.layer_pattern character '{pattern[p]}' at layer {p} is not S, * or -.", nameof(metadata)),
            };
            thetas[p] = profile[p] switch
            {
                1 => theta,
                2 => theta2,
                _ => throw new ArgumentException($"{arch}.rope.profile value {profile[p]} at layer {p} is not 1 or 2.", nameof(metadata)),
            };
        }

        if (!weights.TryGetValue("model.embed_tokens.weight", out Tensor? embed))
            throw new ArgumentException("GGUF weights missing model.embed_tokens.weight (cannot infer vocab).", nameof(weights));
        int vocab = (int)(embed.ElementCount / hidden);

        LoopedTransformerConfig config = new()
        {
            HiddenSize = hidden,
            NumHeads = heads,
            NumKvHeads = kvHeads,
            HeadDim = headDim,
            IntermediateSize = intermediate,
            VocabSize = vocab,
            MaxPositionEmbeddings = maxPos,
            InputLayers = inputLayers,
            RecurrentLayers = recurrentLayers,
            OutputLayers = outputLayers,
            Loops = loops,
            LayerKinds = kinds,
            RopeThetas = thetas,
            SlidingWindow = window,
            KvCacheMode = kvMode,
            InitStd = initStd,
            AlphaInit = alphaInit,
            GateInit = gateInit,
            Architecture = arch,
            LowVramQuant = lowVramQuant,
        };
        config.Validate();
        return config;
    }

    private static int RequireUInt(GgufMetadata metadata, string key)
    {
        if (!metadata.ContainsKey(key))
            throw new ArgumentException($"GGUF metadata lacks required key {key}.", nameof(metadata));
        return (int)metadata.GetUInt32(key);
    }
}

namespace HartsyInference.ModelHandler.Gguf.KeyMappers;

/// <summary>Final-fallback mapper that returns every tensor name unchanged. Selected when neither metadata nor key heuristic finds a match. Useful for novel GGUFs that happen to ship with diffusers naming already.</summary>
public sealed class PassthroughKeyMapper : IGgufKeyMapper
{
    public string Architecture => "passthrough";

    // Encoder-family GGUFs (BERT embedding models) keep their own verbatim tensor names (token_embd, blk.N.attn_q,
    // *_norm, position_embd, token_types). Declaring them here resolves the arch to passthrough instead of letting the
    // llama key-heuristic mangle them. (BertEmbeddingModel reads these names directly.)
    public IReadOnlyCollection<string> Architectures => ["passthrough", "bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "neo-bert", "mamba", "mamba2", "rwkv6", "rwkv7", "t5", "t5encoder"];

    public bool MatchesByKeys(IEnumerable<string> tensorNames) => true;
    public string? MapKey(string ggufKey) => ggufKey;
}

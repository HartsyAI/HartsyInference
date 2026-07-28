namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>Final-fallback mapper that returns every tensor name unchanged. Selected when neither metadata nor key heuristic finds a match. Useful for novel GGUFs that happen to ship with diffusers naming already.</summary>
public sealed class PassthroughKeyMapper : IGgufKeyMapper
{
    public string Architecture => "passthrough";

    // Encoder-family GGUFs (BERT embedding models) keep their own verbatim tensor names (token_embd, blk.N.attn_q,
    // *_norm, position_embd, token_types). Declaring them here resolves the arch to passthrough instead of letting the
    // llama key-heuristic mangle them. (BertEmbeddingModel reads these names directly.)
    // qwen35/qwen35moe (Gated DeltaNet hybrid): also has blk.N.* + token_embd.weight, which would satisfy
    // LlamaKeyMapper's heuristic (hasBlk && hasTokenEmbd) if this arch weren't registered explicitly here —
    // Qwen35Model reads raw GGUF names directly (blk.N.attn_qkv.weight etc), same as the SSM models above.
    // clip: llama.cpp vision sidecars (mmproj-*.gguf) declare general.architecture="clip" and keep verbatim
    // v.blk.N.*/v.patch_embd/mm.* names that every IVlmImageEncoder (Siglip/Qwen2.5-VL/Qwen3-VL/…) reads directly.
    // Without this, the heuristic misfires — e.g. Qwen3-VL's fused v.blk.N.attn_qkv makes PhiKeyMapper claim the file
    // and drop the vision tensors (mmproj then fails to load, model silently degrades to text-only).
    public IReadOnlyCollection<string> Architectures => ["passthrough", "bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "neo-bert", "mamba", "mamba2", "rwkv6", "rwkv7", "t5", "t5encoder", "qwen35", "qwen35moe", "clip"];

    public bool MatchesByKeys(IEnumerable<string> tensorNames) => true;
    public string? MapKey(string ggufKey) => ggufKey;
}

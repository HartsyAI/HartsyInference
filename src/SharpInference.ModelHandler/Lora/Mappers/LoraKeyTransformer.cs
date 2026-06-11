namespace SharpInference.ModelHandler.Lora.Mappers;

/// <summary>Shared helpers for converting Kohya-style underscored LoRA module paths into canonical dotted weight-dictionary keys, while preserving compound identifiers (e.g. "down_blocks", "to_q") whose internal underscores must survive the transformation.</summary>
public static class LoraKeyTransformer
{
    // Each compound identifier maps to a placeholder using a sentinel byte (\x01) that cannot appear in real
    // safetensors keys. Order matters: longer/superset substrings come first so they bind before their
    // substrings (e.g. single_transformer_blocks must replace before transformer_blocks; final_layer_norm
    // before layer_norm).
    private static readonly (string Original, string Placeholder)[] _protected =
    [
        ("single_transformer_blocks", "\x01STBLOCKS\x01"),
        ("transformer_blocks", "\x01TBLOCKS\x01"),
        ("down_blocks", "\x01DBLOCKS\x01"),
        ("up_blocks", "\x01UBLOCKS\x01"),
        ("mid_block", "\x01MBLOCK\x01"),
        ("text_model", "\x01TEXTMODEL\x01"),
        ("self_attn", "\x01SELFATTN\x01"),
        ("cross_attn", "\x01CROSSATTN\x01"),
        ("final_layer_norm", "\x01FLN\x01"),
        ("layer_norm1", "\x01LN1\x01"),
        ("layer_norm2", "\x01LN2\x01"),
        ("layer_norm", "\x01LN\x01"),
        ("position_embedding", "\x01POSEMB\x01"),
        ("token_embedding", "\x01TOKEMB\x01"),
        ("time_embedding", "\x01TIMEMB\x01"),
        ("add_q_proj", "\x01ADDQP\x01"),
        ("add_k_proj", "\x01ADDKP\x01"),
        ("add_v_proj", "\x01ADDVP\x01"),
        ("to_add_out", "\x01TOADDOUT\x01"),
        ("norm_added_q", "\x01NORMADDQ\x01"),
        ("norm_added_k", "\x01NORMADDK\x01"),
        ("norm_k_img", "\x01NORMKIMG\x01"),   // Wan I2V image-KV norm — must bind before norm_k / k_img
        ("norm_q", "\x01NORMQ\x01"),
        ("norm_k", "\x01NORMK\x01"),
        ("k_img", "\x01KIMG\x01"),            // Wan I2V image K/V projections
        ("v_img", "\x01VIMG\x01"),
        ("text_embedding", "\x01TXTEMB\x01"), // Wan top-level condition embedders
        ("time_projection", "\x01TIMEPROJ\x01"),
        ("patch_embedding", "\x01PATCHEMB\x01"),
        ("q_proj", "\x01QPROJ\x01"),
        ("k_proj", "\x01KPROJ\x01"),
        ("v_proj", "\x01VPROJ\x01"),
        ("out_proj", "\x01OUTPROJ\x01"),
        ("to_q", "\x01TOQ\x01"),
        ("to_k", "\x01TOK\x01"),
        ("to_v", "\x01TOV\x01"),
        ("to_out", "\x01TOOUT\x01"),
        ("proj_in", "\x01PROJIN\x01"),
        ("proj_out", "\x01PROJOUT\x01"),
        ("proj_mlp", "\x01PROJMLP\x01"),
        ("ff_context", "\x01FFCTX\x01"),
        ("time_text_embed", "\x01TTE\x01"),
        ("timestep_embedder", "\x01TSE\x01"),
        ("text_embedder", "\x01TXE\x01"),
        ("guidance_embedder", "\x01GE\x01"),
        ("x_embedder", "\x01XEMB\x01"),
        ("context_embedder", "\x01CTXEMB\x01"),
        ("linear_1", "\x01LIN1\x01"),
        ("linear_2", "\x01LIN2\x01"),
    ];

    /// <summary>Transforms a Kohya-style underscored module path body (with prefix and suffix already stripped) into a dotted canonical path. Compound identifiers are preserved via placeholder substitution before the underscore→dot pass.</summary>
    public static string UnderscoreToDot(string body)
    {
        string s = body;
        foreach ((string original, string placeholder) in _protected)
        {
            s = s.Replace(original, placeholder);
        }
        s = s.Replace('_', '.');
        foreach ((string original, string placeholder) in _protected)
        {
            s = s.Replace(placeholder, original);
        }
        return s;
    }
}

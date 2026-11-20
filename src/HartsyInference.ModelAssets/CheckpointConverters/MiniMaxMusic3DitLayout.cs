namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>The tensor layout of a user-supplied MiniMax-Music-3 flow-transformer checkpoint, decided from its tensor names alone.</summary>
public enum MiniMaxMusic3DitLayout
{
    /// <summary>Matches no layout the engine reads.</summary>
    Unrecognized,

    /// <summary>Comfy-Org single file: <c>diffusion_transformer.*</c> with fused QKV (fp32, fp16, or int8 ConvRot).</summary>
    ComfyDit,

    /// <summary>The official diffusers <c>transformer/</c> weights (<c>transformer_blocks.N.attn.to_q…</c>), also the Hartsy repack and GGUF builds that keep these names.</summary>
    DiffusersDit,

    /// <summary>The diffusers transformer and the condition encoder (<c>layer_weight_logits</c>, <c>proj.*</c>) in one file.</summary>
    DiffusersDitWithConditionEncoder,

    /// <summary>The condition encoder alone — it is not the flow transformer a caller selects.</summary>
    ConditionEncoderOnly,

    /// <summary>A llama.cpp-style fused-QKV build (<c>blk.N.attn_qkv</c>) made for a native C++ runtime; the engine has no mapping for it.</summary>
    FusedQkvCppDit,
}

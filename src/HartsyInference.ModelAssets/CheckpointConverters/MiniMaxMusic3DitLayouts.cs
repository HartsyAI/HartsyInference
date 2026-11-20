namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Recognizes which <see cref="MiniMaxMusic3DitLayout"/> a checkpoint's tensor names describe, so a file is judged by what it holds and never by what it is called.</summary>
public static class MiniMaxMusic3DitLayouts
{
    private const string ComfyMarker = "diffusion_transformer.transformer.layers.0.self_attn.to_qkv.weight";

    private static readonly string[] DiffusersMarkers =
    [
        "transformer_blocks.0.attn.to_q.weight", "proj_in.weight", "proj_out.weight",
        "time_embed.linear_1.weight", "preprocess_conv.weight",
    ];

    private static readonly string[] ConditionMarkers = ["layer_weight_logits", "layer_scale", "proj.weight", "proj.bias"];

    /// <summary>The layout whose marker tensors are all present in <paramref name="keys"/>.</summary>
    public static MiniMaxMusic3DitLayout Detect(IEnumerable<string> keys)
    {
        ArgumentNullException.ThrowIfNull(keys);
        HashSet<string> present = new HashSet<string>(keys, StringComparer.Ordinal);
        if (present.Contains(ComfyMarker))
        {
            return MiniMaxMusic3DitLayout.ComfyDit;
        }
        bool condition = HasAll(present, ConditionMarkers);
        if (HasAll(present, DiffusersMarkers))
        {
            return condition ? MiniMaxMusic3DitLayout.DiffusersDitWithConditionEncoder : MiniMaxMusic3DitLayout.DiffusersDit;
        }
        if (condition && !present.Any(key => key.StartsWith("transformer_blocks.", StringComparison.Ordinal)))
        {
            return MiniMaxMusic3DitLayout.ConditionEncoderOnly;
        }
        return present.Any(key => key.Contains("blk.", StringComparison.Ordinal) && key.Contains(".attn_qkv.", StringComparison.Ordinal))
            ? MiniMaxMusic3DitLayout.FusedQkvCppDit : MiniMaxMusic3DitLayout.Unrecognized;
    }

    /// <summary>One line for an error message: what the file is and whether the engine reads it.</summary>
    public static string Describe(MiniMaxMusic3DitLayout layout) => layout switch
    {
        MiniMaxMusic3DitLayout.ComfyDit => "a Comfy-Org single-file DiT",
        MiniMaxMusic3DitLayout.DiffusersDit => "the diffusers flow transformer",
        MiniMaxMusic3DitLayout.DiffusersDitWithConditionEncoder => "the diffusers flow transformer plus the condition encoder",
        MiniMaxMusic3DitLayout.ConditionEncoderOnly => "the condition encoder alone, which is not the flow transformer",
        MiniMaxMusic3DitLayout.FusedQkvCppDit => "a fused-QKV build for a native C++ runtime (blk.N.attn_qkv), which the engine does not map",
        _ => "an unrecognized tensor set",
    };

    private static bool HasAll(HashSet<string> present, string[] markers)
    {
        foreach (string marker in markers)
        {
            if (!present.Contains(marker))
            {
                return false;
            }
        }
        return true;
    }
}

namespace HartsyInference.ModelAssets.Duo;

/// <summary>Maps the diffusers-layout module names of a Duo Realization LoRA (<c>transformer_blocks.N.attn.to_q</c>) to the original-layout backbone weights (<c>blocks.N.attn.qkv_proj.weight</c>), the inverse of upstream's <c>convert_transformer_key</c>.
/// The row order it assumes is the engine's: the fused qkv as <c>[Q; K; V]</c> blocks (see <see cref="DuoQkvLayout"/>), <c>mlp.fc1</c> as <c>[gate; value]</c>.</summary>
public static class DuoLoraKeyMap
{
    public const string LoraASuffix = ".lora_A.default.weight";
    public const string LoraBSuffix = ".lora_B.default.weight";

    private static readonly (string Source, string Target, int Slot, int Slots, bool Swap)[] BlockModules =
    [
        ("attn.to_q", "attn.qkv_proj.weight", 0, 3, false),
        ("attn.to_k", "attn.qkv_proj.weight", 1, 3, false),
        ("attn.to_v", "attn.qkv_proj.weight", 2, 3, false),
        ("attn.to_out.0", "attn.out_proj.weight", 0, 1, false),
        ("ff.net.0.proj", "mlp.fc1.weight", 0, 1, true),
        ("ff.net.2", "mlp.fc2.weight", 0, 1, false),
    ];

    private static readonly (string Source, string Target)[] SingleModules =
    [
        ("proj_in", "video_patch_proj.weight"),
        ("audio_proj_in", "audio_patch_proj.weight"),
        ("context_embedder", "condition_proj.weight"),
        ("proj_out", "final_layer.video_out.weight"),
        ("audio_proj_out", "final_layer.audio_out.weight"),
    ];

    /// <summary>Resolves a module name (the key without its <c>.lora_A.default.weight</c> or <c>.lora_B.default.weight</c> suffix and any <c>.base_layer</c> segment).</summary>
    public static bool TryMap(string module, out DuoLoraTarget target)
    {
        ArgumentNullException.ThrowIfNull(module);
        module = module.Replace(".base_layer", string.Empty, StringComparison.Ordinal);
        foreach ((string source, string dest) in SingleModules)
        {
            if (module == source)
            {
                target = new DuoLoraTarget(dest, 0, 1, false);
                return true;
            }
        }

        if (TrySplitBlock(module, "transformer_blocks.", "blocks.", out string backbonePrefix, out string rest)
            || TrySplitBlock(module, "token_refiner.refiner_blocks.", "token_refiner.blocks.", out backbonePrefix, out rest))
        {
            foreach ((string source, string dest, int slot, int slots, bool swap) in BlockModules)
            {
                if (rest == source)
                {
                    target = new DuoLoraTarget($"{backbonePrefix}.{dest}", slot, slots, swap);
                    return true;
                }
            }
        }

        target = default;
        return false;
    }

    /// <summary>The module name of a LoRA tensor key and whether it is the A (down) or B (up) matrix, or null for any other key.</summary>
    public static string? ModuleOf(string key, out bool isA)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (key.EndsWith(LoraASuffix, StringComparison.Ordinal))
        {
            isA = true;
            return key[..^LoraASuffix.Length];
        }
        isA = false;
        return key.EndsWith(LoraBSuffix, StringComparison.Ordinal) ? key[..^LoraBSuffix.Length] : null;
    }

    private static bool TrySplitBlock(string module, string sourcePrefix, string targetPrefix, out string backbonePrefix, out string rest)
    {
        backbonePrefix = rest = string.Empty;
        if (!module.StartsWith(sourcePrefix, StringComparison.Ordinal)) return false;
        int dot = module.IndexOf('.', sourcePrefix.Length);
        if (dot < 0 || !int.TryParse(module.AsSpan(sourcePrefix.Length, dot - sourcePrefix.Length), out int block) || block < 0) return false;
        backbonePrefix = $"{targetPrefix}{block}";
        rest = module[(dot + 1)..];
        return true;
    }
}

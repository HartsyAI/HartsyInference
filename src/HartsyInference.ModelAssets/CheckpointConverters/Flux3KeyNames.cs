using System.Globalization;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>FLUX.3 module names in the three spellings in circulation, resolved to the BFL/ComfyUI names the engine
/// loads: diffusers (<c>transformer_flux3.py</c>; the inverse of diffusers'
/// <c>convert_flux3_transformer_checkpoint_to_diffusers</c>), kohya (<c>lora_unet_</c> with underscores for dots) and
/// the BFL names themselves. Module paths carry no <c>.weight</c> suffix.</summary>
public static class Flux3KeyNames
{
    /// <summary>The final adaLN projection. BFL rows are <c>[shift | scale]</c>; diffusers' <c>norm_out.linear</c> rows are <c>[scale | shift]</c>.</summary>
    public const string FinalAdaLn = "final_layer.image.adaLN_modulation.1";

    /// <summary>The fused attention projection leaf some BFL builds ship, rows <c>[q | k | v]</c>.</summary>
    public const string FusedQkvLeaf = "qkv";

    /// <summary>The split projection leaves the engine loads, in the fused weight's row order.</summary>
    public static IReadOnlyList<string> SplitQkvLeaves { get; } = ["q_proj", "k_proj", "v_proj"];

    /// <summary>Top-level module roots of the BFL layout.</summary>
    public static IReadOnlyList<string> BflRoots { get; } =
    [
        "emb_in.", "txt_in.", "time_in.", "ref_time_in.", "early_stream_modulations.", "single_stream_modulations.",
        "content_mode_blocks.", "txt_mode_blocks.", "single_blocks.", "final_layer.",
    ];

    /// <summary>BFL roots no other family uses. <c>single_blocks</c> is shared with Flux.1/.2 and is told apart by its leaves.</summary>
    private static readonly string[] DistinctiveBflRoots =
    [
        "emb_in.", "ref_time_in.", "early_stream_modulations.", "single_stream_modulations.", "content_mode_blocks.",
        "txt_mode_blocks.",
    ];

    /// <summary>Diffusers roots no other family uses. <c>single_transformer_blocks</c> is shared with Flux.1/.2.</summary>
    private static readonly string[] DistinctiveDiffusersRoots =
    [
        "x_embedders.", "target_timestep_embedder.", "early_stream_modulations.", "single_stream_modulations.",
        "mode_blocks.", "txt_mode_blocks.",
    ];

    /// <summary>Linear leaves of a BFL block. Flux.1/.2 joint blocks use <c>linear1</c>/<c>linear2</c> instead.</summary>
    private static readonly string[] BlockLeaves = ["q_proj", "k_proj", "v_proj", FusedQkvLeaf, "mlp_in", "mlp_out", "attn_out"];

    /// <summary>Every module leaf of a BFL block: the linears plus the Q/K norms.</summary>
    private static readonly string[] BlockModuleLeaves = [.. BlockLeaves, "norm.query_norm", "norm.key_norm"];

    private static readonly (string Diffusers, string Bfl)[] ExactMap =
    [
        ("context_embedder", "txt_in"),
        ("timestep_embedder.linear_1", "time_in.in_layer"),
        ("timestep_embedder.linear_2", "time_in.out_layer"),
        ("target_timestep_embedder.linear_1", "ref_time_in.in_layer"),
        ("target_timestep_embedder.linear_2", "ref_time_in.out_layer"),
        ("proj_out", "final_layer.image.linear"),
        ("norm_out.linear", FinalAdaLn),
    ];

    private static readonly (string Diffusers, string Bfl)[] PrefixMap =
    [
        ("x_embedders.", "emb_in."),
        ("mode_blocks.", "content_mode_blocks."),
        ("single_transformer_blocks.", "single_blocks."),
    ];

    private static readonly (string Diffusers, string Bfl)[] BlockSuffixMap =
    [
        (".attn.to_q", ".q_proj"),
        (".attn.to_k", ".k_proj"),
        (".attn.to_v", ".v_proj"),
        (".attn.to_out.0", ".attn_out"),
        (".attn.norm_q", ".norm.query_norm"),
        (".attn.norm_k", ".norm.key_norm"),
        (".ff.linear_in", ".mlp_in"),
        (".ff.linear_out", ".mlp_out"),
    ];

    /// <summary>Every BFL module with its kohya flattening, ordered so <c>image_cond</c> wins over <c>image</c> and
    /// <c>ref_time_in</c> over <c>time_in</c>. Entries ending in '.' are indexed block stacks followed by a leaf.</summary>
    private static readonly (string Kohya, string Bfl)[] Modules =
    [
        ("content_mode_blocks_image_cond_", "content_mode_blocks.image_cond."),
        ("content_mode_blocks_image_", "content_mode_blocks.image."),
        ("txt_mode_blocks_", "txt_mode_blocks."),
        ("single_blocks_", "single_blocks."),
        ("early_stream_modulations_image_cond_lin", "early_stream_modulations.image_cond.lin"),
        ("early_stream_modulations_image_lin", "early_stream_modulations.image.lin"),
        ("early_stream_modulations_txt_lin", "early_stream_modulations.txt.lin"),
        ("single_stream_modulations_image_cond_lin", "single_stream_modulations.image_cond.lin"),
        ("single_stream_modulations_image_lin", "single_stream_modulations.image.lin"),
        ("single_stream_modulations_txt_lin", "single_stream_modulations.txt.lin"),
        ("emb_in_image_cond", "emb_in.image_cond"),
        ("emb_in_image", "emb_in.image"),
        ("txt_in", "txt_in"),
        ("ref_time_in_in_layer", "ref_time_in.in_layer"),
        ("ref_time_in_out_layer", "ref_time_in.out_layer"),
        ("time_in_in_layer", "time_in.in_layer"),
        ("time_in_out_layer", "time_in.out_layer"),
        ("final_layer_image_adaLN_modulation_1", FinalAdaLn),
        ("final_layer_image_linear", "final_layer.image.linear"),
    ];

    /// <summary>Whether <paramref name="name"/> (a module path or a weight key) lies under a BFL root.</summary>
    public static bool IsBfl(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        return MatchesRoot(name, BflRoots);
    }

    /// <summary>Whether <paramref name="module"/> is a module of the BFL layout (a whole module path, not a weight key).</summary>
    public static bool IsModule(string module)
    {
        ArgumentNullException.ThrowIfNull(module);
        foreach ((string _, string bfl) in Modules)
        {
            if (!bfl.EndsWith('.'))
            {
                if (module == bfl)
                    return true;
            }
            else if (TrySplitBlock(module, bfl, out string leaf) && Array.IndexOf(BlockModuleLeaves, leaf) >= 0)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Whether the BFL module <paramref name="module"/> exists only in FLUX.3: a FLUX.3-only root, or a joint
    /// block leaf Flux.1/.2 do not have.</summary>
    public static bool IsDistinctiveBfl(string module)
    {
        ArgumentNullException.ThrowIfNull(module);
        if (MatchesRoot(module, DistinctiveBflRoots))
            return true;
        return TrySplitBlock(module, "single_blocks.", out string leaf) && Array.IndexOf(BlockLeaves, leaf) >= 0;
    }

    /// <summary>Whether the diffusers module <paramref name="module"/> exists only in FLUX.3: a FLUX.3-only root, or the
    /// joint block feed-forward (Flux.1/.2 joint blocks fuse it into other projections).</summary>
    public static bool IsDistinctiveDiffusers(string module)
    {
        ArgumentNullException.ThrowIfNull(module);
        if (MatchesRoot(module, DistinctiveDiffusersRoots))
            return true;
        return TrySplitBlock(module, "single_transformer_blocks.", out string leaf)
            && (leaf == "ff.linear_in" || leaf == "ff.linear_out");
    }

    /// <summary>Maps a diffusers module path to its BFL name, or null when it is not a FLUX.3 diffusers module.</summary>
    public static string? FromDiffusers(string module)
    {
        ArgumentNullException.ThrowIfNull(module);
        foreach ((string diffusers, string bfl) in ExactMap)
        {
            if (module == diffusers)
                return bfl;
        }
        string mapped = module;
        foreach ((string diffusers, string bfl) in PrefixMap)
        {
            if (mapped.StartsWith(diffusers, StringComparison.Ordinal))
            {
                mapped = bfl + mapped[diffusers.Length..];
                break;
            }
        }
        // diffusers renames every modulation `.lin` to `.linear`; nothing else in the model has either leaf.
        if (mapped.EndsWith(".linear", StringComparison.Ordinal) && mapped.Contains("_stream_modulations.", StringComparison.Ordinal))
        {
            mapped = mapped[..^".linear".Length] + ".lin";
        }
        else
        {
            foreach ((string diffusers, string bfl) in BlockSuffixMap)
            {
                if (mapped.EndsWith(diffusers, StringComparison.Ordinal))
                {
                    mapped = mapped[..^diffusers.Length] + bfl;
                    break;
                }
            }
        }
        return mapped != module && IsModule(mapped) ? mapped : null;
    }

    /// <summary>Maps a kohya module (<c>lora_unet_</c> stripped, underscores for dots) to its BFL name, or null when it
    /// is not a FLUX.3 module.</summary>
    public static string? FromKohya(string flat)
    {
        ArgumentNullException.ThrowIfNull(flat);
        foreach ((string head, string dotted) in Modules)
        {
            if (!flat.StartsWith(head, StringComparison.Ordinal))
                continue;
            string rest = flat[head.Length..];
            if (!dotted.EndsWith('.'))
                return rest.Length == 0 ? dotted : null;
            int underscore = rest.IndexOf('_', StringComparison.Ordinal);
            if (underscore <= 0 || !int.TryParse(rest.AsSpan(0, underscore), NumberStyles.None, CultureInfo.InvariantCulture, out int index))
                return null;
            string leaf = rest[(underscore + 1)..];
            return Array.IndexOf(BlockLeaves, leaf) >= 0
                ? string.Concat(dotted, index.ToString(CultureInfo.InvariantCulture), ".", leaf)
                : null;
        }
        return null;
    }

    /// <summary>Whether <paramref name="name"/> is one of <paramref name="roots"/> (dotted) or lies under it.</summary>
    private static bool MatchesRoot(string name, IReadOnlyList<string> roots)
    {
        foreach (string root in roots)
        {
            if (name.StartsWith(root, StringComparison.Ordinal)
                || (name.Length == root.Length - 1 && root.StartsWith(name, StringComparison.Ordinal)))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>Splits <c>{prefix}{index}.{leaf}</c> into its leaf.</summary>
    private static bool TrySplitBlock(string module, string prefix, out string leaf)
    {
        leaf = string.Empty;
        if (!module.StartsWith(prefix, StringComparison.Ordinal))
            return false;
        int dot = module.IndexOf('.', prefix.Length);
        if (dot <= prefix.Length || !int.TryParse(module.AsSpan(prefix.Length, dot - prefix.Length), NumberStyles.None,
                CultureInfo.InvariantCulture, out _))
        {
            return false;
        }
        leaf = module[(dot + 1)..];
        return true;
    }
}

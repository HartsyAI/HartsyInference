using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads Krea 2 (`krea/Krea-2-Turbo`, `CalamitousFelicitousness/Krea-2-Base-Diffusers`) checkpoints. Krea 2 ships in the diffusers folder layout: <c>{root}/transformer/</c> (the <see cref="HartsyInference.Diffusion.Models.Denoisers.Krea2Transformer"/>), <c>{root}/text_encoder/</c> (Qwen3-VL-4B language tower), <c>{root}/vae/</c> (the Qwen-Image VAE), plus <c>{root}/scheduler/</c> + <c>{root}/tokenizer/</c>.
///
/// <para>Transformer keys are diffusers-native (bare <c>img_in.weight</c>, <c>transformer_blocks.{i}.*</c>, <c>text_fusion.*</c>, <c>time_embed.*</c>, <c>final_layer.*</c>) — only an optional <c>transformer.</c> / <c>model.diffusion_model.</c> prefix is stripped and fp8_scaled companions are folded via <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/>. The Qwen3-VL-4B keys are remapped to the <see cref="HartsyInference.Diffusion.Models.TextEncoders.LlamaStyleEncoder"/> convention (vision tower + lm_head dropped). The Qwen-Image VAE keys are consumed directly.</para>
///
/// <para>TODO: the Comfy single-file release (fp8/nv4 under <c>diffusion_models/</c>) may use the original short-name config (<c>features</c>/<c>heads</c>/… ) and raw weight keys; add a single-file key remap when that layout is needed.</para></summary>
public sealed class Krea2CheckpointConverter
{
    /// <summary>Loads the Krea 2 transformer from <c>{root}/transformer/</c> (or a <c>krea2*</c> single file). fp8 folded. Keys are normalized to the diffusers <see cref="HartsyInference.Diffusion.Models.Denoisers.Krea2Transformer"/> convention via <see cref="RemapTransformerKey"/> (handles both the released <b>raw</b> Comfy single-file naming — <c>blocks.N.*</c> / <c>txtfusion.*</c> / <c>tmlp</c> / <c>tproj</c> / <c>last.*</c> — and an already-diffusers folder, which passes through).</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadTransformer(string rootPath)
    {
        string[] shards = CheckpointConvertUtils.DiscoverShards(Path.Combine(rootPath, "transformer"), rootPath, "krea2", "Krea 2");
        return CheckpointConvertUtils.LoadShards(shards, 1200,
            k => RemapTransformerKey(CheckpointConvertUtils.StripTransformerPrefix(k)));
    }

    /// <summary>Maps a Krea 2 transformer weight key to the diffusers convention the engine's <see cref="HartsyInference.Diffusion.Models.Denoisers.Krea2Transformer"/> consumes. The released checkpoints use the model's original ("raw") names; this rewrites them. Already-diffusers keys are returned unchanged.
    ///
    /// <para>Top level: <c>first→img_in</c>, <c>tmlp.0/2→time_embed.linear_1/2</c>, <c>tproj.1→time_mod_proj</c>, <c>txtmlp.0.scale→txt_in.norm.weight</c>, <c>txtmlp.1/3→txt_in.linear_1/2</c>, <c>txtfusion→text_fusion</c>, <c>last.modulation.lin→final_layer.scale_shift_table</c>, <c>last.norm.scale→final_layer.norm.weight</c>, <c>last.linear→final_layer.linear</c>. Blocks: <c>blocks.N→transformer_blocks.N</c> with the per-block sub-map (<c>prenorm/postnorm.scale→norm1/norm2.weight</c>, <c>attn.w{q,k,v,o}→attn.to_{q,k,v,out.0}</c>, <c>attn.gate→attn.to_gate</c>, <c>attn.qknorm.{q,k}norm.scale→attn.norm_{q,k}.weight</c>, <c>mlp.{gate,up,down}→ff.{gate,up,down}</c>, <c>mod.lin→scale_shift_table</c>).</para></summary>
    public static string RemapTransformerKey(string key)
    {
        // fp8 scale companions (`.weight_scale` / `.scale_weight`) must remap to the SAME base key as their
        // `.weight` so ApplyFp8ScaledDequant can pair them (e.g. raw `blocks.0.attn.wq.weight_scale` →
        // `transformer_blocks.0.attn.to_q.weight_scale`, matching `…attn.to_q.weight`). Remap the underlying
        // weight name, then restore the scale suffix. Without this the scale is dropped and fp8 weights stay
        // ~250–900× too large → noise.
        foreach (string suffix in new[] { ".weight_scale", ".scale_weight" })
        {
            if (key.EndsWith(suffix, StringComparison.Ordinal))
            {
                string mappedWeight = RemapTransformerKey(key[..^suffix.Length] + ".weight");
                return mappedWeight[..^".weight".Length] + suffix;
            }
        }

        // Already-diffusers keys pass through.
        if (key.StartsWith("img_in.", StringComparison.Ordinal) || key.StartsWith("transformer_blocks.", StringComparison.Ordinal)
            || key.StartsWith("text_fusion.", StringComparison.Ordinal) || key.StartsWith("time_embed.", StringComparison.Ordinal)
            || key.StartsWith("time_mod_proj.", StringComparison.Ordinal) || key.StartsWith("txt_in.", StringComparison.Ordinal)
            || key.StartsWith("final_layer.", StringComparison.Ordinal))
            return key;

        // Top-level renames.
        if (key == "first.weight") return "img_in.weight";
        if (key == "first.bias") return "img_in.bias";
        if (key == "tmlp.0.weight") return "time_embed.linear_1.weight";
        if (key == "tmlp.0.bias") return "time_embed.linear_1.bias";
        if (key == "tmlp.2.weight") return "time_embed.linear_2.weight";
        if (key == "tmlp.2.bias") return "time_embed.linear_2.bias";
        if (key == "tproj.1.weight") return "time_mod_proj.weight";
        if (key == "tproj.1.bias") return "time_mod_proj.bias";
        if (key == "txtmlp.0.scale") return "txt_in.norm.weight";
        if (key == "txtmlp.1.weight") return "txt_in.linear_1.weight";
        if (key == "txtmlp.1.bias") return "txt_in.linear_1.bias";
        if (key == "txtmlp.3.weight") return "txt_in.linear_2.weight";
        if (key == "txtmlp.3.bias") return "txt_in.linear_2.bias";
        if (key == "txtfusion.projector.weight") return "text_fusion.projector.weight";
        if (key == "last.modulation.lin") return "final_layer.scale_shift_table";
        if (key == "last.norm.scale") return "final_layer.norm.weight";
        if (key == "last.linear.weight") return "final_layer.linear.weight";
        if (key == "last.linear.bias") return "final_layer.linear.bias";

        // Block streams: blocks.N → transformer_blocks.N; txtfusion.{layerwise,refiner}_blocks.N → text_fusion.…
        if (key.StartsWith("blocks.", StringComparison.Ordinal))
            return RemapBlockSubKey("transformer_blocks.", key["blocks.".Length..]);
        if (key.StartsWith("txtfusion.layerwise_blocks.", StringComparison.Ordinal))
            return RemapBlockSubKey("text_fusion.layerwise_blocks.", key["txtfusion.layerwise_blocks.".Length..]);
        if (key.StartsWith("txtfusion.refiner_blocks.", StringComparison.Ordinal))
            return RemapBlockSubKey("text_fusion.refiner_blocks.", key["txtfusion.refiner_blocks.".Length..]);

        return key; // unknown — leave as-is so a missing-key error surfaces the real name.
    }

    /// <summary>Rewrites <c>{index}.{rawSub}</c> within a block to <c>{outPrefix}{index}.{diffusersSub}</c>.</summary>
    private static string RemapBlockSubKey(string outPrefix, string rest)
    {
        int dot = rest.IndexOf('.');
        if (dot < 0) return outPrefix + rest;
        string idx = rest[..dot];
        string sub = rest[(dot + 1)..];
        string mapped = sub switch
        {
            "prenorm.scale" => "norm1.weight",
            "postnorm.scale" => "norm2.weight",
            "mod.lin" => "scale_shift_table",
            "attn.wq.weight" => "attn.to_q.weight",
            "attn.wk.weight" => "attn.to_k.weight",
            "attn.wv.weight" => "attn.to_v.weight",
            "attn.wo.weight" => "attn.to_out.0.weight",
            "attn.gate.weight" => "attn.to_gate.weight",
            "attn.qknorm.qnorm.scale" => "attn.norm_q.weight",
            "attn.qknorm.knorm.scale" => "attn.norm_k.weight",
            "mlp.gate.weight" => "ff.gate.weight",
            "mlp.up.weight" => "ff.up.weight",
            "mlp.down.weight" => "ff.down.weight",
            _ => sub,
        };
        return $"{outPrefix}{idx}.{mapped}";
    }

    /// <summary>Loads the Qwen-Image VAE from <c>{root}/vae/</c> (keys consumed directly by <c>QwenImageVaeDecoder</c>).</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadVae(string rootPath)
    {
        string dir = Path.Combine(rootPath, "vae");
        if (!Directory.Exists(dir))
            throw new DirectoryNotFoundException($"VAE folder not found: {dir}");
        string[] shards = Directory.GetFiles(dir, "*.safetensors");
        if (shards.Length == 0)
            throw new FileNotFoundException($"No VAE .safetensors found under {dir}.");
        Array.Sort(shards);
        return CheckpointConvertUtils.LoadShards(shards, 400, k => k);
    }

    /// <summary>Loads + remaps the Qwen3-VL-4B language tower from <c>{root}/text_encoder/</c> to the <c>LlamaStyleEncoder</c> convention (drops the vision tower and <c>lm_head</c>).</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadTextEncoder(string rootPath)
    {
        string te1 = Path.Combine(rootPath, "text_encoder");
        string te2 = Path.Combine(rootPath, "text_encoders");
        string dir = Directory.Exists(te1) ? te1 : te2;
        string[] shards = CheckpointConvertUtils.DiscoverShards(dir, rootPath, "qwen", "Krea 2");
        return CheckpointConvertUtils.LoadShards(shards, 800, CheckpointConvertUtils.RemapQwenLanguageKey);
    }
}

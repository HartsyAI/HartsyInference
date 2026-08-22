using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Loads Boogu-Image-0.1 (<c>Boogu/Boogu-Image-0.1-Base</c> / <c>-Edit</c> / <c>-Turbo</c>) checkpoints. Boogu ships in the diffusers folder layout: <c>{root}/transformer/</c> (the <see cref="HartsyInference.Diffusion.Models.Denoisers.BooguImageTransformer"/>), <c>{root}/vae/</c> (the FLUX.1 VAE), <c>{root}/mllm/</c> (the full Qwen3-VL-8B — language tower + vision tower), and <c>{root}/scheduler/</c> + <c>{root}/processor/</c> (config only).
///
/// <para>Transformer keys are diffusers-native (bare <c>x_embedder.weight</c>, <c>double_stream_layers.{i}.*</c>, <c>single_stream_layers.{i}.*</c>) — only an optional <c>transformer.</c> / <c>model.diffusion_model.</c> prefix is stripped and fp8_scaled companions are folded via <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/>. The Qwen3-VL language tower is remapped to the <see cref="HartsyInference.Diffusion.Models.TextEncoders.LlamaStyleEncoder"/> convention; the vision tower (<c>visual.*</c>) is kept separately (re-rooted to bare keys) for the Boogu vision encoder used by the edit path.</para></summary>
public sealed class BooguImageCheckpointConverter
{
    /// <summary>Loads the Boogu transformer from <c>{root}/transformer/</c> (sharded). fp8 scale companions folded.</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadTransformer(string rootPath)
    {
        string dir = Path.Combine(rootPath, "transformer");
        string[] shards = CheckpointConvertUtils.DiscoverShards(dir, rootPath, "transformer", "Boogu");
        return CheckpointConvertUtils.LoadShards(shards, 2200, CheckpointConvertUtils.StripTransformerPrefix);
    }

    /// <summary>Loads the FLUX.1 VAE from <c>{root}/vae/</c>. Boogu ships the Comfy/ldm single-file VAE (<c>flux1_vae_bf16.safetensors</c>) with bare ldm keys (<c>decoder.mid.block_1.*</c>, <c>encoder.down.*</c>), so each key is remapped to the diffusers convention via <see cref="CheckpointConvertUtils.ConvertVaeKey"/> for <c>VaeEncoder</c>/<c>VaeDecoder</c> + <c>VaeConfig.Flux</c>.</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadVae(string rootPath)
    {
        string dir = Path.Combine(rootPath, "vae");
        if (!Directory.Exists(dir))
            throw new DirectoryNotFoundException($"VAE folder not found: {dir}");
        string[] shards = Directory.GetFiles(dir, "*.safetensors");
        if (shards.Length == 0)
            throw new FileNotFoundException($"No VAE .safetensors found under {dir}.");
        Array.Sort(shards);
        return CheckpointConvertUtils.LoadShards(shards, 400, k => CheckpointConvertUtils.ConvertVaeKey(k));
    }

    /// <summary>Loads + remaps the Qwen3-VL-8B language tower from <c>{root}/mllm/</c> to the <c>LlamaStyleEncoder</c> convention (drops the vision tower and <c>lm_head</c>).</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadTextEncoder(string rootPath)
    {
        string[] shards = CheckpointConvertUtils.DiscoverShards(Path.Combine(rootPath, "mllm"), rootPath, "mllm", "Boogu");
        return CheckpointConvertUtils.LoadShards(shards, 800, CheckpointConvertUtils.RemapQwenLanguageKey);
    }

    /// <summary>Loads the Qwen3-VL-8B vision tower from <c>{root}/mllm/</c> (the <c>visual.*</c> subtree), re-rooted to bare keys (<c>patch_embed.*</c>, <c>blocks.{i}.*</c>, <c>merger.*</c>) for the Boogu vision encoder.</summary>
    public static (Dictionary<string, Tensor> Weights, IReadOnlyList<SafeTensorsLoader> Loaders) LoadVisionTower(string rootPath)
    {
        string[] shards = CheckpointConvertUtils.DiscoverShards(Path.Combine(rootPath, "mllm"), rootPath, "mllm", "Boogu");
        return CheckpointConvertUtils.LoadShards(shards, 600, RemapQwenVisionKey);
    }

    /// <summary>Maps a Qwen3-VL vision-tower key to bare keys (strips <c>…visual.</c>), or null to drop non-vision keys.</summary>
    private static string? RemapQwenVisionKey(string key)
    {
        int v = key.LastIndexOf(".visual.", StringComparison.Ordinal);
        if (v >= 0)
            return key[(v + ".visual.".Length)..];
        if (key.StartsWith("visual.", StringComparison.Ordinal))
            return key["visual.".Length..];
        return null;
    }
}

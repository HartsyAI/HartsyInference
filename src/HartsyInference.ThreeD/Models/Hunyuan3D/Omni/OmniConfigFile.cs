using System.Globalization;
using System.Text.Json;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Omni;

/// <summary>Reads the architecture of a <c>tencent/Hunyuan3D-Omni</c> folder from its JSON configs (<c>config.json</c>, <c>model/config.json</c>, <c>vae/config.json</c>) into the <see cref="Hunyuan3DConfig"/> that the
/// Hunyuan3D-2.1 shape stack runs: the Omni DiT is <c>HunYuanDiTPlain</c> with the 2.1 key set and shapes, its VAE is the 2.1 decoder with its own values and scale factor.</summary>
public static class OmniConfigFile
{
    /// <summary>The DINO tokens at the start of the DiT's cross-attention context: 1 + 37 x 37 from DINOv2-large at 518 px. The control tokens and the 10 signal tokens follow them.</summary>
    public const int ImageTokens = 1370;

    /// <summary>Reads the three config files under <paramref name="folder"/>.</summary>
    /// <exception cref="FileNotFoundException">A config file is missing.</exception>
    /// <exception cref="UnsupportedModelException">The configs ask for something the engine does not implement.</exception>
    public static Hunyuan3DConfig Read(string folder)
    {
        ArgumentException.ThrowIfNullOrEmpty(folder);
        return Parse(Text(folder, "config.json"), Text(folder, Path.Combine("model", "config.json")), Text(folder, Path.Combine("vae", "config.json")));
    }

    /// <summary>Builds the config from the three JSON documents.</summary>
    public static Hunyuan3DConfig Parse(string pipelineJson, string modelJson, string vaeJson)
    {
        using JsonDocument pipeline = JsonDocument.Parse(pipelineJson), model = JsonDocument.Parse(modelJson), vae = JsonDocument.Parse(vaeJson);
        JsonElement run = Section(pipeline.RootElement, "params", "config.json");
        JsonElement dit = Section(model.RootElement, "params", "model/config.json");
        JsonElement vaeParams = Section(Section(Section(vae.RootElement, "params", "vae/config.json"), "module_cfg", "vae/config.json"), "params", "vae/config.json");

        if (!Target(pipeline.RootElement).EndsWith(".Hunyuan3DOmniSiTFlowMatchingPipeline", StringComparison.Ordinal))
            throw new UnsupportedModelException($"Hunyuan3D-Omni config.json names the pipeline '{Target(pipeline.RootElement)}', not Hunyuan3DOmniSiTFlowMatchingPipeline.");
        if (!Target(model.RootElement).EndsWith(".HunYuanDiTPlain", StringComparison.Ordinal))
            throw new UnsupportedModelException($"Hunyuan3D-Omni model/config.json names the denoiser '{Target(model.RootElement)}', not HunYuanDiTPlain.");
        if (!Target(vae.RootElement).EndsWith(".ShapeVAEPL", StringComparison.Ordinal))
            throw new UnsupportedModelException($"Hunyuan3D-Omni vae/config.json names the autoencoder '{Target(vae.RootElement)}', not ShapeVAEPL.");

        string normType = Str(dit, "qk_norm_type", "model/config.json");
        if (normType != "rms")
            throw new UnsupportedModelException($"Hunyuan3D-Omni qk_norm_type={normType} is not supported: the Q/K norm must be rms.");
        foreach (string off in new[] { "with_decoupled_ca", "use_attention_pooling", "use_pos_emb", "qkv_bias" })
        {
            if (Flag(dit, off))
                throw new UnsupportedModelException($"Hunyuan3D-Omni model/config.json {off}=true is not supported.");
        }

        if (!Flag(dit, "qk_norm") || dit.TryGetProperty("guidance_embed", out JsonElement guidance) && guidance.ValueKind == JsonValueKind.True)
            throw new UnsupportedModelException("Hunyuan3D-Omni without QK norm or with a guidance embedding is not supported.");
        if (Int(dit, "text_len", "model/config.json") != ImageTokens)
            throw new UnsupportedModelException($"Hunyuan3D-Omni text_len must be {ImageTokens} (DINOv2-large at 518 px, CLS token kept).");
        if (Flag(vaeParams, "include_pi") || !Flag(vaeParams, "qk_norm") || Flag(vaeParams, "qkv_bias"))
            throw new UnsupportedModelException("Hunyuan3D-Omni ShapeVAE with include_pi, without QK norm or with qkv bias is not supported.");

        int latents = Int(vaeParams, "num_latents", "vae/config.json"), width = Int(dit, "hidden_size", "model/config.json");
        if (Int(dit, "input_size", "model/config.json") != latents)
            throw new UnsupportedModelException("Hunyuan3D-Omni DiT input_size differs from the VAE num_latents.");
        float scale = Float(vae.RootElement, "scale_factor", "vae/config.json");
        if (Math.Abs(scale - Float(run, "scaling_factor", "config.json")) > 1e-9f)
            throw new UnsupportedModelException("Hunyuan3D-Omni vae scale_factor differs from the pipeline scaling_factor.");

        // The released 2.1 stack stays the base for everything the Omni configs do not state (geometry decoder shape, ratios).
        return Hunyuan3DConfig.Hunyuan3D2 with
        {
            LatentTokens = latents,
            LatentChannels = Int(dit, "in_channels", "model/config.json"),
            Width = width,
            NumHeads = Int(dit, "num_heads", "model/config.json"),
            CondDim = Int(dit, "context_dim", "model/config.json"),
            MlpDim = width * 4, // mlp_ratio is not set in the Omni config: upstream's default 4.0
            VaeWidth = Int(vaeParams, "width", "vae/config.json"),
            VaeDepth = Int(vaeParams, "num_decoder_layers", "vae/config.json"),
            VaeNumHeads = Int(vaeParams, "heads", "vae/config.json"),
            FourierBands = Int(vaeParams, "num_freqs", "vae/config.json"),
            VaeScaleFactor = scale,
            PlainDit = new Hunyuan3DPlainDitConfig(
                Int(dit, "depth", "model/config.json"), Int(dit, "num_moe_layers", "model/config.json"),
                Int(dit, "num_experts", "model/config.json"), Int(dit, "moe_top_k", "model/config.json")),
            NumInferenceSteps = Int(run, "num_inference_steps", "config.json"),
            GuidanceScale = 4.5f, // what upstream's inference.py passes for every control (the signature of Hunyuan3DOmniSiTFlowMatchingPipeline.__call__ defaults to 7.5)
            GridResolution = Int(run, "octree_resolution", "config.json"),
            BoundingBox = Float(run, "box_v", "config.json"),
            IsoLevel = Float(run, "mc_level", "config.json"),
        };
    }

    private static string Text(string folder, string relative)
    {
        string path = Path.Combine(folder, relative);
        if (!File.Exists(path))
            throw new FileNotFoundException($"The Hunyuan3D-Omni folder '{folder}' has no {relative.Replace('\\', '/')}.", path);
        return File.ReadAllText(path);
    }

    private static string Target(JsonElement root) => root.TryGetProperty("target", out JsonElement t) && t.ValueKind == JsonValueKind.String ? t.GetString()! : string.Empty;

    private static JsonElement Section(JsonElement parent, string name, string file) =>
        parent.TryGetProperty(name, out JsonElement e) && e.ValueKind == JsonValueKind.Object ? e : throw new UnsupportedModelException($"Hunyuan3D-Omni {file} has no '{name}' section.");

    private static string Str(JsonElement e, string name, string file) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String ? v.GetString()! : throw new UnsupportedModelException($"Hunyuan3D-Omni {file} has no '{name}'.");

    private static int Int(JsonElement e, string name, string file) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out int i) ? i : throw new UnsupportedModelException($"Hunyuan3D-Omni {file} has no integer '{name}'.");

    private static float Float(JsonElement e, string name, string file) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? float.Parse(v.GetRawText(), CultureInfo.InvariantCulture) : throw new UnsupportedModelException($"Hunyuan3D-Omni {file} has no number '{name}'.");

    private static bool Flag(JsonElement e, string name) => e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.True;
}

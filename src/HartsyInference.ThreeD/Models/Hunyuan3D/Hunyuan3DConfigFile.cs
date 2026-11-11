using System.Globalization;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Reads the <c>config.yaml</c> that ships beside a Hunyuan3D-2 checkpoint into a <see cref="Hunyuan3DConfig"/>.</summary>
/// <remarks>Only the plain nested mappings these files use are understood, not general YAML. Features the engine does not
/// implement are refused by name rather than loaded with the wrong shapes.</remarks>
public static class Hunyuan3DConfigFile
{
    private const string ModelPrefix = "model.params.";
    private const string VaePrefix = "vae.params.";
    private const string EncoderPrefix = "conditioner.params.main_image_encoder.";

    /// <summary>The checkpoint-bundled config of <paramref name="yaml"/>: DiT, ShapeVAE, conditioner, scheduler and preprocessor sections.</summary>
    /// <exception cref="UnsupportedModelException">The config needs something the engine does not implement.</exception>
    public static Hunyuan3DConfig Parse(string yaml)
    {
        ArgumentNullException.ThrowIfNull(yaml);
        Dictionary<string, string> values = Flatten(yaml);
        if (Require(values, "model.target").EndsWith(".HunYuanDiTPlain", StringComparison.Ordinal))
            return ParsePlain(values);

        RequireValue(values, EncoderPrefix + "kwargs.image_size", "518", "the DINOv2 input size (other sizes need position-embedding interpolation)");
        RequireValue(values, "image_processor.params.size", "512", "the image preprocessor size");
        RequireValue(values, EncoderPrefix + "kwargs.config.hidden_size", "1536", "the DINOv2 variant (only giant is implemented)");
        RequireValue(values, EncoderPrefix + "kwargs.config.num_hidden_layers", "40", "the DINOv2 variant (only giant is implemented)");

        string encoder = Require(values, EncoderPrefix + "type");
        int views = encoder switch
        {
            "DinoImageEncoder" => 1,
            "DinoImageEncoderMV" => 4,
            _ => throw new UnsupportedModelException($"Hunyuan3D conditioner '{encoder}' is not supported.", encoder, "config.yaml"),
        };

        string schedulerTarget = Require(values, "scheduler.target");
        Hunyuan3DScheduler scheduler = schedulerTarget.EndsWith("ConsistencyFlowMatchEulerDiscreteScheduler", StringComparison.Ordinal)
            ? Hunyuan3DScheduler.ConsistencyFlowMatchEuler
            : schedulerTarget.EndsWith("FlowMatchEulerDiscreteScheduler", StringComparison.Ordinal)
                ? Hunyuan3DScheduler.FlowMatchEuler
                : throw new UnsupportedModelException($"Hunyuan3D scheduler '{schedulerTarget}' is not supported.", schedulerTarget, "config.yaml");
        bool guidanceEmbed = Flag(values, ModelPrefix + "guidance_embed", defaultValue: false);

        Hunyuan3DConfig vae = ParseVaeSection(values, VaePrefix);
        double mlpRatio = double.Parse(Require(values, ModelPrefix + "mlp_ratio"), CultureInfo.InvariantCulture);
        int width = Int(values, ModelPrefix + "hidden_size");
        return vae with
        {
            LatentChannels = Int(values, ModelPrefix + "in_channels"),
            CondDim = Int(values, ModelPrefix + "context_in_dim"),
            Width = width,
            MlpDim = (int)(width * mlpRatio),
            NumHeads = Int(values, ModelPrefix + "num_heads"),
            DepthDouble = Int(values, ModelPrefix + "depth"),
            DepthSingle = Int(values, ModelPrefix + "depth_single_blocks"),
            GuidanceEmbed = guidanceEmbed,
            Scheduler = scheduler,
            PcmTimesteps = scheduler == Hunyuan3DScheduler.ConsistencyFlowMatchEuler
                ? Int(values, "scheduler.params.pcm_timesteps") : vae.PcmTimesteps,
            ViewCount = views,
            NumInferenceSteps = scheduler == Hunyuan3DScheduler.ConsistencyFlowMatchEuler ? 5 : guidanceEmbed ? 10 : vae.NumInferenceSteps,
        };
    }

    private static Hunyuan3DConfig ParsePlain(Dictionary<string, string> values)
    {
        RequireValue(values, EncoderPrefix + "kwargs.image_size", "518", "the DINOv2 input size (other sizes need position-embedding interpolation)");
        RequireValue(values, "image_processor.params.size", "512", "the image preprocessor size");
        RequireValue(values, EncoderPrefix + "kwargs.config.hidden_size", "1024", "the DINOv2 variant (only large is implemented for 2.1)");
        RequireValue(values, EncoderPrefix + "kwargs.config.num_hidden_layers", "24", "the DINOv2 variant (only large is implemented for 2.1)");
        RequireValue(values, EncoderPrefix + "kwargs.config.use_swiglu_ffn", "false", "the DINOv2 feed-forward");
        RequireValue(values, EncoderPrefix + "type", "DinoImageEncoder", "the conditioner");
        string normType = Require(values, ModelPrefix + "qk_norm_type").Trim('\'', '"');
        if (normType != "rms")
            throw new UnsupportedModelException($"Hunyuan3D config {ModelPrefix}qk_norm_type={normType} is not supported: the Q/K norm must be rms.");
        if (values.TryGetValue(ModelPrefix + "guidance_cond_proj_dim", out string? guidanceProj) && guidanceProj is not ("null" or "None" or "~" or ""))
            throw new UnsupportedModelException("Hunyuan3D-2.1 guidance_cond_proj_dim is not supported (no guidance embedding is implemented for the plain DiT).");
        double mlpRatio = values.ContainsKey(ModelPrefix + "mlp_ratio")
            ? double.Parse(Require(values, ModelPrefix + "mlp_ratio"), CultureInfo.InvariantCulture) : 4.0; // upstream default
        foreach (string off in new[] { "with_decoupled_ca", "use_attention_pooling", "use_pos_emb" })
        {
            if (Flag(values, ModelPrefix + off, defaultValue: false))
                throw new UnsupportedModelException($"Hunyuan3D-2.1 config {off}=true is not supported.");
        }

        if (Flag(values, ModelPrefix + "qkv_bias", defaultValue: true) || !Flag(values, ModelPrefix + "qk_norm", defaultValue: false))
            throw new UnsupportedModelException("Hunyuan3D-2.1 without QK norm or with qkv bias is not supported.");
        if (!Require(values, "scheduler.target").EndsWith(".FlowMatchEulerDiscreteScheduler", StringComparison.Ordinal))
            throw new UnsupportedModelException("Hunyuan3D-2.1 scheduler is not FlowMatchEulerDiscreteScheduler.");
        int heads = Int(values, ModelPrefix + "num_heads"), width = Int(values, ModelPrefix + "hidden_size");
        if (Int(values, ModelPrefix + "input_size") != Int(values, VaePrefix + "num_latents"))
            throw new UnsupportedModelException("Hunyuan3D-2.1 DiT input_size differs from the VAE num_latents.");
        return ParseVaeSection(values, VaePrefix) with
        {
            LatentChannels = Int(values, ModelPrefix + "in_channels"),
            CondDim = Int(values, ModelPrefix + "context_dim"),
            Width = width,
            MlpDim = (int)(width * mlpRatio),
            NumHeads = heads,
            PlainDit = new Hunyuan3DPlainDitConfig(
                Int(values, ModelPrefix + "depth"), Int(values, ModelPrefix + "num_moe_layers"),
                Int(values, ModelPrefix + "num_experts"), Int(values, ModelPrefix + "moe_top_k")),
        };
    }

    /// <summary>The ShapeVAE of a stand-alone VAE config (<c>hunyuan3d-vae-*</c>), laid over <paramref name="baseConfig"/>.</summary>
    /// <exception cref="UnsupportedModelException">The VAE needs something the engine does not implement.</exception>
    public static Hunyuan3DConfig ParseVae(string yaml, Hunyuan3DConfig baseConfig)
    {
        ArgumentNullException.ThrowIfNull(yaml);
        ArgumentNullException.ThrowIfNull(baseConfig);
        Hunyuan3DConfig vae = ParseVaeSection(Flatten(yaml), "params.");
        return baseConfig with
        {
            LatentTokens = vae.LatentTokens, VaeWidth = vae.VaeWidth, VaeDepth = vae.VaeDepth, VaeNumHeads = vae.VaeNumHeads,
            FourierBands = vae.FourierBands, VaeScaleFactor = vae.VaeScaleFactor, GeoDownsample = vae.GeoDownsample,
            GeoMlpRatio = vae.GeoMlpRatio, GeoLnPost = vae.GeoLnPost,
        };
    }

    /// <summary>Flattens nested mappings to dotted keys (<c>model.params.depth</c>), dropping comments and resolving <c>&amp;name</c> and <c>*name</c> anchors; flow lists stay as text.</summary>
    /// <remarks>Only scalar anchors work (<c>&amp;name 4096</c> then <c>*name</c>); an anchor on a mapping or a merge key (<c>&lt;&lt;: *name</c>) is not expanded.</remarks>
    internal static Dictionary<string, string> Flatten(string yaml)
    {
        Dictionary<string, string> values = new(StringComparer.Ordinal);
        List<(int Indent, string Key)> path = [];
        Dictionary<string, string> anchors = new(StringComparer.Ordinal);
        foreach (string raw in yaml.Split('\n'))
        {
            string line = raw.TrimEnd('\r');
            int hash = line.IndexOf('#', StringComparison.Ordinal);
            if (hash >= 0)
                line = line[..hash];
            line = line.TrimEnd();
            string trimmed = line.TrimStart();
            if (trimmed.Length == 0 || trimmed == "---")
                continue;
            int indent = line.Length - trimmed.Length;
            int colon = trimmed.IndexOf(':', StringComparison.Ordinal);
            if (colon <= 0)
                throw new UnsupportedModelException($"Hunyuan3D config line is not 'key: value': '{trimmed}'.");
            while (path.Count > 0 && path[^1].Indent >= indent)
                path.RemoveAt(path.Count - 1);
            string key = trimmed[..colon].Trim();
            string value = trimmed[(colon + 1)..].Trim();
            if (value.Length == 0)
            {
                path.Add((indent, key));
                continue;
            }

            if (value[0] == '&')
            {
                // "&name value" defines an anchor that "*name" repeats elsewhere in the file.
                int space = value.IndexOf(' ', StringComparison.Ordinal);
                string anchor = space < 0 ? value[1..] : value[1..space];
                value = space < 0 ? string.Empty : value[(space + 1)..].Trim();
                anchors[anchor] = value;
            }
            else if (value[0] == '*')
            {
                value = anchors.TryGetValue(value[1..], out string? alias)
                    ? alias : throw new UnsupportedModelException($"Hunyuan3D config uses the undefined anchor '{value}'.");
            }

            values[string.Join('.', path.Select(p => p.Key).Append(key))] = value;
        }

        return values;
    }

    private static Hunyuan3DConfig ParseVaeSection(Dictionary<string, string> values, string prefix)
    {
        if (Flag(values, prefix + "include_pi", defaultValue: true))
            throw new UnsupportedModelException("Hunyuan3D ShapeVAE include_pi=true is not supported.");
        if (!Flag(values, prefix + "qk_norm", defaultValue: false) || Flag(values, prefix + "qkv_bias", defaultValue: true))
            throw new UnsupportedModelException("Hunyuan3D ShapeVAE without QK norm or with qkv bias is not supported.");
        int downsample = values.ContainsKey(prefix + "geo_decoder_downsample_ratio") ? Int(values, prefix + "geo_decoder_downsample_ratio") : 1;
        int mlpRatio = values.ContainsKey(prefix + "geo_decoder_mlp_expand_ratio") ? Int(values, prefix + "geo_decoder_mlp_expand_ratio") : 4;
        bool lnPost = Flag(values, prefix + "geo_decoder_ln_post", defaultValue: true);
        Hunyuan3DConfig fallback = Hunyuan3DConfig.Hunyuan3D2;
        return fallback with
        {
            LatentTokens = Int(values, prefix + "num_latents"),
            VaeWidth = Int(values, prefix + "width"),
            VaeDepth = Int(values, prefix + "num_decoder_layers"),
            VaeNumHeads = Int(values, prefix + "heads"),
            FourierBands = Int(values, prefix + "num_freqs"),
            VaeScaleFactor = float.Parse(Require(values, prefix + "scale_factor"), CultureInfo.InvariantCulture),
            GeoDownsample = downsample,
            GeoMlpRatio = mlpRatio,
            GeoLnPost = lnPost,
        };
    }

    private static string Require(Dictionary<string, string> values, string key) =>
        values.TryGetValue(key, out string? value) ? value : throw new UnsupportedModelException($"Hunyuan3D config has no '{key}'.");

    private static int Int(Dictionary<string, string> values, string key) =>
        int.TryParse(Require(values, key), NumberStyles.Integer, CultureInfo.InvariantCulture, out int result)
            ? result : throw new UnsupportedModelException($"Hunyuan3D config '{key}' is not an integer: '{values[key]}'.");

    private static bool Flag(Dictionary<string, string> values, string key, bool defaultValue) =>
        !values.TryGetValue(key, out string? value) ? defaultValue
            : value.Equals("true", StringComparison.OrdinalIgnoreCase) ? true
            : value.Equals("false", StringComparison.OrdinalIgnoreCase) ? false
            : throw new UnsupportedModelException($"Hunyuan3D config '{key}' is not true or false: '{value}'.");

    private static void RequireValue(Dictionary<string, string> values, string key, string expected, string what)
    {
        string actual = Require(values, key);
        if (actual != expected)
            throw new UnsupportedModelException($"Hunyuan3D config {key}={actual} is not supported: {what} must be {expected}.");
    }
}

using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>Maps the torch state-dict names of a CoLa3D <c>.pt</c> checkpoint to the decomposer's parameter names.</summary>
/// <remarks>Backbone parameters are named <c>backbone.blocks.{i}.{module}.{suffix}</c>. <c>qRead*</c> is the query side of
/// queries attending to points, with <c>pReadNorm</c> and <c>pReadKv</c> the point side of that read. <c>qWrite*</c> is the
/// query side of points attending to queries, with <c>pWrite*</c> the point side of that update.</remarks>
public static class CoLa3DKeyMap
{
    private const string Index = "#";

    private static readonly Dictionary<string, string> Bare = new(StringComparer.Ordinal)
    {
        ["mask_decoder.temperature"] = "head.temperature",
        ["pat.query_embed"] = "query.learnable",
    };

    private static readonly Dictionary<string, string> Modules = BuildModules();

    /// <summary>Maps one torch key to its parameter name, or returns null when the key is not a CoLa3D tensor.</summary>
    public static string? TryMap(string torchKey)
    {
        ArgumentNullException.ThrowIfNull(torchKey);
        string[] parts = torchKey.Split('.');
        string? index = null;
        for (int i = 1; i < parts.Length; i++)
        {
            if (parts[i].All(char.IsAsciiDigit) && parts[i - 1] is "blocks" or "encoder")
            {
                index = parts[i];
                parts[i] = Index;
                break;
            }
        }

        string whole = string.Join('.', parts);
        if (Bare.TryGetValue(whole, out string? bare))
            return Substitute(bare, index);
        if (parts.Length < 2)
            return null;

        string module = string.Join('.', parts, 0, parts.Length - 1);
        string suffix = parts[^1];
        if (suffix is not ("weight" or "bias" or "gamma") || !Modules.TryGetValue(module, out string? name))
            return null;
        return Substitute(name + "." + suffix, index);
    }

    /// <summary>Maps a torch state dict to parameter names and checks it against the config: every key must map to a
    /// parameter the config uses, with the expected shape, and every parameter must be present.</summary>
    public static Dictionary<string, Tensor> Remap(CoLa3DConfig config, IReadOnlyDictionary<string, Tensor> torchTensors)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(torchTensors);
        config.Validate();
        IReadOnlyDictionary<string, long[]> expected = ExpectedShapes(config);
        Dictionary<string, Tensor> named = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> entry in torchTensors)
        {
            string? name = TryMap(entry.Key)
                ?? throw new HartsyInferenceException($"CoLa3D checkpoint tensor '{entry.Key}' has no key-map entry.");
            if (!expected.TryGetValue(name, out long[]? shape))
                throw new HartsyInferenceException(
                    $"CoLa3D checkpoint tensor '{entry.Key}' ({name}) is not used by this configuration.");
            CheckShape(entry.Key, entry.Value, shape);
            named[name] = entry.Value;
        }

        ValidateNamed(config, named);
        return named;
    }

    /// <summary>Checks a parameter-named dictionary against the config: exactly the expected names, each F32 with the expected shape.</summary>
    public static void ValidateNamed(CoLa3DConfig config, IReadOnlyDictionary<string, Tensor> named)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(named);
        IReadOnlyDictionary<string, long[]> expected = ExpectedShapes(config);
        List<string> missing = expected.Keys.Where(k => !named.ContainsKey(k)).Order(StringComparer.Ordinal).ToList();
        if (missing.Count > 0)
            throw new HartsyInferenceException(
                $"CoLa3D weights are missing {missing.Count} parameter(s), first: {missing[0]}.");
        foreach (KeyValuePair<string, long[]> entry in expected)
        {
            Tensor tensor = named[entry.Key];
            CheckShape(entry.Key, tensor, entry.Value);
        }
    }

    /// <summary>Every parameter name the config uses, with its shape.</summary>
    public static IReadOnlyDictionary<string, long[]> ExpectedShapes(CoLa3DConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        Dictionary<string, long[]> s = new(StringComparer.Ordinal);
        long c = config.HiddenDim, l = config.LatentChannels, u = config.UtoniaFeatureDim;
        long m = config.MlpHidden, q = config.MaxParts, h = config.NumHeads, d = config.HeadDim;

        Dense(s, "backbone.latentProj", c, l);
        Dense(s, "backbone.utoniaProj", c, u);
        Affine(s, "backbone.latentNorm", c);
        Affine(s, "backbone.utoniaNorm", c);
        Dense(s, "backbone.fusion", c, 2 * c);
        Affine(s, "backbone.pFinalNorm", c);
        Affine(s, "backbone.qFinalNorm", c);

        Dense(s, "head.pointProj", c, c);
        Dense(s, "head.queryProj", c, c);
        s["head.temperature"] = [1];

        for (int i = 0; i < config.NumLayers; i++)
        {
            string b = $"backbone.blocks.{i}";
            Affine(s, $"{b}.qSelfNorm", c);
            Dense(s, $"{b}.qSelfQkv", 3 * c, c);
            Dense(s, $"{b}.qSelfOut", c, c);
            Affine(s, $"{b}.qReadNorm", c);
            Dense(s, $"{b}.qReadQuery", c, c);
            Affine(s, $"{b}.pReadNorm", c);
            Dense(s, $"{b}.pReadKv", 2 * c, c);
            Dense(s, $"{b}.qReadOut", c, c);
            Affine(s, $"{b}.qFfnNorm", c);
            Dense(s, $"{b}.qFfnUp", m, c);
            Dense(s, $"{b}.qFfnDown", c, m);
            Dense(s, $"{b}.qWriteKv", 2 * c, c);
            Affine(s, $"{b}.pWriteNorm", c);
            Dense(s, $"{b}.pWriteQuery", c, c);
            s[$"{b}.pWriteQkNorm.gamma"] = [h, d];
            s[$"{b}.qWriteKNorm.gamma"] = [h, d];
            Dense(s, $"{b}.pWriteOut", c, c);
            Affine(s, $"{b}.pFfnNorm", c);
            Dense(s, $"{b}.pFfnUp", m, c);
            Dense(s, $"{b}.pFfnDown", c, m);
        }

        switch (config.QuerySource)
        {
            case CoLa3DConfig.QuerySourceKind.Fixed when config.QueryInit == CoLa3DConfig.QueryInitKind.Learnable:
                s["query.learnable"] = [1, q, c];
                break;
            case CoLa3DConfig.QuerySourceKind.Fixed:
                Dense(s, "query.positional.up", c, CoLa3DOps.PartFrequencyDim);
                Dense(s, "query.positional.down", c, c);
                break;
            case CoLa3DConfig.QuerySourceKind.PromptMask:
                AddMaskEncoder(s, config, c);
                break;
            case CoLa3DConfig.QuerySourceKind.PromptPoint:
                AddPointEncoder(s, config, c, l);
                break;
        }

        return s;
    }

    private static void AddMaskEncoder(Dictionary<string, long[]> s, CoLa3DConfig config, long c)
    {
        long h = config.MaskEncoderHidden, h2 = h / 2, h4 = h / 4;
        const string e = "maskPrompt.conv";
        s[$"{e}.0.weight"] = [h4, 1, 3, 3];
        s[$"{e}.0.bias"] = [h4];
        s[$"{e}.1.weight"] = [h4];
        s[$"{e}.1.bias"] = [h4];
        s[$"{e}.3.weight"] = [h2, h4, 3, 3];
        s[$"{e}.3.bias"] = [h2];
        s[$"{e}.4.weight"] = [h2];
        s[$"{e}.4.bias"] = [h2];
        s[$"{e}.6.weight"] = [h, h2, 3, 3];
        s[$"{e}.6.bias"] = [h];
        s[$"{e}.7.weight"] = [h];
        s[$"{e}.7.bias"] = [h];
        s[$"{e}.9.weight"] = [c, h, 1, 1];
        s[$"{e}.9.bias"] = [c];
        if (config.MaskEncoder != CoLa3DConfig.MaskEncoderKind.CnnCenterPe)
            return;
        Affine(s, "maskPrompt.cnnNorm", c);
        Dense(s, "maskPrompt.peUp", c, 2L * config.MaskPePerAxis);
        Dense(s, "maskPrompt.peDown", c, c);
        Affine(s, "maskPrompt.peNorm", c);
        Dense(s, "maskPrompt.fuseUp", c, c);
        Dense(s, "maskPrompt.fuseDown", c, c);
    }

    private static void AddPointEncoder(Dictionary<string, long[]> s, CoLa3DConfig config, long c, long latentChannels)
    {
        long nf = config.PointFourierFeatures;
        long latentDim = config.PointFeatureSource == CoLa3DConfig.PointFeatureSourceKind.LatentUtonia ? c : latentChannels;
        s["pointPrompt.fourierWr.weight"] = [nf, 3];
        Dense(s, "pointPrompt.fourierUp", c, 2 * nf);
        Dense(s, "pointPrompt.fourierDown", c, c);
        Dense(s, "pointPrompt.latentProj", c, latentDim);
        Affine(s, "pointPrompt.outputNorm", c);
    }

    private static void Dense(Dictionary<string, long[]> s, string name, long outDim, long inDim)
    {
        s[name + ".weight"] = [outDim, inDim];
        s[name + ".bias"] = [outDim];
    }

    private static void Affine(Dictionary<string, long[]> s, string name, long dim)
    {
        s[name + ".weight"] = [dim];
        s[name + ".bias"] = [dim];
    }

    private static string Substitute(string name, string? index) =>
        index is null ? name : name.Replace(Index, index, StringComparison.Ordinal);

    private static void CheckShape(string label, Tensor tensor, long[] expected)
    {
        if (tensor.DType != DType.F32)
            throw new HartsyInferenceException($"CoLa3D tensor '{label}' is {tensor.DType}; the decomposer expects F32.");
        if (tensor.Shape.Rank != expected.Length)
            throw new HartsyInferenceException(
                $"CoLa3D tensor '{label}' has rank {tensor.Shape.Rank}; expected {expected.Length} dims {FormatDims(expected)}.");
        for (int i = 0; i < expected.Length; i++)
        {
            if (tensor.Shape[i] != expected[i])
                throw new HartsyInferenceException(
                    $"CoLa3D tensor '{label}' has shape {tensor.Shape}; expected {FormatDims(expected)}.");
        }
    }

    private static string FormatDims(long[] dims) => "[" + string.Join(", ", dims) + "]";

    private static Dictionary<string, string> BuildModules()
    {
        Dictionary<string, string> m = new(StringComparer.Ordinal)
        {
            ["pat.latent_proj"] = "backbone.latentProj",
            ["pat.utonia_proj"] = "backbone.utoniaProj",
            ["pat.latent_ln"] = "backbone.latentNorm",
            ["pat.utonia_ln"] = "backbone.utoniaNorm",
            ["pat.fusion_proj"] = "backbone.fusion",
            ["pat.p_final_norm"] = "backbone.pFinalNorm",
            ["pat.q_final_norm"] = "backbone.qFinalNorm",
            ["pat.part_index_embedder.mlp.0"] = "query.positional.up",
            ["pat.part_index_embedder.mlp.2"] = "query.positional.down",
            ["mask_decoder.point_proj"] = "head.pointProj",
            ["mask_decoder.query_proj"] = "head.queryProj",
            ["mask_prompt_encoder.encoder.#"] = "maskPrompt.conv.#",
            ["mask_prompt_encoder.cnn_norm"] = "maskPrompt.cnnNorm",
            ["mask_prompt_encoder.pe_mlp.0"] = "maskPrompt.peUp",
            ["mask_prompt_encoder.pe_mlp.2"] = "maskPrompt.peDown",
            ["mask_prompt_encoder.pe_norm"] = "maskPrompt.peNorm",
            ["mask_prompt_encoder.fuse.0"] = "maskPrompt.fuseUp",
            ["mask_prompt_encoder.fuse.2"] = "maskPrompt.fuseDown",
            ["point_prompt_encoder.coord_encoder.Wr"] = "pointPrompt.fourierWr",
            ["point_prompt_encoder.coord_encoder.mlp.0"] = "pointPrompt.fourierUp",
            ["point_prompt_encoder.coord_encoder.mlp.2"] = "pointPrompt.fourierDown",
            ["point_prompt_encoder.latent_proj"] = "pointPrompt.latentProj",
            ["point_prompt_encoder.output_norm"] = "pointPrompt.outputNorm",
        };

        // Block modules: torch name → parameter name, with the layer index written as '#'.
        (string Torch, string Name)[] blockModules =
        [
            ("q_norm1", "qSelfNorm"),
            ("q_to_qkv", "qSelfQkv"),
            ("q_self_attn_out", "qSelfOut"),
            ("q_norm2", "qReadNorm"),
            ("q_to_q_p2q", "qReadQuery"),
            ("p_norm_p2q", "pReadNorm"),
            ("p_to_kv", "pReadKv"),
            ("p2q_out", "qReadOut"),
            ("q_norm_ff", "qFfnNorm"),
            ("q_mlp.0", "qFfnUp"),
            ("q_mlp.2", "qFfnDown"),
            ("q_to_kv_q2p", "qWriteKv"),
            ("p_norm", "pWriteNorm"),
            ("p_to_q_q2p", "pWriteQuery"),
            ("q2p_q_rms_norm", "pWriteQkNorm"),
            ("q2p_k_rms_norm", "qWriteKNorm"),
            ("q2p_out", "pWriteOut"),
            ("p_norm_ff", "pFfnNorm"),
            ("p_mlp.mlp.0", "pFfnUp"),
            ("p_mlp.mlp.2", "pFfnDown"),
        ];
        foreach ((string torch, string name) in blockModules)
            m[$"pat.blocks.{Index}.{torch}"] = $"backbone.blocks.{Index}.{name}";
        return m;
    }
}

using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converts a FLUX.3 Klein transformer checkpoint (BFL / ComfyUI single file, optionally under
/// <c>model.diffusion_model.</c>) to the layout <c>Flux3Transformer</c> loads. Names stay ComfyUI's; the only rewrites are
/// splitting a fused <c>qkv</c> into <c>q_proj</c>/<c>k_proj</c>/<c>v_proj</c> (quant-aware, so fp8, int8 and NVFP4
/// companions follow their rows) and accepting <c>.scale</c> for the Q/K norm weights. Bundled <c>vae.</c> and
/// <c>text_encoders.</c> trees are left to their own loaders; any other unrecognized key is refused rather than dropped.</summary>
public static class Flux3CheckpointConverter
{
    /// <summary>Top-level module prefixes a FLUX.3 transformer carries.</summary>
    private static readonly string[] TransformerRoots =
    [
        "emb_in.", "txt_in.", "time_in.", "ref_time_in.", "early_stream_modulations.", "single_stream_modulations.",
        "content_mode_blocks.", "txt_mode_blocks.", "single_blocks.", "final_layer.",
    ];

    /// <summary>Component trees an all-in-one file may bundle, which belong to other loaders.</summary>
    private static readonly string[] BundledRoots = ["vae.", "text_encoders.", "first_stage_model.", "conditioner."];

    /// <summary>ComfyUI's FLUX.3 signature (<c>model_detection.py</c>): a per-mode image block plus the joint text modulation.</summary>
    public static bool MatchesByKeys(IEnumerable<string> keys)
    {
        ArgumentNullException.ThrowIfNull(keys);
        bool hasContentBlock = false;
        bool hasJointModulation = false;
        foreach (string raw in keys)
        {
            string key = CheckpointConvertUtils.StripTransformerPrefix(raw);
            hasContentBlock |= key.StartsWith("content_mode_blocks.image.0.attn_out.", StringComparison.Ordinal);
            hasJointModulation |= key.StartsWith("single_stream_modulations.txt.lin.", StringComparison.Ordinal);
            if (hasContentBlock && hasJointModulation)
                return true;
        }
        return false;
    }

    /// <summary>Converts the transformer weights. Quantization companions must already be folded (open the file through
    /// <c>CheckpointSource</c>).</summary>
    public static Dictionary<string, Tensor> ConvertTransformer(IReadOnlyDictionary<string, Tensor> allWeights)
    {
        ArgumentNullException.ThrowIfNull(allWeights);
        CheckpointConvertUtils.RequireFoldedCompanions(allWeights, nameof(Flux3CheckpointConverter));
        Dictionary<string, Tensor> output = new Dictionary<string, Tensor>(allWeights.Count + 64, StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> entry in allWeights)
        {
            string key = CheckpointConvertUtils.StripTransformerPrefix(entry.Key);
            if (StartsWithAny(key, BundledRoots))
                continue;
            if (!StartsWithAny(key, TransformerRoots))
                throw new NotSupportedException($"FLUX.3 checkpoint key '{entry.Key}' is not part of the FLUX.3 transformer.");
            ConvertKey(key, entry.Value, output);
        }
        if (output.Count == 0)
            throw new InvalidDataException("FLUX.3 checkpoint contains no transformer weights.");
        return output;
    }

    private static void ConvertKey(string key, Tensor tensor, Dictionary<string, Tensor> output)
    {
        const string fusedSuffix = ".qkv.weight";
        if (key.EndsWith(fusedSuffix, StringComparison.Ordinal))
        {
            if (tensor.Shape.Rank != 2 || tensor.Shape[0] % 3 != 0)
                throw new InvalidDataException($"FLUX.3 fused '{key}' has shape {tensor.Shape}; expected [3·D, D].");
            string blockPrefix = key[..^fusedSuffix.Length];
            CheckpointConvertUtils.SplitQkvWeight(tensor, (int)(tensor.Shape[0] / 3), blockPrefix,
                "q_proj", "k_proj", "v_proj", output);
            return;
        }
        if (key.EndsWith("_norm.scale", StringComparison.Ordinal))
        {
            output[key[..^".scale".Length] + ".weight"] = tensor;
            return;
        }
        output[key] = tensor;
    }

    private static bool StartsWithAny(string key, string[] prefixes)
    {
        foreach (string prefix in prefixes)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal))
                return true;
        }
        return false;
    }
}

using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Converts a MOSS-Image transformer checkpoint to the layout <c>MossImageTransformer</c> loads. OpenMOSS's
/// diffusers export (<c>transformer/diffusion_pytorch_model-*.safetensors</c>), its ComfyUI single file and SGLang all
/// use the same module names, so the only rewrites are stripping a <c>model.diffusion_model.</c>/<c>transformer.</c>
/// prefix and dropping <c>rope.bands</c>, which the reference keeps only for checkpoint compatibility and rebuilds in
/// F32 on load. Bundled <c>vae.</c> and <c>text_encoders.</c> trees are left to their own loaders; any other
/// unrecognized key is refused rather than dropped.</summary>
public static class MossImageCheckpointConverter
{
    /// <summary>The keys ComfyUI's loader requires (OpenMOSS <c>comfyui/moss-image/loaders.py</c>).</summary>
    public static IReadOnlyList<string> SignatureKeys { get; } =
    [
        "video_embedder.bias", "text_embedder.proj.weight", "text_embedder.blocks.0.qkv.weight", "shared_block_modulation.weight",
        "layers.0.q_norm.weight", "layers.0.proj.ffn_up_gate.weight", "final_linear_video.weight", "video_head_modulation.weight",
        "time_embedding.0.bias",
    ];

    /// <summary>Top-level module roots of the transformer.</summary>
    private static readonly string[] Roots =
    [
        "video_embedder.", "text_embedder.", "time_embedding.", "shared_block_modulation.", "layers.", "final_norm_video.",
        "final_linear_video.", "video_head_modulation.",
    ];

    /// <summary>Component trees an all-in-one file may bundle, which belong to other loaders.</summary>
    private static readonly string[] BundledRoots = ["vae.", "text_encoders.", "first_stage_model.", "conditioner."];

    /// <summary>The derived RoPE frequency buffer; rebuilt in F32 on load, never read from the file.</summary>
    private const string RopeBandsKey = "rope.bands";

    /// <summary>Whether <paramref name="keys"/> carry the MOSS-Image transformer's signature: the shared block modulation
    /// beside a fused <c>ffn_up_gate</c> joint block. ERNIE-Image's shared adaLN uses other names.</summary>
    public static bool MatchesByKeys(IEnumerable<string> keys)
    {
        ArgumentNullException.ThrowIfNull(keys);
        bool modulation = false, joint = false, head = false;
        foreach (string raw in keys)
        {
            string key = CheckpointConvertUtils.StripTransformerPrefix(raw);
            modulation |= key.StartsWith("shared_block_modulation.", StringComparison.Ordinal);
            joint |= key.StartsWith("layers.0.proj.ffn_up_gate.", StringComparison.Ordinal);
            head |= key.StartsWith("video_head_modulation.", StringComparison.Ordinal);
            if (modulation && joint && head)
                return true;
        }
        return false;
    }

    /// <summary>Converts the transformer weights. Quantization companions must already be folded (open the file through
    /// <c>CheckpointSource</c>).</summary>
    public static Dictionary<string, Tensor> ConvertTransformer(IReadOnlyDictionary<string, Tensor> allWeights)
    {
        ArgumentNullException.ThrowIfNull(allWeights);
        CheckpointConvertUtils.RequireFoldedCompanions(allWeights, nameof(MossImageCheckpointConverter));
        Dictionary<string, Tensor> output = new(allWeights.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> entry in allWeights)
        {
            string key = CheckpointConvertUtils.StripTransformerPrefix(entry.Key);
            if (key == RopeBandsKey || StartsWithAny(key, BundledRoots))
                continue;
            if (!StartsWithAny(key, Roots))
                throw new NotSupportedException($"MOSS-Image checkpoint key '{entry.Key}' is not part of the MOSS-Image transformer.");
            output[key] = entry.Value;
        }
        foreach (string required in SignatureKeys)
        {
            if (!output.ContainsKey(required))
                throw new InvalidDataException($"MOSS-Image checkpoint is incomplete: it lacks '{required}'.");
        }
        return output;
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

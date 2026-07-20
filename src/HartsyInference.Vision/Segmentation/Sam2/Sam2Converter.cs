using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Vision.Segmentation.Sam2;

/// <summary>Splits a Meta SAM 2 checkpoint into per-component weight dictionaries, mirroring the
/// <c>*CheckpointConverter</c> pattern used elsewhere in the engine. SAM 2 checkpoints key every tensor
/// under one of five top-level modules; this groups them so each component loads its own slice.
///
/// <para>The Meta release stores weights under a <c>model.</c> wrapper (Lightning checkpoint); that prefix
/// is stripped. The remaining five groups are the Hiera image encoder, the prompt encoder, the mask
/// decoder, and (video) the memory attention + memory encoder.</para></summary>
public static class Sam2Converter
{
    /// <summary>Per-component weight groups extracted from a SAM 2 checkpoint.</summary>
    public sealed class Groups
    {
        /// <summary>Hiera image encoder + FPN neck (<c>image_encoder.*</c>).</summary>
        public required Dictionary<string, Tensor> ImageEncoder { get; init; }

        /// <summary>Prompt encoder (<c>sam_prompt_encoder.*</c>): Gaussian PE, point embeddings, mask downscaler.</summary>
        public required Dictionary<string, Tensor> PromptEncoder { get; init; }

        /// <summary>Two-way mask decoder (<c>sam_mask_decoder.*</c>).</summary>
        public required Dictionary<string, Tensor> MaskDecoder { get; init; }

        /// <summary>Video memory attention (<c>memory_attention.*</c>); empty for image-only checkpoints.</summary>
        public required Dictionary<string, Tensor> MemoryAttention { get; init; }

        /// <summary>Video memory encoder (<c>memory_encoder.*</c>); empty for image-only checkpoints.</summary>
        public required Dictionary<string, Tensor> MemoryEncoder { get; init; }
    }

    private const string ModelPrefix = "model.";

    /// <summary>Groups an in-memory SAM 2 state dict by component, stripping the Lightning <c>model.</c> wrapper.</summary>
    public static Groups Convert(IReadOnlyDictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> imageEncoder = new();
        Dictionary<string, Tensor> promptEncoder = new();
        Dictionary<string, Tensor> maskDecoder = new();
        Dictionary<string, Tensor> memoryAttention = new();
        Dictionary<string, Tensor> memoryEncoder = new();

        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            string key = kvp.Key.StartsWith(ModelPrefix, StringComparison.Ordinal)
                ? kvp.Key[ModelPrefix.Length..]
                : kvp.Key;

            if (Strip(key, "image_encoder.", out string ie)) imageEncoder[ie] = kvp.Value;
            else if (Strip(key, "sam_prompt_encoder.", out string pe)) promptEncoder[pe] = kvp.Value;
            else if (Strip(key, "sam_mask_decoder.", out string md)) maskDecoder[md] = kvp.Value;
            else if (Strip(key, "memory_attention.", out string ma)) memoryAttention[ma] = kvp.Value;
            else if (Strip(key, "memory_encoder.", out string me)) memoryEncoder[me] = kvp.Value;
            // Unknown keys (e.g. obj_ptr / no_mem_embed scalars) are ignored by component loaders.
        }

        return new Groups
        {
            ImageEncoder = imageEncoder,
            PromptEncoder = promptEncoder,
            MaskDecoder = maskDecoder,
            MemoryAttention = memoryAttention,
            MemoryEncoder = memoryEncoder,
        };
    }

    /// <summary>Loads + groups a single-file safetensors SAM 2 checkpoint. Returns the groups plus the loader
    /// (keep it alive while the tensors are in use).</summary>
    public static (Groups groups, SafeTensorsLoader loader) LoadAndConvert(string safetensorsPath)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(safetensorsPath);
        return (Convert(loader.GetAllTensors()), loader);
    }

    private static bool Strip(string key, string prefix, out string rest)
    {
        if (key.StartsWith(prefix, StringComparison.Ordinal))
        {
            rest = key[prefix.Length..];
            return true;
        }
        rest = key;
        return false;
    }
}

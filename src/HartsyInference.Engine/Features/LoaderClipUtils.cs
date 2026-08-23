using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Features;

/// <summary>Shared staging for standalone CLIP text-encoder files (the split-file <c>clip_l</c>/<c>clip_g</c> layout ComfyUI and SwarmUI ship): unwraps whichever container prefix the publisher used so the keys land in the <c>text_model.*</c> naming <c>ClipTextEncoder.LoadWeights</c> expects.</summary>
public static class LoaderClipUtils
{
    /// <summary>Strips the Comfy (<c>text_encoders.{slot}.transformer.</c>) or LDM (<c>conditioner.embedders.{embedderIndex}.transformer.</c>) wrapper from a standalone CLIP file and drops the <c>position_ids</c> buffer the encoder doesn't consume. Diffusers-native files (already <c>text_model.*</c>, e.g. SDXL's <c>text_encoder/model.fp16.safetensors</c>) pass through untouched, so <c>text_projection.weight</c> survives for CLIP-G's pooled output.</summary>
    /// <param name="slot">Comfy's encoder slot name — "clip_l" or "clip_g".</param>
    /// <param name="embedderIndex">LDM conditioner index for the same encoder — 0 for CLIP-L, 1 for CLIP-G.</param>
    public static Dictionary<string, Tensor> StripClipPrefix(IReadOnlyDictionary<string, Tensor> raw, string slot, int embedderIndex)
    {
        ArgumentNullException.ThrowIfNull(raw);
        ArgumentException.ThrowIfNullOrEmpty(slot);
        return LoaderPrefixUtils.StripPrefixes(
            raw,
            [$"text_encoders.{slot}.transformer.", $"conditioner.embedders.{embedderIndex}.transformer."],
            LoaderPrefixUtils.PositionIdsBuffer);
    }
}

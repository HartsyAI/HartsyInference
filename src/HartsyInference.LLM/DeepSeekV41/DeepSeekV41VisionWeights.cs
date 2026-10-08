using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>The F32 weights of the vision tower (patch embedding, blocks, final norm), named after the checkpoint's <c>vision.*</c> keys.</summary>
/// <param name="PatchProj">Patch embedding projection, <c>[dim, 3 * patch * patch]</c>.</param>
/// <param name="PatchBias">Its bias, <c>[dim]</c>.</param>
/// <param name="Blocks">The blocks in order.</param>
/// <param name="FinalNorm">RMSNorm weight after the last block, <c>[dim]</c>.</param>
public sealed record DeepSeekV41VisionWeights(Tensor PatchProj, Tensor PatchBias, IReadOnlyList<DeepSeekV41VisionBlockWeights> Blocks, Tensor FinalNorm)
{
    /// <summary>Checks every tensor against <paramref name="config"/>.</summary>
    /// <exception cref="HartsyInferenceException">A tensor is missing, not F32 or has the wrong shape, or the block count differs from the config.</exception>
    internal void Validate(DeepSeekV41VisionConfig config)
    {
        DeepSeekV41VisionTensors.Require(PatchProj, "vision.patch_embed.proj.weight", config.HiddenSize, config.PatchInputDim);
        DeepSeekV41VisionTensors.Require(PatchBias, "vision.patch_embed.proj.bias", config.HiddenSize);
        DeepSeekV41VisionTensors.Require(FinalNorm, "vision.norm.weight", config.HiddenSize);
        if (Blocks is null || Blocks.Count != config.NumLayers)
            throw new HartsyInferenceException($"The vision tower has {Blocks?.Count ?? 0} blocks of weights, the config says {config.NumLayers}.");
        for (int i = 0; i < Blocks.Count; i++) Blocks[i].Validate(config, i);
    }

    internal void Dispose()
    {
        PatchProj?.Dispose();
        PatchBias?.Dispose();
        FinalNorm?.Dispose();
        if (Blocks is null) return;
        foreach (DeepSeekV41VisionBlockWeights block in Blocks) block.Dispose();
    }
}

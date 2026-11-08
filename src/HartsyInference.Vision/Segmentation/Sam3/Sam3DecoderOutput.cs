using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Final-layer outputs of the SAM 3 detector transformer for one prompt (batch 1, F32).</summary>
/// <param name="Boxes">Predicted boxes in normalized <c>cxcywh</c>, <c>[1, NumQueries, 4]</c>.</param>
/// <param name="QueryHidden">Decoder query states, <c>[1, NumQueries, DModel]</c>.</param>
/// <param name="PresenceLogit">Presence-token logit, <c>[1, 1]</c>.</param>
/// <param name="EncoderMemory">Fusion-encoder output reshaped to the feature grid, <c>[1, DModel, 72, 72]</c>.</param>
/// <param name="PromptFeatures">Prompt tokens fed to the scorer (text plus the geometry CLS token), <c>[1, Seq, DModel]</c>.</param>
/// <param name="PromptPaddingMask"><c>true</c> where a prompt token is padding.</param>
public sealed record Sam3DecoderOutput(
    Tensor Boxes,
    Tensor QueryHidden,
    Tensor PresenceLogit,
    Tensor EncoderMemory,
    Tensor PromptFeatures,
    bool[] PromptPaddingMask);

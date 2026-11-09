using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Caller-built position tables for one forward of a decoder whose gated cross-attention layers read image features
/// with their own rotary positions (MOSS-VL). They replace the decoder's built-in position ramp for that call.</summary>
/// <remarks>
/// <see cref="TextCos"/>/<see cref="TextSin"/> are <c>[1, T, headDim]</c>: the rows of self-attention, and of the cross-attention
/// queries after their per-head norm. <see cref="VisionCos"/>/<see cref="VisionSin"/> are <c>[1, L, headDim]</c>: the cross-attention
/// keys after their per-head norm. <see cref="RowMask"/> is <c>[1, T, hidden]</c> F32, each row filled with 1 where the text row sees
/// an image and 0 where it sees none; a 0 row contributes nothing from its cross-attention and MLP sublayers. The full width lets
/// the layer apply it with an elementwise multiply. Every tensor is borrowed: the caller owns and disposes it.
/// </remarks>
public sealed record MultimodalRope(Tensor TextCos, Tensor TextSin, Tensor VisionCos, Tensor VisionSin, Tensor? RowMask);

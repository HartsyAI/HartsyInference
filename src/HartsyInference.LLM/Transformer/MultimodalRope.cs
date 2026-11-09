using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Caller-built position tables for one forward of a decoder whose gated cross-attention layers read image features
/// with their own rotary positions (MOSS-VL). They replace the decoder's built-in position ramp for that call.</summary>
/// <remarks>
/// <see cref="TextCos"/>/<see cref="TextSin"/> are <c>[1, T, headDim]</c>: the rows of self-attention, and of the cross-attention
/// queries after their per-head norm. <see cref="VisionCos"/>/<see cref="VisionSin"/> are <c>[1, L, headDim]</c>: the cross-attention
/// keys after their per-head norm. <see cref="RowMask"/> is <c>[1, T, hidden]</c> F32, each row filled with 1 where the text row sees
/// an image and 0 where it sees none; a 0 row contributes nothing from its cross-attention and MLP sublayers. The full width lets
/// the layer apply it with an elementwise multiply. <see cref="ImageGroups"/>, when set, splits the cross-attention by visible image
/// set (multi-image prompts); null means every row attends every vision key. Every tensor is borrowed: the caller owns and disposes it.
/// The vision tables are null for a forward that brings no new vision tokens (a decode step over vision already cached).
/// </remarks>
public sealed record MultimodalRope(Tensor TextCos, Tensor TextSin, Tensor? VisionCos, Tensor? VisionSin, Tensor? RowMask,
    IReadOnlyList<CrossImageGroup>? ImageGroups = null);

/// <summary>One group of text rows that see the same images, for multi-image cross-attention.</summary>
/// <param name="KvLength">The vision keys the group attends: the blocks of its images and every image before them, a prefix of the sequence.</param>
/// <param name="Select">The group's rows as <c>[1, T, hidden]</c> F32, 1 for each row in the group and 0 elsewhere. The group's
/// attention output is multiplied by it, so the other groups' rows contribute nothing. Required unless the range is given; a ranged
/// group may leave it null (with one group per video frame the masks alone would be hundreds of megabytes).</param>
/// <param name="RowStart">When the group's rows are one contiguous range, its first row; -1 when unknown. With every group's range
/// known, the cross layer attends each group's own rows only, instead of every row once per group.</param>
/// <param name="RowCount">The number of rows in that range.</param>
public sealed record CrossImageGroup(int KvLength, Tensor? Select, int RowStart = -1, int RowCount = 0)
{
    /// <summary>The contiguous row range of <paramref name="rowSelect"/> (1 = in the group), or (-1, 0) when the rows are not one range.</summary>
    public static (int Start, int Count) ContiguousRange(ReadOnlySpan<float> rowSelect)
    {
        int first = rowSelect.IndexOfAnyExcept(0f);
        if (first < 0) return (-1, 0);
        int end = first;
        while (end < rowSelect.Length && rowSelect[end] != 0f) end++;
        return rowSelect[end..].IndexOfAnyExcept(0f) >= 0 ? (-1, 0) : (first, end - first);
    }
}

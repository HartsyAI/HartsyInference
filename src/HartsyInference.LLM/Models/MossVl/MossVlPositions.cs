using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>One image of a MOSS-VL prompt: its merged patch grid (<c>H/merge</c> by <c>W/merge</c>). Each image is one placeholder
/// token in the text and one separator-terminated block of merged tokens in the vision sequence.</summary>
public readonly record struct MossVlImage(int MergedHeight, int MergedWidth);

/// <summary>Multimodal positions of one MOSS-VL prompt (single frame per image), following the modeling code's
/// <c>compute_position_ids</c> and <c>compute_vision_position_ids</c>.</summary>
/// <remarks>
/// Text tokens advance all three axes together. An image placeholder sits at <c>t0 + max(h, w)</c>, where <c>t0</c> is the position
/// the image starts at, and the text after it resumes <c>max(h, w) + 1</c> further on. The image's merged token <c>(y, x)</c> sits
/// at <c>(t0, t0 + y, t0 + x)</c>; its separator, the token after the grid, sits at <c>t0 + max(h, w)</c> on all axes.
/// Only single-image prompts get a row mask, because a text row's visible image set changes with its position past several images.
/// </remarks>
public sealed class MossVlPositionPlan
{
    /// <summary>Axis-major text positions <c>[3 · T]</c> (temporal, height, width).</summary>
    public required int[] Text { get; init; }

    /// <summary>Axis-major vision positions <c>[3 · L]</c>, where <c>L = Σ (h·w + 1)</c> over the images.</summary>
    public required int[] Vision { get; init; }

    /// <summary>Vision tokens in the cross-attention sequence, separators included.</summary>
    public required int VisionLength { get; init; }

    /// <summary>Per text row, 1 when the row sees the image, 0 when it sees none; null when the prompt has no image.</summary>
    public required float[]? RowMask { get; init; }
}

/// <summary>Builds MOSS-VL positions and the multimodal rotary tables that the decoder's cross-attention consumes.</summary>
public static class MossVlPositions
{
    /// <summary>Plans positions for <paramref name="tokenIds"/>, where each occurrence of <paramref name="imageTokenId"/> is one image
    /// in <paramref name="images"/> order.</summary>
    /// <exception cref="HartsyInferenceException">The placeholder count differs from the image count, or several images are given
    /// (a per-row column mask over several images is not implemented).</exception>
    public static MossVlPositionPlan Plan(ReadOnlySpan<int> tokenIds, int imageTokenId, ReadOnlySpan<MossVlImage> images)
    {
        int t = tokenIds.Length;
        int placeholders = 0;
        foreach (int id in tokenIds) if (id == imageTokenId) placeholders++;
        if (placeholders != images.Length)
            throw new HartsyInferenceException($"MOSS-VL positions: {placeholders} image placeholder(s) in the prompt for {images.Length} image(s).");
        if (images.Length > 1)
            throw new HartsyInferenceException("MOSS-VL positions: multi-image cross-attention masks are not implemented yet; pass one image.");

        int visionLength = 0;
        foreach (MossVlImage image in images)
        {
            if (image.MergedHeight <= 0 || image.MergedWidth <= 0)
                throw new HartsyInferenceException($"MOSS-VL positions: image grid {image.MergedHeight}x{image.MergedWidth} is empty.");
            visionLength += image.MergedHeight * image.MergedWidth + 1;
        }

        int[] text = new int[3 * t];
        int[] vision = new int[3 * visionLength];
        int regularBefore = 0, shift = 0, imageIndex = 0, visionOffset = 0, firstImageRow = -1;
        for (int i = 0; i < t; i++)
        {
            if (tokenIds[i] != imageTokenId)
            {
                for (int a = 0; a < 3; a++) text[a * t + i] = regularBefore + shift;
                regularBefore++;
                continue;
            }

            MossVlImage image = images[imageIndex++];
            int h = image.MergedHeight, w = image.MergedWidth;
            int t0 = regularBefore + shift;
            int span = Math.Max(h, w);
            for (int a = 0; a < 3; a++) text[a * t + i] = t0 + span;
            if (firstImageRow < 0) firstImageRow = i;

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int col = visionOffset + y * w + x;
                    vision[col] = t0;
                    vision[visionLength + col] = t0 + y;
                    vision[2 * visionLength + col] = t0 + x;
                }
            }
            int sep = visionOffset + h * w;
            vision[sep] = t0 + span;
            vision[visionLength + sep] = t0 + span;
            vision[2 * visionLength + sep] = t0 + span;

            visionOffset += h * w + 1;
            shift += span + 1;
        }

        float[]? rowMask = null;
        if (images.Length == 1)
        {
            rowMask = new float[t];
            for (int i = firstImageRow; i < t; i++) rowMask[i] = 1f;
        }

        return new MossVlPositionPlan { Text = text, Vision = vision, VisionLength = visionLength, RowMask = rowMask };
    }

    /// <summary>Builds the rotary tables for <paramref name="plan"/> from a MOSS-VL configuration's text section.</summary>
    public static MossVlRope BuildRope(MossVlPositionPlan plan, MossVlConfig config) =>
        BuildRope(plan, config.Text.HeadDim, config.Text.RopeTheta, [.. config.MropeSection], config.MropeInterleaved);

    /// <summary>Builds the rotary tables for <paramref name="plan"/>: pair <c>j</c> of each row rotates by its interleaved M-RoPE axis,
    /// as the modeling code's <c>apply_interleaved_mrope</c> does. The result owns its tensors.</summary>
    public static MossVlRope BuildRope(MossVlPositionPlan plan, int headDim, double theta, ReadOnlySpan<int> mropeSection, bool interleaved)
    {
        int t = plan.Text.Length / 3;
        int maxPosition = 0;
        foreach (int p in plan.Text) maxPosition = Math.Max(maxPosition, p);
        foreach (int p in plan.Vision) maxPosition = Math.Max(maxPosition, p);

        (double[] invFreq, double scaling) = RopeFrequencyBuilder.Build(headDim, theta, null, maxPosition + 1);
        byte[] pairAxes = MRopePairAxes.Build(mropeSection, headDim / 2, interleaved ? MRopeLayout.Interleaved : MRopeLayout.Sectioned);

        (Tensor textCos, Tensor textSin) = MRopeTable.BuildTensors(plan.Text, t, 0, t, invFreq, scaling, pairAxes, headDim, headDim);
        try
        {
            (Tensor visionCos, Tensor visionSin) = MRopeTable.BuildTensors(plan.Vision, plan.VisionLength, 0, plan.VisionLength,
                invFreq, scaling, pairAxes, headDim, headDim);
            try
            {
                Tensor? rowMask = null;
                if (plan.RowMask is not null)
                {
                    rowMask = new Tensor(new TensorShape(t), DType.F32);
                    plan.RowMask.AsSpan().CopyTo(rowMask.AsSpan<float>());
                }
                return new MossVlRope(new MultimodalRope(textCos, textSin, visionCos, visionSin, rowMask));
            }
            catch
            {
                visionCos.Dispose();
                visionSin.Dispose();
                throw;
            }
        }
        catch
        {
            textCos.Dispose();
            textSin.Dispose();
            throw;
        }
    }
}

/// <summary>The rotary tables of one MOSS-VL forward. Dispose after the forward that reads them.</summary>
public sealed class MossVlRope(MultimodalRope rope) : IDisposable
{
    private int _disposed;

    /// <summary>The tables to pass to <see cref="GenericTransformer.ForwardEmbeds"/>.</summary>
    public MultimodalRope Rope { get; } = rope;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Rope.TextCos.Dispose();
        Rope.TextSin.Dispose();
        Rope.VisionCos.Dispose();
        Rope.VisionSin.Dispose();
        Rope.RowMask?.Dispose();
    }
}

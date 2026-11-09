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
/// A text row sees an image when the row is at or after that image's placeholder (the reference's cumulative image-token mask).
/// </remarks>
public sealed class MossVlPositionPlan
{
    /// <summary>The additive score bias of a key the row does not see: far enough below any score that its softmax weight is zero,
    /// finite so a row that sees no image still normalizes (its output is then zeroed by the row mask).</summary>
    public const float MaskedBias = -1e9f;

    /// <summary>Axis-major text positions <c>[3 · T]</c> (temporal, height, width).</summary>
    public required int[] Text { get; init; }

    /// <summary>Axis-major vision positions <c>[3 · L]</c>, where <c>L = Σ (h·w + 1)</c> over the images.</summary>
    public required int[] Vision { get; init; }

    /// <summary>Vision tokens in the cross-attention sequence, separators included.</summary>
    public required int VisionLength { get; init; }

    /// <summary>Per text row, 1 when the row sees an image, 0 when it sees none; null when the prompt has no image.</summary>
    public required float[]? RowMask { get; init; }

    /// <summary>Row-major <c>[T · L]</c> additive score bias: 0 where a text row sees a vision key's image, <see cref="MaskedBias"/>
    /// where it does not. Null for one image (the row mask alone is exact there) and for a prompt with no image.</summary>
    public float[]? CrossBias { get; init; }
}

/// <summary>Builds MOSS-VL positions and the multimodal rotary tables that the decoder's cross-attention consumes.</summary>
public static class MossVlPositions
{
    /// <summary>Plans positions for <paramref name="tokenIds"/>, where each occurrence of <paramref name="imageTokenId"/> is one image
    /// in <paramref name="images"/> order.</summary>
    /// <exception cref="HartsyInferenceException">The placeholder count differs from the image count, or an image grid is empty.</exception>
    public static MossVlPositionPlan Plan(ReadOnlySpan<int> tokenIds, int imageTokenId, ReadOnlySpan<MossVlImage> images)
    {
        int t = tokenIds.Length;
        int placeholders = 0;
        foreach (int id in tokenIds) if (id == imageTokenId) placeholders++;
        if (placeholders != images.Length)
            throw new HartsyInferenceException($"MOSS-VL positions: {placeholders} image placeholder(s) in the prompt for {images.Length} image(s).");

        int visionLength = 0;
        foreach (MossVlImage image in images)
        {
            if (image.MergedHeight <= 0 || image.MergedWidth <= 0)
                throw new HartsyInferenceException($"MOSS-VL positions: image grid {image.MergedHeight}x{image.MergedWidth} is empty.");
            visionLength += image.MergedHeight * image.MergedWidth + 1;
        }

        int[] text = new int[3 * t];
        int[] vision = new int[3 * visionLength];
        int[] placeholderRow = new int[images.Length];   // the text row of each image's placeholder
        int[] blockStart = new int[images.Length];       // the first vision column of each image's block
        int regularBefore = 0, shift = 0, imageIndex = 0, visionOffset = 0;
        for (int i = 0; i < t; i++)
        {
            if (tokenIds[i] != imageTokenId)
            {
                for (int a = 0; a < 3; a++) text[a * t + i] = regularBefore + shift;
                regularBefore++;
                continue;
            }

            int k = imageIndex++;
            MossVlImage image = images[k];
            int h = image.MergedHeight, w = image.MergedWidth;
            int t0 = regularBefore + shift;
            int span = Math.Max(h, w);
            for (int a = 0; a < 3; a++) text[a * t + i] = t0 + span;
            placeholderRow[k] = i;
            blockStart[k] = visionOffset;

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
        float[]? crossBias = null;
        if (images.Length > 0)
        {
            // A row sees the images whose placeholder is at or before it; it sees none before the first placeholder.
            rowMask = new float[t];
            for (int i = placeholderRow[0]; i < t; i++) rowMask[i] = 1f;
        }
        if (images.Length > 1)
        {
            crossBias = new float[t * visionLength];
            for (int r = 0; r < t; r++)
            {
                for (int k = 0; k < images.Length; k++)
                {
                    float bias = r >= placeholderRow[k] ? 0f : MossVlPositionPlan.MaskedBias;
                    int end = blockStart[k] + images[k].MergedHeight * images[k].MergedWidth + 1;
                    for (int j = blockStart[k]; j < end; j++) crossBias[r * visionLength + j] = bias;
                }
            }
        }

        return new MossVlPositionPlan { Text = text, Vision = vision, VisionLength = visionLength, RowMask = rowMask, CrossBias = crossBias };
    }

    /// <summary>Builds the rotary tables for <paramref name="plan"/> from a MOSS-VL configuration's text section.</summary>
    public static MossVlRope BuildRope(MossVlPositionPlan plan, MossVlConfig config) =>
        BuildRope(plan, config.Text.HeadDim, config.Text.RopeTheta, [.. config.MropeSection], config.MropeInterleaved, config.Text.HiddenSize);

    /// <summary>The vision rotary tables for the image in <paramref name="plan"/>. They depend only on the image, so a decode session
    /// builds them once and borrows them into every step (<see cref="BuildRopeOverVisionTables"/>). The result owns its tensors.</summary>
    public static (Tensor Cos, Tensor Sin) BuildVisionTables(MossVlPositionPlan plan, MossVlConfig config)
    {
        int headDim = config.Text.HeadDim;
        // Null scaling makes the frequencies independent of the position range, so the tables match a per-step build exactly.
        (double[] invFreq, double scaling) = RopeFrequencyBuilder.Build(headDim, config.Text.RopeTheta, null, 1);
        byte[] pairAxes = MRopePairAxes.Build([.. config.MropeSection], headDim / 2,
            config.MropeInterleaved ? MRopeLayout.Interleaved : MRopeLayout.Sectioned);
        return MRopeTable.BuildTensors(plan.Vision, plan.VisionLength, 0, plan.VisionLength, invFreq, scaling, pairAxes, headDim, headDim);
    }

    /// <summary>The rotary tables for the text rows of <paramref name="plan"/> (the whole prompt, or one decode step) over the image's vision
    /// tables. The text tables, row mask and cross bias are built fresh; the vision tables are borrowed, so the caller keeps them alive and
    /// disposes them. Text positions match <see cref="BuildRope(MossVlPositionPlan, MossVlConfig)"/>.</summary>
    public static MossVlRope BuildRopeOverVisionTables(MossVlPositionPlan plan, MossVlConfig config, Tensor visionCos, Tensor visionSin)
    {
        int headDim = config.Text.HeadDim, hidden = config.Text.HiddenSize, t = plan.Text.Length / 3;
        int maxPosition = 0;
        foreach (int p in plan.Text) maxPosition = Math.Max(maxPosition, p);
        (double[] invFreq, double scaling) = RopeFrequencyBuilder.Build(headDim, config.Text.RopeTheta, null, maxPosition + 1);
        byte[] pairAxes = MRopePairAxes.Build([.. config.MropeSection], headDim / 2,
            config.MropeInterleaved ? MRopeLayout.Interleaved : MRopeLayout.Sectioned);
        (Tensor textCos, Tensor textSin) = MRopeTable.BuildTensors(plan.Text, t, 0, t, invFreq, scaling, pairAxes, headDim, headDim);
        Tensor? rowMask = null, crossBias = null;
        try
        {
            rowMask = plan.RowMask is null ? null : BuildRowMask(plan.RowMask, t, hidden);
            crossBias = BuildCrossBias(plan, t);
            return new MossVlRope(new MultimodalRope(textCos, textSin, visionCos, visionSin, rowMask, crossBias), ownsVision: false);
        }
        catch
        {
            textCos.Dispose();
            textSin.Dispose();
            rowMask?.Dispose();
            crossBias?.Dispose();
            throw;
        }
    }

    /// <summary>Expands a per-row mask to <c>[1, T, hiddenSize]</c>, each row filled with its value.</summary>
    private static Tensor BuildRowMask(float[] rows, int t, int hiddenSize)
    {
        Tensor rowMask = new(new TensorShape(1, t, hiddenSize), DType.F32);
        Span<float> dst = rowMask.AsSpan<float>();
        for (int r = 0; r < t; r++)
            dst.Slice(r * hiddenSize, hiddenSize).Fill(rows[r]);
        return rowMask;
    }

    /// <summary>Copies the plan's cross bias to <c>[1, 1, T, L]</c>, or returns null when the plan has none.</summary>
    private static Tensor? BuildCrossBias(MossVlPositionPlan plan, int t)
    {
        if (plan.CrossBias is null) return null;
        Tensor bias = new(new TensorShape(1, 1, t, plan.VisionLength), DType.F32);
        plan.CrossBias.AsSpan().CopyTo(bias.AsSpan<float>());
        return bias;
    }

    /// <summary>Builds the rotary tables for <paramref name="plan"/>: pair <c>j</c> of each row rotates by its interleaved M-RoPE axis,
    /// as the modeling code's <c>apply_interleaved_mrope</c> does. The row mask is expanded to <c>[1, T, hiddenSize]</c> so the
    /// cross-attention layers can apply it with one elementwise multiply. The result owns its tensors.</summary>
    public static MossVlRope BuildRope(MossVlPositionPlan plan, int headDim, double theta, ReadOnlySpan<int> mropeSection, bool interleaved,
        int hiddenSize)
    {
        int t = plan.Text.Length / 3;
        int maxPosition = 0;
        foreach (int p in plan.Text) maxPosition = Math.Max(maxPosition, p);
        foreach (int p in plan.Vision) maxPosition = Math.Max(maxPosition, p);

        (double[] invFreq, double scaling) = RopeFrequencyBuilder.Build(headDim, theta, null, maxPosition + 1);
        byte[] pairAxes = MRopePairAxes.Build(mropeSection, headDim / 2, interleaved ? MRopeLayout.Interleaved : MRopeLayout.Sectioned);

        Tensor? textCos = null, textSin = null, visionCos = null, visionSin = null, rowMask = null, crossBias = null;
        try
        {
            (textCos, textSin) = MRopeTable.BuildTensors(plan.Text, t, 0, t, invFreq, scaling, pairAxes, headDim, headDim);
            (visionCos, visionSin) = MRopeTable.BuildTensors(plan.Vision, plan.VisionLength, 0, plan.VisionLength, invFreq, scaling, pairAxes,
                headDim, headDim);
            rowMask = plan.RowMask is null ? null : BuildRowMask(plan.RowMask, t, hiddenSize);
            crossBias = BuildCrossBias(plan, t);
            return new MossVlRope(new MultimodalRope(textCos!, textSin!, visionCos!, visionSin!, rowMask, crossBias));
        }
        catch
        {
            textCos?.Dispose();
            textSin?.Dispose();
            visionCos?.Dispose();
            visionSin?.Dispose();
            rowMask?.Dispose();
            crossBias?.Dispose();
            throw;
        }
    }
}

/// <summary>The rotary tables of one MOSS-VL forward. Dispose after the forward that reads them.</summary>
/// <param name="rope">The tables. Text tables, the row mask and the cross bias are always owned here.</param>
/// <param name="ownsVision">False when the vision tables are borrowed from a longer-lived owner (the decode session's per-image tables).</param>
public sealed class MossVlRope(MultimodalRope rope, bool ownsVision = true) : IDisposable
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
        if (ownsVision)
        {
            Rope.VisionCos.Dispose();
            Rope.VisionSin.Dispose();
        }
        Rope.RowMask?.Dispose();
        Rope.CrossBias?.Dispose();
    }
}

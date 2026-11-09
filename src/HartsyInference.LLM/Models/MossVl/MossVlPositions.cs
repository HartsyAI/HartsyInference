using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>One image of a MOSS-VL prompt: its merged patch grid (<c>H/merge</c> by <c>W/merge</c>). Each image is one placeholder
/// token in the text and one separator-terminated block of merged tokens in the vision sequence.</summary>
public readonly record struct MossVlImage(int MergedHeight, int MergedWidth);

/// <summary>One group of text rows of a multi-image prompt that see the same images (see <see cref="MossVlPositionPlan.CrossGroups"/>).</summary>
/// <param name="KvLength">The vision keys the group attends: its images' blocks and every image before them, a prefix of the sequence.</param>
/// <param name="RowSelect">Per text row, 1 when the row is in the group, 0 otherwise.</param>
public sealed record MossVlCrossGroup(int KvLength, float[] RowSelect);

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
    /// <summary>Axis-major text positions <c>[3 · T]</c> (temporal, height, width).</summary>
    public required int[] Text { get; init; }

    /// <summary>Axis-major vision positions <c>[3 · L]</c>, where <c>L = Σ (h·w + 1)</c> over the images.</summary>
    public required int[] Vision { get; init; }

    /// <summary>Vision tokens in the cross-attention sequence, separators included.</summary>
    public required int VisionLength { get; init; }

    /// <summary>Per text row, 1 when the row sees an image, 0 when it sees none; null when the prompt has no image.</summary>
    public required float[]? RowMask { get; init; }

    /// <summary>For a multi-image prompt, one group per image: the rows whose last visible image is that image, and the vision prefix
    /// through it. Each group's rows attend only that prefix, so no score bias is needed. Null for one image (the row mask alone is
    /// exact there) and for a prompt with no image.</summary>
    public IReadOnlyList<MossVlCrossGroup>? CrossGroups { get; init; }

    /// <summary>The position the next token after this segment takes: one past its last text position.</summary>
    public int NextPosition { get; init; }
}

/// <summary>Builds MOSS-VL positions and the multimodal rotary tables that the decoder's cross-attention consumes.</summary>
public static class MossVlPositions
{
    /// <summary>Plans positions for a whole prompt <paramref name="tokenIds"/>, where each occurrence of <paramref name="imageTokenId"/> is one
    /// image in <paramref name="images"/> order. Same as <see cref="PlanSegment"/> starting at position 0 with no vision before it.</summary>
    /// <exception cref="HartsyInferenceException">The placeholder count differs from the image count, or an image grid is empty.</exception>
    public static MossVlPositionPlan Plan(ReadOnlySpan<int> tokenIds, int imageTokenId, ReadOnlySpan<MossVlImage> images) =>
        PlanSegment(tokenIds, imageTokenId, images, startPosition: 0, priorVisionLength: 0);

    /// <summary>Plans one segment of a stream that already holds <paramref name="priorVisionLength"/> vision tokens and whose next position is
    /// <paramref name="startPosition"/>: the realtime model's <c>_compute_realtime_mrope_for_segment</c>, which is the whole-prompt rule
    /// shifted to the segment's start.</summary>
    /// <remarks>
    /// <para><see cref="MossVlPositionPlan.Vision"/> and <see cref="MossVlPositionPlan.VisionLength"/> cover only this segment's images (the
    /// tokens this forward appends). <see cref="MossVlPositionPlan.RowMask"/> and <see cref="MossVlPositionPlan.CrossGroups"/> describe the
    /// whole stream: a row sees every earlier image, so once the stream holds an image before the segment every row sees something and the
    /// row mask is dropped, and a group's <see cref="MossVlCrossGroup.KvLength"/> counts the prior vision too. Groups are built only when
    /// rows differ in what they see.</para>
    /// </remarks>
    /// <exception cref="HartsyInferenceException">The placeholder count differs from the image count, or an image grid is empty.</exception>
    public static MossVlPositionPlan PlanSegment(ReadOnlySpan<int> tokenIds, int imageTokenId, ReadOnlySpan<MossVlImage> images,
        int startPosition, int priorVisionLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(startPosition);
        ArgumentOutOfRangeException.ThrowIfNegative(priorVisionLength);
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
        int[] blockEnd = new int[images.Length];         // one past the last vision column of each image's block
        int cur = startPosition, imageIndex = 0, visionOffset = 0;
        for (int i = 0; i < t; i++)
        {
            if (tokenIds[i] != imageTokenId)
            {
                for (int a = 0; a < 3; a++) text[a * t + i] = cur;
                cur++;
                continue;
            }

            int k = imageIndex++;
            MossVlImage image = images[k];
            int h = image.MergedHeight, w = image.MergedWidth;
            int t0 = cur;
            int span = Math.Max(h, w);
            for (int a = 0; a < 3; a++) text[a * t + i] = t0 + span;
            placeholderRow[k] = i;

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
            blockEnd[k] = visionOffset;
            cur = t0 + span + 1;
        }

        // What each row attends: the prior vision plus the blocks of this segment's images at or before it (0 = nothing yet).
        int[] rowKv = new int[t];
        for (int r = 0, seen = 0; r < t; r++)
        {
            while (seen < images.Length && placeholderRow[seen] <= r) seen++;
            rowKv[r] = priorVisionLength + (seen == 0 ? 0 : blockEnd[seen - 1]);
        }
        int total = priorVisionLength + visionLength;

        float[]? rowMask = null;
        if (total > 0 && rowKv.Any(kv => kv == 0))
        {
            // Rows before the first image of the stream see none: their cross-attention and MLP contributions are zeroed.
            rowMask = new float[t];
            for (int r = 0; r < t; r++) rowMask[r] = rowKv[r] > 0 ? 1f : 0f;
        }

        IReadOnlyList<MossVlCrossGroup>? groups = null;
        int[] distinct = rowKv.Where(kv => kv > 0).Distinct().Order().ToArray();
        if (distinct.Length > 1 || (distinct.Length == 1 && distinct[0] != total))
        {
            // Rows are in stream order, so each distinct visible prefix is one contiguous run of rows attending that prefix.
            MossVlCrossGroup[] built = new MossVlCrossGroup[distinct.Length];
            for (int g = 0; g < distinct.Length; g++)
            {
                float[] select = new float[t];
                for (int r = 0; r < t; r++) if (rowKv[r] == distinct[g]) select[r] = 1f;
                built[g] = new MossVlCrossGroup(distinct[g], select);
            }
            groups = built;
        }

        return new MossVlPositionPlan
        {
            Text = text, Vision = vision, VisionLength = visionLength, RowMask = rowMask, CrossGroups = groups, NextPosition = cur,
        };
    }

    /// <summary>The rotary tables for one segment: text tables for its rows and, when it brings images, vision tables for their tokens.
    /// The result owns every tensor.</summary>
    public static MossVlRope BuildSegmentRope(MossVlPositionPlan plan, MossVlConfig config)
    {
        int headDim = config.Text.HeadDim, hidden = config.Text.HiddenSize, t = plan.Text.Length / 3;
        // Null scaling makes the frequencies independent of the position range, so a segment's tables match a whole-prompt build.
        (double[] invFreq, double scaling) = RopeFrequencyBuilder.Build(headDim, config.Text.RopeTheta, null, 1);
        byte[] pairAxes = MRopePairAxes.Build([.. config.MropeSection], headDim / 2,
            config.MropeInterleaved ? MRopeLayout.Interleaved : MRopeLayout.Sectioned);
        Tensor? textCos = null, textSin = null, visionCos = null, visionSin = null, rowMask = null;
        List<CrossImageGroup>? groups = null;
        try
        {
            (textCos, textSin) = MRopeTable.BuildTensors(plan.Text, t, 0, t, invFreq, scaling, pairAxes, headDim, headDim);
            if (plan.VisionLength > 0)
                (visionCos, visionSin) = MRopeTable.BuildTensors(plan.Vision, plan.VisionLength, 0, plan.VisionLength, invFreq, scaling,
                    pairAxes, headDim, headDim);
            rowMask = plan.RowMask is null ? null : BuildRowMask(plan.RowMask, t, hidden);
            groups = BuildImageGroups(plan, t, hidden);
            return new MossVlRope(new MultimodalRope(textCos!, textSin!, visionCos, visionSin, rowMask, groups));
        }
        catch
        {
            textCos?.Dispose();
            textSin?.Dispose();
            visionCos?.Dispose();
            visionSin?.Dispose();
            rowMask?.Dispose();
            DisposeGroups(groups);
            throw;
        }
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
    /// tables. The text tables, row mask and image groups are built fresh; the vision tables are borrowed, so the caller keeps them alive and
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
        Tensor? rowMask = null;
        List<CrossImageGroup>? groups = null;
        try
        {
            rowMask = plan.RowMask is null ? null : BuildRowMask(plan.RowMask, t, hidden);
            groups = BuildImageGroups(plan, t, hidden);
            return new MossVlRope(new MultimodalRope(textCos, textSin, visionCos, visionSin, rowMask, groups), ownsVision: false);
        }
        catch
        {
            textCos.Dispose();
            textSin.Dispose();
            rowMask?.Dispose();
            DisposeGroups(groups);
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

    /// <summary>The plan's image groups as device-shaped <c>[1, T, hidden]</c> selections, or null when the plan has none. The caller owns
    /// the result and disposes it with <see cref="DisposeGroups"/> if a later step fails.</summary>
    private static List<CrossImageGroup>? BuildImageGroups(MossVlPositionPlan plan, int t, int hiddenSize)
    {
        if (plan.CrossGroups is null) return null;
        List<CrossImageGroup> built = [];
        try
        {
            foreach (MossVlCrossGroup group in plan.CrossGroups)
                built.Add(new CrossImageGroup(group.KvLength, BuildRowMask(group.RowSelect, t, hiddenSize)));
            return built;
        }
        catch
        {
            DisposeGroups(built);
            throw;
        }
    }

    private static void DisposeGroups(IEnumerable<CrossImageGroup>? groups)
    {
        if (groups is null) return;
        foreach (CrossImageGroup group in groups) group.Select.Dispose();
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

        Tensor? textCos = null, textSin = null, visionCos = null, visionSin = null, rowMask = null;
        List<CrossImageGroup>? groups = null;
        try
        {
            (textCos, textSin) = MRopeTable.BuildTensors(plan.Text, t, 0, t, invFreq, scaling, pairAxes, headDim, headDim);
            (visionCos, visionSin) = MRopeTable.BuildTensors(plan.Vision, plan.VisionLength, 0, plan.VisionLength, invFreq, scaling, pairAxes,
                headDim, headDim);
            rowMask = plan.RowMask is null ? null : BuildRowMask(plan.RowMask, t, hiddenSize);
            groups = BuildImageGroups(plan, t, hiddenSize);
            return new MossVlRope(new MultimodalRope(textCos!, textSin!, visionCos!, visionSin!, rowMask, groups));
        }
        catch
        {
            textCos?.Dispose();
            textSin?.Dispose();
            visionCos?.Dispose();
            visionSin?.Dispose();
            rowMask?.Dispose();
            DisposeGroups(groups);
            throw;
        }
    }
}

/// <summary>The rotary tables of one MOSS-VL forward. Dispose after the forward that reads them.</summary>
/// <param name="rope">The tables. Text tables, the row mask and the image groups are always owned here.</param>
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
            Rope.VisionCos?.Dispose();
            Rope.VisionSin?.Dispose();
        }
        Rope.RowMask?.Dispose();
        if (Rope.ImageGroups is not null)
            foreach (CrossImageGroup group in Rope.ImageGroups) group.Select.Dispose();
    }
}

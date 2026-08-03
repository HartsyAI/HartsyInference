using System.Globalization;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Builds MiniMax-H3's conditioning prompt presentation for the Qwen3-VL-32B tower and the per-token modality
/// tags the DiT modulates by.
///
/// <para>H3 is <b>not</b> chat-templated: the token ids are raw prompt/label text with no special tokens, with vision
/// blocks spliced in as <c>&lt;|vision_start|&gt;</c> + one <c>&lt;|image_pad|&gt;</c> per merged vision token +
/// <c>&lt;|vision_end|&gt;</c>. Conditions are presented in request order with 1-based per-type ordinals:
/// <c>"&lt;Picture i&gt;: "</c> then a vision block, <c>"&lt;Audio j&gt;: "</c> alone (audio never enters Qwen), and
/// <c>"&lt;Video k&gt;: "</c> then, per 2-frame temporal block sampled at 2 fps, <c>"&lt;T.T seconds&gt;"</c> and that
/// block's vision block. The prompt text is appended last. t2va is the no-condition case; fl2va is one or two
/// <c>&lt;Picture i&gt;</c> conditions.</para>
///
/// <para>Tags: text 1, video 0, audio 2. The whole vision block — <i>including</i> the flanking vision start/end
/// tokens — carries the VIDEO tag, which is what selects the video modulation row for those rows in the DiT; every
/// other token, audio labels included, is text. <see cref="Encoded.TagRuns"/> is the same information as contiguous
/// runs, the shape a packed-layout text-segment split needs.</para></summary>
public static class MiniMaxH3TextEncoding
{
    /// <summary>adaLN modality tag for ordinary prompt/label text.</summary>
    public const int TextTag = 1;

    /// <summary>adaLN modality tag carried by vision-block tokens.</summary>
    public const int VideoTag = 0;

    /// <summary>adaLN modality tag for audio rows; no text token ever carries it.</summary>
    public const int AudioTag = 2;

    /// <summary>What a reference condition is; the presentation label and ordinal counter follow from it.</summary>
    public enum ConditionKind
    {
        /// <summary>A still reference image, presented as <c>&lt;Picture i&gt;</c>.</summary>
        Image,
        /// <summary>A reference video, presented as <c>&lt;Video k&gt;</c> plus one timestamped block per frame pair.</summary>
        Video,
        /// <summary>A reference audio clip, presented as the label <c>&lt;Audio j&gt;</c> only.</summary>
        Audio,
    }

    /// <summary>One spliced vision block: how many merged vision tokens it expands to, and the timestamp label that
    /// precedes it (video blocks only).</summary>
    public readonly record struct VisionBlock(int MergedTokenCount, double? TimestampSeconds = null);

    /// <summary>One reference condition in request order.</summary>
    public sealed record Condition
    {
        public required ConditionKind Kind { get; init; }

        /// <summary>Vision blocks this condition splices; empty for <see cref="ConditionKind.Audio"/>.</summary>
        public IReadOnlyList<VisionBlock> Blocks { get; init; } = [];
    }

    /// <summary>A contiguous run of tokens sharing one modality tag.</summary>
    public readonly record struct TagRun(int Start, int Stop, int Tag);

    /// <summary>The built presentation: token ids, their modality tags, and the vision-block bookkeeping a caller
    /// needs to feed the vision tower in sequence order.</summary>
    public sealed record Encoded
    {
        public required int[] TokenIds { get; init; }

        /// <summary>Per-token modality tag, parallel to <see cref="TokenIds"/>.</summary>
        public required int[] ModalityTags { get; init; }

        /// <summary>The same tags as contiguous runs, in order.</summary>
        public required IReadOnlyList<TagRun> TagRuns { get; init; }

        /// <summary>Merged vision-token count of each spliced block, in the order the blocks appear.</summary>
        public required IReadOnlyList<int> VisionBlockTokenCounts { get; init; }

        /// <summary>The presented text segments in order, before tokenization.</summary>
        public required IReadOnlyList<string> TextSegments { get; init; }

        public int Length => TokenIds.Length;
    }

    /// <summary>Merged vision tokens a grid of <paramref name="gridH"/>x<paramref name="gridW"/> patches collapses to.</summary>
    public static int MergedTokenCount(int gridH, int gridW, int spatialMergeSize = 2)
    {
        if (spatialMergeSize <= 0) throw new ArgumentOutOfRangeException(nameof(spatialMergeSize));
        return gridH / spatialMergeSize * (gridW / spatialMergeSize);
    }

    /// <summary>An image condition expanding to <paramref name="mergedTokenCount"/> vision tokens.</summary>
    public static Condition Image(int mergedTokenCount) => new Condition
    {
        Kind = ConditionKind.Image,
        Blocks = [new VisionBlock(Positive(mergedTokenCount, nameof(mergedTokenCount)))],
    };

    /// <summary>An audio condition; it contributes a label and no vision tokens.</summary>
    public static Condition Audio() => new Condition { Kind = ConditionKind.Audio };

    /// <summary>A video condition over <paramref name="frameCount"/> frames sampled at 2 fps.</summary>
    public static Condition Video(int frameCount, int mergedTokensPerBlock, IReadOnlyList<double>? frameTimestamps = null) =>
        new Condition { Kind = ConditionKind.Video, Blocks = VideoBlocks(frameCount, mergedTokensPerBlock, frameTimestamps) };

    /// <summary>Pairs 2 fps frames into temporal blocks: an odd frame count repeat-pads its last frame, and each block's
    /// label timestamp is the mean of its two frame timestamps (defaulting to <c>i/2</c> seconds).</summary>
    public static IReadOnlyList<VisionBlock> VideoBlocks(int frameCount, int mergedTokensPerBlock,
        IReadOnlyList<double>? frameTimestamps = null)
    {
        Positive(frameCount, nameof(frameCount));
        Positive(mergedTokensPerBlock, nameof(mergedTokensPerBlock));
        if (frameTimestamps is not null && frameTimestamps.Count != frameCount)
        {
            throw new ArgumentException($"Expected {frameCount} frame timestamps, got {frameTimestamps.Count}.",
                nameof(frameTimestamps));
        }

        int padded = frameCount + (frameCount % 2);
        double[] times = new double[padded];
        for (int i = 0; i < padded; i++)
        {
            int src = Math.Min(i, frameCount - 1);
            times[i] = frameTimestamps is null ? src / 2.0 : frameTimestamps[src];
        }

        VisionBlock[] blocks = new VisionBlock[padded / 2];
        for (int i = 0; i < blocks.Length; i++)
        {
            blocks[i] = new VisionBlock(mergedTokensPerBlock, (times[i * 2] + times[i * 2 + 1]) / 2.0);
        }
        return blocks;
    }

    /// <summary>Renders a block timestamp the way the reference's <c>"%.1f"</c> does — round-half-to-even on the exact
    /// binary value, which the default midpoint timestamps (0.25, 1.25, …) all land on.</summary>
    public static string FormatTimestamp(double seconds) =>
        Math.Round(seconds, 1, MidpointRounding.ToEven).ToString("F1", CultureInfo.InvariantCulture);

    /// <summary>Builds the presentation with a Qwen tokenizer, encoding each text segment byte-level-exact and without
    /// special tokens, as the reference's per-segment <c>add_special_tokens=False</c> calls do.</summary>
    public static Encoded Build(Qwen2Tokenizer tokenizer, string prompt, IReadOnlyList<Condition>? conditions = null)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        return Build(tokenizer.EncodeRawByteLevel, prompt, conditions);
    }

    /// <summary>Builds the presentation over any text encoder; <paramref name="encodeText"/> must add no special
    /// tokens and is called once per presented segment.</summary>
    public static Encoded Build(Func<string, IReadOnlyList<int>> encodeText, string prompt,
        IReadOnlyList<Condition>? conditions = null)
    {
        ArgumentNullException.ThrowIfNull(encodeText);
        ArgumentNullException.ThrowIfNull(prompt);

        List<int> ids = new List<int>(256);
        List<int> tags = new List<int>(256);
        List<int> blockCounts = new List<int>();
        List<string> segments = new List<string>();

        void AddText(string text)
        {
            segments.Add(text);
            foreach (int id in encodeText(text))
            {
                ids.Add(id);
                tags.Add(TextTag);
            }
        }

        void AddVision(VisionBlock block)
        {
            Positive(block.MergedTokenCount, nameof(block.MergedTokenCount));
            blockCounts.Add(block.MergedTokenCount);
            ids.Add(Qwen2Tokenizer.VisionStartId);
            for (int i = 0; i < block.MergedTokenCount; i++) ids.Add(Qwen2Tokenizer.ImagePadId);
            ids.Add(Qwen2Tokenizer.VisionEndId);
            for (int i = 0; i < block.MergedTokenCount + 2; i++) tags.Add(VideoTag);
        }

        int images = 0, audios = 0, videos = 0;
        foreach (Condition condition in conditions ?? [])
        {
            switch (condition.Kind)
            {
                case ConditionKind.Image:
                    AddText(Invariant($"<Picture {++images}>: "));
                    foreach (VisionBlock block in condition.Blocks) AddVision(block);
                    break;
                case ConditionKind.Audio:
                    AddText(Invariant($"<Audio {++audios}>: "));
                    break;
                case ConditionKind.Video:
                    AddText(Invariant($"<Video {++videos}>: "));
                    foreach (VisionBlock block in condition.Blocks)
                    {
                        double seconds = block.TimestampSeconds
                            ?? throw new ArgumentException("Video vision blocks need a timestamp; build them with VideoBlocks.",
                                nameof(conditions));
                        AddText(Invariant($"<{FormatTimestamp(seconds)} seconds>"));
                        AddVision(block);
                    }
                    break;
                default:
                    throw new ArgumentException($"Unknown MiniMax-H3 condition kind '{condition.Kind}'.", nameof(conditions));
            }
        }
        AddText(prompt);

        if (ids.Count == 0)
        {
            ids.Add(Qwen2Tokenizer.EndOfTextId);
            tags.Add(TextTag);
        }

        int[] tagArray = tags.ToArray();
        return new Encoded
        {
            TokenIds = ids.ToArray(),
            ModalityTags = tagArray,
            TagRuns = BuildRuns(tagArray),
            VisionBlockTokenCounts = blockCounts,
            TextSegments = segments,
        };
    }

    /// <summary>Collapses a per-token tag array into contiguous runs.</summary>
    public static IReadOnlyList<TagRun> BuildRuns(IReadOnlyList<int> tags)
    {
        ArgumentNullException.ThrowIfNull(tags);
        List<TagRun> runs = new List<TagRun>();
        int start = 0;
        for (int i = 1; i <= tags.Count; i++)
        {
            if (i < tags.Count && tags[i] == tags[start]) continue;
            runs.Add(new TagRun(start, i, tags[start]));
            start = i;
        }
        return runs;
    }

    private static string Invariant(FormattableString text) => text.ToString(CultureInfo.InvariantCulture);

    private static int Positive(int value, string name) =>
        value > 0 ? value : throw new ArgumentOutOfRangeException(name, value, "Must be positive.");
}

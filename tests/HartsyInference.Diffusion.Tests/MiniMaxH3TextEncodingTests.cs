using Xunit;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.Tokenizers;
using C = HartsyInference.Diffusion.Models.TextEncoders.MiniMaxH3TextEncoding.Condition;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tests MiniMax-H3's prompt presentation and modality tagging against the reference ComfyUI text encoder
/// (<c>comfy/text_encoders/minimax.py</c>): per-type 1-based ordinals, 2 fps timestamped video blocks, and the
/// VIDEO tag covering the vision block including its flanking start/end tokens. Uses a synthetic per-character
/// tokenizer, so no tokenizer files or weights are needed.</summary>
public class MiniMaxH3TextEncodingTests
{
    /// <summary>One id per character: makes segment boundaries and lengths directly assertable.</summary>
    private static IReadOnlyList<int> FakeEncode(string text)
    {
        int[] ids = new int[text.Length];
        for (int i = 0; i < text.Length; i++) ids[i] = text[i];
        return ids;
    }

    [Fact]
    public void TextOnlyPromptIsTheRawPromptWithNoTemplate()
    {
        MiniMaxH3TextEncoding.Encoded encoded = MiniMaxH3TextEncoding.Build(FakeEncode, "a cat");
        Assert.Equal(["a cat"], encoded.TextSegments);
        Assert.Equal(5, encoded.Length);
        Assert.All(encoded.ModalityTags, tag => Assert.Equal(MiniMaxH3TextEncoding.TextTag, tag));
        Assert.Empty(encoded.VisionBlockTokenCounts);
    }

    [Fact]
    public void EmptyPresentationFallsBackToThePadToken()
    {
        MiniMaxH3TextEncoding.Encoded encoded = MiniMaxH3TextEncoding.Build(FakeEncode, string.Empty);
        Assert.Equal([Qwen2Tokenizer.EndOfTextId], encoded.TokenIds);
        Assert.Equal([MiniMaxH3TextEncoding.TextTag], encoded.ModalityTags);
    }

    [Fact]
    public void FirstLastFrameSplicesTwoLabelledVisionBlocks()
    {
        MiniMaxH3TextEncoding.Encoded encoded = MiniMaxH3TextEncoding.Build(FakeEncode, "p",
            [MiniMaxH3TextEncoding.Image(4), MiniMaxH3TextEncoding.Image(4)]);

        Assert.Equal(["<Picture 1>: ", "<Picture 2>: ", "p"], encoded.TextSegments);
        Assert.Equal(39, encoded.Length);
        Assert.Equal([4, 4], encoded.VisionBlockTokenCounts);

        Assert.Equal(Qwen2Tokenizer.VisionStartId, encoded.TokenIds[13]);
        Assert.Equal(Qwen2Tokenizer.ImagePadId, encoded.TokenIds[14]);
        Assert.Equal(Qwen2Tokenizer.ImagePadId, encoded.TokenIds[17]);
        Assert.Equal(Qwen2Tokenizer.VisionEndId, encoded.TokenIds[18]);
        Assert.Equal('p', encoded.TokenIds[38]);

        // The VIDEO tag covers the start/end markers too, not just the pads.
        Assert.Equal(
        [
            new MiniMaxH3TextEncoding.TagRun(0, 13, MiniMaxH3TextEncoding.TextTag),
            new MiniMaxH3TextEncoding.TagRun(13, 19, MiniMaxH3TextEncoding.VideoTag),
            new MiniMaxH3TextEncoding.TagRun(19, 32, MiniMaxH3TextEncoding.TextTag),
            new MiniMaxH3TextEncoding.TagRun(32, 38, MiniMaxH3TextEncoding.VideoTag),
            new MiniMaxH3TextEncoding.TagRun(38, 39, MiniMaxH3TextEncoding.TextTag),
        ], encoded.TagRuns);
        Assert.Equal(encoded.TagRuns, MiniMaxH3TextEncoding.BuildRuns(encoded.ModalityTags));
    }

    [Fact]
    public void ReferenceOrdinalsCountPerTypeAndAudioContributesNoVisionTokens()
    {
        C[] conditions =
        [
            MiniMaxH3TextEncoding.Image(1),
            MiniMaxH3TextEncoding.Audio(),
            MiniMaxH3TextEncoding.Video(frameCount: 4, mergedTokensPerBlock: 2),
            MiniMaxH3TextEncoding.Image(1),
            MiniMaxH3TextEncoding.Audio(),
        ];
        MiniMaxH3TextEncoding.Encoded encoded = MiniMaxH3TextEncoding.Build(FakeEncode, "go", conditions);

        Assert.Equal(
        [
            "<Picture 1>: ",
            "<Audio 1>: ",
            "<Video 1>: ",
            "<0.2 seconds>",
            "<1.2 seconds>",
            "<Picture 2>: ",
            "<Audio 2>: ",
            "go",
        ], encoded.TextSegments);
        Assert.Equal([1, 2, 2, 1], encoded.VisionBlockTokenCounts);
        Assert.DoesNotContain(MiniMaxH3TextEncoding.AudioTag, encoded.ModalityTags);
    }

    [Fact]
    public void VideoBlocksPairFramesAndRepeatPadAnOddCount()
    {
        IReadOnlyList<MiniMaxH3TextEncoding.VisionBlock> even = MiniMaxH3TextEncoding.VideoBlocks(8, 3);
        Assert.Equal(4, even.Count);
        Assert.Equal([0.25, 1.25, 2.25, 3.25], even.Select(b => b.TimestampSeconds!.Value));
        Assert.All(even, b => Assert.Equal(3, b.MergedTokenCount));

        IReadOnlyList<MiniMaxH3TextEncoding.VisionBlock> odd = MiniMaxH3TextEncoding.VideoBlocks(3, 3);
        Assert.Equal(2, odd.Count);
        Assert.Equal([0.25, 1.0], odd.Select(b => b.TimestampSeconds!.Value));
    }

    [Fact]
    public void ExplicitFrameTimestampsOverrideThe2FpsDefault()
    {
        IReadOnlyList<MiniMaxH3TextEncoding.VisionBlock> blocks =
            MiniMaxH3TextEncoding.VideoBlocks(4, 1, [10.0, 10.5, 11.0, 11.5]);
        Assert.Equal([10.25, 11.25], blocks.Select(b => b.TimestampSeconds!.Value));
        Assert.Throws<ArgumentException>(() => MiniMaxH3TextEncoding.VideoBlocks(4, 1, [0.0, 0.5]));
    }

    [Theory]
    // Python's "%.1f" rounds half-to-even on the exact binary value; every default block timestamp is a midpoint.
    [InlineData(0.25, "0.2")]
    [InlineData(1.25, "1.2")]
    [InlineData(2.25, "2.2")]
    [InlineData(3.25, "3.2")]
    [InlineData(0.75, "0.8")]
    [InlineData(1.0, "1.0")]
    [InlineData(12.5, "12.5")]
    public void FormatTimestampMatchesThePythonFormat(double seconds, string expected)
    {
        Assert.Equal(expected, MiniMaxH3TextEncoding.FormatTimestamp(seconds));
    }

    [Fact]
    public void MergedTokenCountCollapsesTheMergeBlocks()
    {
        Assert.Equal(64, MiniMaxH3TextEncoding.MergedTokenCount(16, 16));
        Assert.Equal(24, MiniMaxH3TextEncoding.MergedTokenCount(12, 8));
    }

    [Fact]
    public void ZeroSizedConditionsAreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => MiniMaxH3TextEncoding.Image(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => MiniMaxH3TextEncoding.VideoBlocks(0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => MiniMaxH3TextEncoding.VideoBlocks(2, 0));
    }
}

using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Pipelines;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>How long a YuE2 song is allowed to be. Three separate things decide it — the requested duration, the
/// release's own token preset, and how much of the 24,576-token context the prompt and score already spent — and
/// before this they disagreed: the duration knob could not exceed 360 s no matter what it was set to, and a prompt
/// that overran the context threw <i>after</i> the score had already been planned.
/// <para>Pure protocol arithmetic, so no checkpoint, backend or GPU.</para></summary>
public sealed class Yue2BudgetTests
{
    [Fact]
    public void DefaultDuration_IsTheReleaseTokenPreset()
    {
        // 9,000 semantic tokens at 25 a second is exactly what yue2_generation_config.json budgets.
        Assert.Equal(9_000, Yue2Protocol.TokensForSeconds(Yue2Protocol.DefaultDurationSeconds));
        Assert.Equal(Yue2Protocol.DefaultDurationSeconds, new Yue2Request().MaxDurationSeconds);
    }

    [Fact]
    public void Ceiling_IsAboveTheReleasePreset_AndStillFitsTheContext()
    {
        Assert.True(Yue2Protocol.MaxDurationSeconds > Yue2Protocol.DefaultDurationSeconds);
        int ceiling = Yue2Protocol.TokensForSeconds(Yue2Protocol.MaxDurationSeconds);
        Assert.Equal(22_500, ceiling);
        // The ceiling has to leave room for a prefix, or no request could ever reach it.
        Assert.True(ceiling < Yue2Protocol.Context, $"{ceiling} leaves nothing for a prompt in {Yue2Protocol.Context}.");
        // Asking past the ceiling clamps rather than overflowing the context.
        Assert.Equal(ceiling, Yue2Protocol.TokensForSeconds(Yue2Protocol.MaxDurationSeconds * 4));
    }

    [Fact]
    public void SemanticPreset_DoesNotCapTheDurationKnob()
    {
        // The old default was the release's 9,000, which silently won every Math.Min against a longer duration.
        Assert.True(Yue2Sampling.Semantic.MaxTokens >= Yue2Protocol.TokensForSeconds(Yue2Protocol.MaxDurationSeconds),
            "the semantic preset is below the duration ceiling, so the duration knob cannot reach it");
        Assert.Equal(4_096, Yue2Sampling.Abc.MaxTokens);
    }

    [Theory]
    // Comfortable prompt: the request is granted in full.
    [InlineData(9_000, 1_300, 0, 9_000)]
    // A prompt that leaves less than asked for shortens the song instead of failing it.
    [InlineData(22_500, 4_400, 0, 20_176)]
    // Under guidance the negative branch is prefilled separately, so the LONGER branch binds.
    [InlineData(22_500, 1_300, 5_000, 19_576)]
    [InlineData(22_500, 5_000, 1_300, 19_576)]
    // Nothing left at all.
    [InlineData(9_000, Yue2Protocol.Context, 0, 0)]
    [InlineData(9_000, Yue2Protocol.Context + 500, 0, 0)]
    public void BudgetForPrefix_FitsTheLongerBranch(int requested, int prefix, int negative, int expected)
        => Assert.Equal(expected, Yue2Protocol.BudgetForPrefix(requested, prefix, negative));

    [Fact]
    public void BudgetForPrefix_NeverExceedsTheRequest()
    {
        // A tiny prompt does not entitle a short request to more tokens than it asked for.
        Assert.Equal(250, Yue2Protocol.BudgetForPrefix(250, 40));
    }

    /// <summary>Past roughly seven and a half minutes the acoustic stage runs more than one chunk. Nothing exercised
    /// that before the ceiling moved — at the release's 360 s a typical prefix always yielded exactly one.</summary>
    [Fact]
    public void ChunkRanges_SplitLongSongs_AndCoverEveryFrameOnce()
    {
        const int prefixTokens = 1_300;
        Assert.Single(Yue2Protocol.ChunkRanges(Yue2Protocol.TokensForSeconds(Yue2Protocol.DefaultDurationSeconds), prefixTokens));

        int frames = Yue2Protocol.TokensForSeconds(Yue2Protocol.MaxDurationSeconds);
        (int Start, int End)[] ranges = Yue2Protocol.ChunkRanges(frames, prefixTokens);
        Assert.True(ranges.Length > 1, $"{frames} frames behind a {prefixTokens}-token prefix still fits one chunk.");

        // Contiguous, gapless, and every chunk small enough that its own AR prefill fits the context: each frame
        // costs two positions (the codec token and the latent), plus the prefix and three boundary tokens.
        Assert.Equal(0, ranges[0].Start);
        Assert.Equal(frames, ranges[^1].End);
        for (int i = 0; i < ranges.Length; i++)
        {
            Assert.True(ranges[i].End > ranges[i].Start, $"chunk {i} is empty");
            if (i > 0) Assert.Equal(ranges[i - 1].End, ranges[i].Start);
            Assert.True(prefixTokens + (ranges[i].End - ranges[i].Start) * 2 + 3 <= Yue2Protocol.Context,
                $"chunk {i} overruns the context");
        }
    }
}

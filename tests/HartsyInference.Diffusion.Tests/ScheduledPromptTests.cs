using System.Linq;
using Xunit;
using HartsyInference.Diffusion.Prompting;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The token-side half of per-step conditioning: which distinct prompt texts a scheduled prompt resolves
/// to, and how steps map onto them. The pipeline encodes one tensor per variant and selects by that map, so a wrong
/// count here is a wasted encode and a wrong map is the right words at the wrong steps.</summary>
public sealed class ScheduledPromptTests
{
    /// <summary>Tokenizer stand-in: one token per character, recording the exact texts it was handed so a test can
    /// assert how many distinct encodes a schedule costs.</summary>
    private sealed class RecordingTokenizer
    {
        public List<string> Calls { get; } = [];

        public WeightedTokenSequence Tokenize(string text)
        {
            Calls.Add(text);
            int[] ids = [.. text.Select(c => (int)c)];
            float[] weights = [.. ids.Select(_ => 1f)];
            return new WeightedTokenSequence(ids, weights) { UniformWeight = 1f };
        }
    }

    /// <summary>An ordinary prompt must not be routed through a one-variant schedule — the caller's existing
    /// single-encode path stays byte-identical, and nothing is tokenized twice to discover that.</summary>
    [Fact]
    public void APromptWithNoSchedulingTagBuildsNoSchedule()
    {
        RecordingTokenizer tokenizer = new RecordingTokenizer();
        Assert.Null(ScheduledPrompt.TryBuild("a red fox in snow", 8, tokenizer.Tokenize));
        Assert.Empty(tokenizer.Calls);
    }

    /// <summary>Alternation is per step, but only two texts ever occur — so it costs two encodes across the whole
    /// loop, not one per step. That dedup is the reason this indexes variants rather than steps.</summary>
    [Fact]
    public void AlternateResolvesToTwoVariantsWhateverTheStepCount()
    {
        RecordingTokenizer tokenizer = new RecordingTokenizer();
        ScheduledPrompt? schedule = ScheduledPrompt.TryBuild("<alternate:fox, whale>", 30, tokenizer.Tokenize);

        Assert.NotNull(schedule);
        Assert.True(schedule!.IsScheduled);
        Assert.Equal(2, schedule.Variants.Count);
        Assert.Equal(2, tokenizer.Calls.Count);
        Assert.Equal(30, schedule.StepToVariant.Length);
        Assert.Equal(schedule.IndexForStep(0), schedule.IndexForStep(2));
        Assert.NotEqual(schedule.IndexForStep(0), schedule.IndexForStep(1));
    }

    /// <summary>A fractional switch lands where SwarmUI puts it: <c>when &lt; 1</c> is a fraction of the step
    /// count, so half of eight steps switches after the fourth.</summary>
    [Fact]
    public void FromToSwitchesAtTheFractionOfTheStepCount()
    {
        RecordingTokenizer tokenizer = new RecordingTokenizer();
        ScheduledPrompt? schedule = ScheduledPrompt.TryBuild("<fromto[0.5]:fox, whale>", 8, tokenizer.Tokenize);

        Assert.NotNull(schedule);
        Assert.Equal(2, schedule!.Variants.Count);
        int[] map = schedule.StepToVariant;
        Assert.Equal([0, 0, 0, 0, 1, 1, 1, 1], map);
    }

    /// <summary>The two ends of the gate the plan specifies, and the reason they are byte-identity checks rather
    /// than eyeball ones: both collapse to a SINGLE variant, so the pipeline never leaves its ordinary path and the
    /// output must match the corresponding plain prompt exactly.</summary>
    [Theory]
    [InlineData("<fromto[0]:fox, whale>", "whale")]
    [InlineData("<fromto[99]:fox, whale>", "fox")]
    public void AFromToOutsideTheLoopCollapsesToOneVariant(string prompt, string expected)
    {
        RecordingTokenizer tokenizer = new RecordingTokenizer();
        ScheduledPrompt? schedule = ScheduledPrompt.TryBuild(prompt, 8, tokenizer.Tokenize);

        Assert.NotNull(schedule);
        Assert.False(schedule!.IsScheduled);
        Assert.Single(schedule.Variants);
        Assert.Equal([expected], tokenizer.Calls);
    }

    /// <summary>Branches that resolve to the same text are one variant, not two identical encodes.</summary>
    [Fact]
    public void BranchesThatResolveAlikeDoNotBecomeTwoVariants()
    {
        RecordingTokenizer tokenizer = new RecordingTokenizer();
        ScheduledPrompt? schedule = ScheduledPrompt.TryBuild("<fromto[0.5]:fox, fox>", 8, tokenizer.Tokenize);

        Assert.NotNull(schedule);
        Assert.False(schedule!.IsScheduled);
        Assert.Single(schedule.Variants);
    }

    /// <summary>Each branch is tokenized on its own, so the emphasis inside one branch is resolved against that
    /// branch alone. The recorded calls show it: the weighted branch arrives as its own text, and the unweighted
    /// one is not dragged through the weighted path just because its neighbour carried a weight.</summary>
    [Fact]
    public void EachBranchIsTokenizedWithItsOwnEmphasis()
    {
        RecordingTokenizer tokenizer = new RecordingTokenizer();
        ScheduledPrompt? schedule = ScheduledPrompt.TryBuild("<fromto[0.5]:(fox:1.2), whale>", 8, tokenizer.Tokenize);

        Assert.NotNull(schedule);
        Assert.Equal(2, schedule!.Variants.Count);
        Assert.Equal(["(fox:1.2)", "whale"], tokenizer.Calls);
    }

    /// <summary>A sampler that runs past the step count the schedule was planned for reuses the last entry rather
    /// than throwing — the schedule is built from the requested steps, and a second-order sampler can ask for more.
    /// </summary>
    [Fact]
    public void AStepPastTheEndReusesTheLastVariant()
    {
        RecordingTokenizer tokenizer = new RecordingTokenizer();
        ScheduledPrompt? schedule = ScheduledPrompt.TryBuild("<fromto[0.5]:fox, whale>", 8, tokenizer.Tokenize);

        Assert.NotNull(schedule);
        Assert.Equal(schedule!.IndexForStep(7), schedule.IndexForStep(99));
        Assert.Equal(schedule.IndexForStep(0), schedule.IndexForStep(-5));
    }
}

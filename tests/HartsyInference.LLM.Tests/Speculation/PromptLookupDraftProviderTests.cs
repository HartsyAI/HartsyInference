using HartsyInference.LLM.Generation.Speculation;
using Xunit;

namespace HartsyInference.LLM.Tests.Speculation;

/// <summary>Checks <see cref="PromptLookupDraftProvider"/> against the inline drafter that
/// <c>TextGenerationPipeline.GenerateSpeculative</c> used before extraction.
/// The reference below is a verbatim copy of that old <c>FindDraftContinuation</c>, with its <c>SpecMaxLookback</c>
/// constant passed in as a parameter. The pipeline itself cannot serve as the oracle: speculative output equals plain
/// greedy output whatever the drafts are, so only the drafts themselves reveal a difference.</summary>
public sealed class PromptLookupDraftProviderTests
{
    [Fact]
    public void HandBuiltContext_ProposesTheContinuationOfTheOldestSuffixMatch()
    {
        // The suffix [5,6,7] occurs at index 0 and at the end. The draft is what followed index 0, capped by maxDraft.
        PromptLookupDraftProvider provider = new();
        int[] prompt = [5, 6, 7, 8, 5, 6, 7];
        Assert.Equal(new[] { 8, 5, 6, 7 }, provider.Propose(prompt, new List<int>(), 8));
        Assert.Equal(new[] { 8, 5 }, provider.Propose(prompt, new List<int>(), 2));
    }

    [Fact]
    public void MatchInGeneratedOutput_IsFoundInTheSameWindow()
    {
        // The needle [1,2,3] lies across the prompt and the generated tokens; its earlier copy is in the prompt.
        PromptLookupDraftProvider provider = new();
        int[] prompt = [1, 2, 3, 9];
        List<int> generated = [1, 2, 3];
        Assert.Equal(new[] { 9, 1, 2, 3 }, provider.Propose(prompt, generated, 8));
    }

    [Fact]
    public void NoMatch_ProposesNothing()
    {
        PromptLookupDraftProvider provider = new();
        Assert.Empty(provider.Propose([1, 2, 3, 4, 5, 6], new List<int>(), 8));
    }

    [Fact]
    public void ContextShorterThanTheNgram_ProposesNothing()
    {
        PromptLookupDraftProvider provider = new();
        Assert.Empty(provider.Propose([1, 2], new List<int>(), 8));
    }

    [Fact]
    public void NonPositiveDraftLength_ProposesNothing()
    {
        PromptLookupDraftProvider provider = new();
        Assert.Empty(provider.Propose([5, 6, 7, 8, 5, 6, 7], new List<int>(), 0));
    }

    [Fact]
    public void Name_IsStable()
    {
        Assert.Equal("prompt-lookup", new PromptLookupDraftProvider().Name);
    }

    [Theory]
    [InlineData(1, 3)]
    [InlineData(2, 5)]
    [InlineData(3, 7)]
    [InlineData(4, 16)]
    public void RandomContexts_ProposeExactlyWhatTheOldInlineDrafterProposed(int ngramSize, int maxLookback)
    {
        // Small vocabularies make repeats (and so matches) common. Lengths cross the lookback so the window cap is exercised.
        uint state = 0x9E3779B9u ^ (uint)(ngramSize * 7919 + maxLookback);
        PromptLookupDraftProvider provider = new(ngramSize, maxLookback);
        for (int trial = 0; trial < 4000; trial++)
        {
            int vocab = 1 + (int)(Next(ref state) % 6);
            int promptLen = (int)(Next(ref state) % 24);
            int generatedLen = (int)(Next(ref state) % 60);
            int[] prompt = new int[promptLen];
            for (int i = 0; i < promptLen; i++) prompt[i] = (int)(Next(ref state) % (uint)vocab);
            List<int> generated = new(generatedLen);
            for (int i = 0; i < generatedLen; i++) generated.Add((int)(Next(ref state) % (uint)vocab));
            int maxDraft = (int)(Next(ref state) % 12);

            int[] expected = ReferenceFindDraftContinuation(prompt, generated, ngramSize, maxDraft, maxLookback);
            int[] actual = provider.Propose(prompt, generated, maxDraft);
            Assert.True(expected.SequenceEqual(actual),
                $"trial {trial}: expected [{string.Join(",", expected)}] but got [{string.Join(",", actual)}]");
        }
    }

    [Fact]
    public void LongContextsWithDefaults_ProposeExactlyWhatTheOldInlineDrafterProposed()
    {
        // Contexts past the default 4096-token lookback, so the window cap decides which matches are visible.
        uint state = 0xC0FFEEu;
        PromptLookupDraftProvider provider = new();
        for (int trial = 0; trial < 40; trial++)
        {
            int vocab = 2 + (int)(Next(ref state) % 5);
            int promptLen = (int)(Next(ref state) % 300);
            int generatedLen = 4000 + (int)(Next(ref state) % 1500);
            int[] prompt = new int[promptLen];
            for (int i = 0; i < promptLen; i++) prompt[i] = (int)(Next(ref state) % (uint)vocab);
            List<int> generated = new(generatedLen);
            for (int i = 0; i < generatedLen; i++) generated.Add((int)(Next(ref state) % (uint)vocab));

            int[] expected = ReferenceFindDraftContinuation(prompt, generated, 3, 8, 4096);
            int[] actual = provider.Propose(prompt, generated, 8);
            Assert.True(expected.SequenceEqual(actual), $"trial {trial} diverged");
        }
    }

    private static uint Next(ref uint s)
    {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }

    /// <summary>Verbatim copy of the pre-extraction <c>TextGenerationPipeline.FindDraftContinuation</c>;
    /// <c>SpecMaxLookback</c> is the <paramref name="maxLookback"/> argument.</summary>
    private static int[] ReferenceFindDraftContinuation(int[] promptIds, List<int> generated, int ngramSize, int maxDraftLen, int maxLookback)
    {
        int totalLen = promptIds.Length + generated.Count;
        if (maxDraftLen <= 0 || totalLen < ngramSize) return [];

        int searchFloor = Math.Max(0, totalLen - maxLookback);
        int windowLen = totalLen - searchFloor;
        int[] context = new int[windowLen];
        for (int i = 0; i < windowLen; i++)
        {
            int idx = searchFloor + i;
            context[i] = idx < promptIds.Length ? promptIds[idx] : generated[idx - promptIds.Length];
        }

        int needleStart = windowLen - ngramSize;
        for (int start = 0; start < needleStart; start++)
        {
            bool match = true;
            for (int k = 0; k < ngramSize; k++)
            {
                if (context[start + k] != context[needleStart + k]) { match = false; break; }
            }
            if (!match) continue;

            int matchEnd = start + ngramSize;
            int draftLen = Math.Min(maxDraftLen, windowLen - matchEnd);
            if (draftLen <= 0) continue;
            int[] draft = new int[draftLen];
            Array.Copy(context, matchEnd, draft, 0, draftLen);
            return draft;
        }
        return [];
    }
}

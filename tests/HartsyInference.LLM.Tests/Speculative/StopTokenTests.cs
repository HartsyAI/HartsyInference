using HartsyInference.LLM.Generation.Speculative;
using HartsyInference.LLM.Sampling;
using Xunit;

namespace HartsyInference.LLM.Tests.Speculative;

/// <summary>Stop tokens end a speculative run at the first stop token emitted, even inside an accepted draft. A scripted target makes every drafted token its greedy
/// choice, so every draft is accepted and the stop lands where each test puts it.</summary>
public sealed class StopTokenTests
{
    private const int Vocab = 8;

    /// <summary>Scores each drafted position as its own token and the position after the draft as <paramref name="bonus"/>, as one-hot rows. At
    /// <paramref name="mismatchAt"/> the target names <paramref name="other"/> instead, so the draft is cut there and that token is the correction.</summary>
    private sealed class ScriptedScorer(int bonus, int mismatchAt = -1, int other = 0) : ISpeculativeScorer
    {
        public void Score(ReadOnlySpan<int> context, ReadOnlySpan<int> draft, float[][] rows)
        {
            for (int j = 0; j < rows.Length; j++)
            {
                Array.Fill(rows[j], 0f);
                int named = j < draft.Length ? (j == mismatchAt ? other : draft[j]) : bonus;
                rows[j][named] = 1f;
            }
        }
    }

    /// <summary>Always proposes the same block, cut to the limit.</summary>
    private sealed class FixedProposer(int[] block) : IDraftProposer
    {
        public DraftBlock Propose(ReadOnlySpan<int> context, int maxTokens) => new(block[..Math.Min(maxTokens, block.Length)], null);
    }

    [Fact]
    public void A_Stop_Token_Inside_An_Accepted_Draft_Ends_The_Run_There()
    {
        List<int> tokens = [1, 2];
        int produced = SpeculativeLoop.Generate(new ScriptedScorer(bonus: 7), new FixedProposer([3, 5, 6, 2]),
            SamplerChain.FromOptions(new SamplingOptions { Greedy = true }), SpeculativeTestSupport.Uniform(1), tokens, 10, 4, Vocab, new HashSet<int> { 5 });

        // 3 and the stop token 5 are emitted; the rest of the accepted draft, and the bonus token, are not
        Assert.Equal(2, produced);
        Assert.Equal(new[] { 1, 2, 3, 5 }, tokens);
    }

    [Fact]
    public void A_Stop_Token_As_The_Bonus_Token_Ends_The_Run_There()
    {
        List<int> tokens = [1, 2];
        int produced = SpeculativeLoop.Generate(new ScriptedScorer(bonus: 0), new FixedProposer([3, 4]),
            SamplerChain.FromOptions(new SamplingOptions { Greedy = true }), SpeculativeTestSupport.Uniform(1), tokens, 10, 4, Vocab, new HashSet<int> { 0 });

        Assert.Equal(3, produced);
        Assert.Equal(new[] { 1, 2, 3, 4, 0 }, tokens);
    }

    [Fact]
    public void A_Stop_Token_As_The_Correction_Ends_The_Run_There()
    {
        // the draft [3, 4, 6] is cut at position 1, where the target names 1: the correction is the stop token
        List<int> tokens = [1, 2];
        int produced = SpeculativeLoop.Generate(new ScriptedScorer(bonus: 7, mismatchAt: 1, other: 1), new FixedProposer([3, 4, 6]),
            SamplerChain.FromOptions(new SamplingOptions { Greedy = true }), SpeculativeTestSupport.Uniform(1), tokens, 10, 4, Vocab, new HashSet<int> { 1 });

        Assert.Equal(2, produced);
        Assert.Equal(new[] { 1, 2, 3, 1 }, tokens);
    }

    [Fact]
    public void An_Empty_Stop_Set_Changes_Nothing()
    {
        List<int> tokens = [1, 2];
        int produced = SpeculativeLoop.Generate(new ScriptedScorer(bonus: 7), new FixedProposer([3, 5, 6, 2]),
            SamplerChain.FromOptions(new SamplingOptions { Greedy = true }), SpeculativeTestSupport.Uniform(1), tokens, 6, 4, Vocab, new HashSet<int>());

        Assert.Equal(6, produced);
        Assert.Equal(new[] { 1, 2, 3, 5, 6, 2, 7, 7 }, tokens);
    }

    [Fact]
    public void Without_Stop_Tokens_The_Run_Is_Unchanged()
    {
        List<int> tokens = [1, 2];
        int produced = SpeculativeLoop.Generate(new ScriptedScorer(bonus: 7), new FixedProposer([3, 5, 6, 2]),
            SamplerChain.FromOptions(new SamplingOptions { Greedy = true }), SpeculativeTestSupport.Uniform(1), tokens, 6, 4, Vocab);

        // the first round accepts the four drafted tokens and adds the bonus 7; the second has no room for a draft and adds its own 7
        Assert.Equal(6, produced);
        Assert.Equal(new[] { 1, 2, 3, 5, 6, 2, 7, 7 }, tokens);
    }
}

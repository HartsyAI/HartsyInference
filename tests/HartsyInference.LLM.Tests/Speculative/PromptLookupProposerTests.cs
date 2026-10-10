using HartsyInference.LLM.Generation.Speculative;
using Xunit;

namespace HartsyInference.LLM.Tests.Speculative;

public sealed class PromptLookupProposerTests
{
    [Fact]
    public void Proposes_The_Continuation_Of_The_Earlier_Match()
    {
        DraftBlock block = new PromptLookupProposer().Propose([5, 6, 7, 8, 5, 6], 4);
        Assert.Equal([7, 8, 5, 6], block.Tokens);
        Assert.Null(block.Probs);
        Assert.Equal([7, 8], new PromptLookupProposer().Propose([5, 6, 7, 8, 5, 6], 2).Tokens);
    }

    [Fact]
    public void Returns_Nothing_Without_A_Match()
    {
        Assert.Empty(new PromptLookupProposer().Propose([1, 2, 3], 4).Tokens);
        Assert.Empty(new PromptLookupProposer().Propose([], 4).Tokens);
        Assert.Empty(new PromptLookupProposer().Propose([5, 6, 5, 6], 0).Tokens);
    }
}

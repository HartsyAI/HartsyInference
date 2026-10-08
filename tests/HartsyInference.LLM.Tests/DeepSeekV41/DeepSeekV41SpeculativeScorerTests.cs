using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Generation.Speculative;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Tests.Speculative;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The V4.1 speculative scorer against plain decoding on the synthetic model: the rows it reads are the per-token logits, and greedy speculation reproduces
/// plain greedy decoding token for token, through the rollbacks that rejected drafts force.</summary>
public sealed class DeepSeekV41SpeculativeScorerTests
{
    private const int Capacity = 256;

    // the fixture's own context: its repeated tokens give the lookup proposer drafts to verify
    private static readonly int[] Prompt = [5, 13, 1, 1, 1, 15, 13, 23, 27, 29, 2];

    /// <summary>Counts the rounds that drafted something, so the identity test cannot pass by never speculating.</summary>
    private sealed class CountingProposer(IDraftProposer inner) : IDraftProposer
    {
        public int Drafted { get; private set; }

        public DraftBlock Propose(ReadOnlySpan<int> context, int maxTokens)
        {
            DraftBlock block = inner.Propose(context, maxTokens);
            if (block.Tokens.Length > 0) Drafted++;
            return block;
        }
    }

    [Fact]
    public void Scored_Rows_Match_The_Plain_Per_Token_Logits()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu);
        int[] draft = [13, 1, 15];

        float[][] expected = new float[draft.Length + 1][];
        DeepSeekV41GenerationState plain = new(model, Capacity);
        float[] hidden = new float[Prompt.Length * model.Dim];
        plain.Append(Prompt, hidden);
        expected[0] = model.Logits(hidden.AsSpan((Prompt.Length - 1) * model.Dim, model.Dim));
        for (int j = 0; j < draft.Length; j++)
        {
            float[] step = new float[model.Dim];
            plain.Append([draft[j]], step);
            expected[j + 1] = model.Logits(step);
        }

        float[][] rows = Enumerable.Range(0, draft.Length + 1).Select(_ => new float[model.VocabSize]).ToArray();
        new DeepSeekV41SpeculativeScorer(model, new DeepSeekV41GenerationState(model, Capacity)).Score(Prompt, draft, rows);
        for (int j = 0; j < rows.Length; j++) Assert.Equal(expected[j], rows[j]);
    }

    [Fact]
    public void Greedy_Speculation_On_The_Host_Model_Reproduces_Plain_Greedy_Decoding_Token_For_Token()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu);
        SamplingOptions greedy = new() { Greedy = true };
        const int count = 24;

        List<int> plain = [.. Prompt];
        DeepSeekV41GenerationState plainState = new(model, Capacity);
        float[] hidden = new float[plain.Count * model.Dim];
        plainState.Append(Prompt, hidden);
        float[] logits = model.Logits(hidden.AsSpan((plain.Count - 1) * model.Dim, model.Dim));
        SamplerChain plainChain = SamplerChain.FromOptions(greedy);
        for (int i = 0; i < count; i++)
        {
            int next = plainChain.Next(logits, plain);
            plain.Add(next);
            float[] step = new float[model.Dim];
            plainState.Append([next], step);
            logits = model.Logits(step);
        }

        List<int> speculative = [.. Prompt];
        CountingProposer proposer = new(new PromptLookupProposer());
        int produced = SpeculativeLoop.Generate(new DeepSeekV41SpeculativeScorer(model, new DeepSeekV41GenerationState(model, Capacity)), proposer,
            SamplerChain.FromOptions(greedy), SpeculativeTestSupport.Uniform(1), speculative, count, 4, model.VocabSize);

        Assert.Equal(count, produced);
        Assert.True(proposer.Drafted > 0, "the lookup proposer never drafted, so the identity below was not exercised");
        Assert.Equal(plain, speculative);
    }
}

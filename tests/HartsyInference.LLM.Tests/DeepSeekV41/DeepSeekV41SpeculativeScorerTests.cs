using System.Text.Json;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Generation.Speculative;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Tests.Speculative;
using HartsyInference.Tests.Common;
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

    [Fact]
    public void Scored_Rows_Match_The_Plain_Per_Token_Logits_Across_Prompt_And_Draft_Lengths()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu);
        int[] ids = FixtureIds();
        foreach (int length in new[] { 1, 2, 6, 11 })
            for (int draftLength = 1; draftLength <= 4; draftLength++)
            {
                int[] context = ids[..length], draft = ids[length..(length + draftLength)];
                float[][] expected = new float[draft.Length + 1][];
                DeepSeekV41GenerationState plain = new(model, Capacity);
                float[] hidden = new float[context.Length * model.Dim];
                plain.Append(context, hidden);
                expected[0] = model.Logits(hidden.AsSpan((context.Length - 1) * model.Dim, model.Dim));
                for (int j = 0; j < draft.Length; j++)
                {
                    float[] step = new float[model.Dim];
                    plain.Append([draft[j]], step);
                    expected[j + 1] = model.Logits(step);
                }

                float[][] rows = Enumerable.Range(0, draft.Length + 1).Select(_ => new float[model.VocabSize]).ToArray();
                new DeepSeekV41SpeculativeScorer(model, new DeepSeekV41GenerationState(model, Capacity)).Score(context, draft, rows);
                for (int j = 0; j < rows.Length; j++) Assert.Equal(expected[j], rows[j]);
            }
    }

    [Fact]
    public void Greedy_Speculation_Matches_Plain_Decoding_Across_Prompt_Lengths_And_Draft_Sizes()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu);
        SamplingOptions greedy = new() { Greedy = true };
        int[] ids = FixtureIds();
        const int count = 16;
        foreach (int length in new[] { 6, 9, 11 })
        {
            List<int> plain = PlainGreedy(model, ids[..length], count);
            foreach (int maxDraft in new[] { 1, 3, 6 })
            {
                List<int> speculative = [.. ids[..length]];
                CountingProposer proposer = new(new PromptLookupProposer());
                int produced = SpeculativeLoop.Generate(new DeepSeekV41SpeculativeScorer(model, new DeepSeekV41GenerationState(model, Capacity)), proposer,
                    SamplerChain.FromOptions(greedy), SpeculativeTestSupport.Uniform(1), speculative, count, maxDraft, model.VocabSize);
                Assert.Equal(count, produced);
                Assert.Equal(plain, speculative);
                if (length == 11) Assert.True(proposer.Drafted > 0, $"no drafts at prompt {length}, draft {maxDraft}");
            }
        }
    }

    /// <summary>Plain greedy decoding: the prompt as one prefill, then one token at a time.</summary>
    private static List<int> PlainGreedy(DeepSeekV41HostModel model, int[] prompt, int count)
    {
        List<int> tokens = [.. prompt];
        DeepSeekV41GenerationState state = new(model, Capacity);
        float[] hidden = new float[prompt.Length * model.Dim];
        state.Append(prompt, hidden);
        float[] logits = model.Logits(hidden.AsSpan((prompt.Length - 1) * model.Dim, model.Dim));
        SamplerChain chain = SamplerChain.FromOptions(new SamplingOptions { Greedy = true });
        for (int i = 0; i < count; i++)
        {
            int next = chain.Next(logits, tokens);
            tokens.Add(next);
            float[] step = new float[model.Dim];
            state.Append([next], step);
            logits = model.Logits(step);
        }
        return tokens;
    }

    /// <summary>The fixture's 17 token ids: its 11-token prefill followed by six decode steps.</summary>
    private static int[] FixtureIds()
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", "model_forward.json")));
        return doc.RootElement.GetProperty("steps").EnumerateArray()
            .SelectMany(step => step.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32())).ToArray();
    }
}

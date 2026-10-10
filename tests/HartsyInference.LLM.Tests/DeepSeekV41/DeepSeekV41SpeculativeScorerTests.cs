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
    public void Scored_Rows_Match_The_Plain_Per_Token_Logits_Across_Prompt_And_Draft_Lengths()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu);
        int[] ids = FixtureIds();
        foreach (int length in Enumerable.Range(1, 11))
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

    [Fact]
    public void Rejected_Calls_Leave_The_Sequence_Unchanged()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu);
        DeepSeekV41GenerationState state = new(model, Capacity);
        DeepSeekV41SpeculativeScorer scorer = new(model, state);
        scorer.Score(Prompt, [13], Rows(model, 2));
        int[] before = state.Tokens.ToArray();

        // the draft does not fit the sequence: the check runs before any rollback or append
        Assert.Throws<InvalidOperationException>(() => scorer.Score(Prompt, new int[Capacity], Rows(model, Capacity + 1)));
        // one row is one logit short
        Assert.Throws<ArgumentException>(() => scorer.Score(Prompt, [13], new[] { new float[model.VocabSize], new float[model.VocabSize - 1] }));
        Assert.Equal(before, state.Tokens);
        Assert.Equal(before.Length, state.Length);
    }

    [Fact]
    public void A_Context_That_Diverges_Inside_The_Prompt_Is_Prefilled_Afresh()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu);
        DeepSeekV41GenerationState state = new(model, Capacity);
        DeepSeekV41SpeculativeScorer scorer = new(model, state);
        scorer.Score(Prompt, [13, 1], Rows(model, 3));

        // shares two tokens with the prompt, then diverges inside the prompt's chunk
        int[] other = [5, 13, 9, 9, 9, 1];
        int[] draft = [27, 29];
        float[][] rows = Rows(model, draft.Length + 1);
        scorer.Score(other, draft, rows);

        // plain decoding of that context: a fresh prefill of it, then the draft one token at a time
        DeepSeekV41GenerationState plain = new(model, Capacity);
        float[] hidden = new float[other.Length * model.Dim];
        plain.Append(other, hidden);
        Assert.Equal(model.Logits(hidden.AsSpan((other.Length - 1) * model.Dim, model.Dim)), rows[0]);
        for (int j = 0; j < draft.Length; j++)
        {
            float[] step = new float[model.Dim];
            plain.Append([draft[j]], step);
            Assert.Equal(model.Logits(step), rows[j + 1]);
        }
    }

    [Fact]
    public void Recorded_Main_Rows_Match_A_Direct_Tap_Before_And_After_A_Rollback()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu, new[] { 2, 5 });
        int width = model.MainHiddenWidth, dim = model.Dim;
        int[] ids = FixtureIds();
        DeepSeekV41GenerationState state = new(model, Capacity, recordMainRows: true);
        state.Append(ids[..8], new float[8 * dim]);
        state.Append(ids[8..11], new float[3 * dim]);
        state.Truncate(9);
        state.Append(ids[9..12], new float[3 * dim]);

        // the same history through the host directly: the prompt as one chunk, then one token at a time
        DeepSeekV41SequenceState raw = model.CreateState(Capacity);
        float[] direct = new float[12 * width];
        model.Forward(ids[..8], raw, new float[8 * dim], direct.AsSpan(0, 8 * width));
        for (int p = 8; p < 12; p++) model.Forward(ids[p..(p + 1)], raw, new float[dim], direct.AsSpan(p * width, width));

        Assert.Equal(width, state.MainRow(0).Length);
        for (int p = 0; p < 12; p++) Assert.Equal(direct.AsSpan(p * width, width).ToArray(), state.MainRow(p).ToArray());
    }

    private static float[][] Rows(DeepSeekV41HostModel model, int count) => Enumerable.Range(0, count).Select(_ => new float[model.VocabSize]).ToArray();

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

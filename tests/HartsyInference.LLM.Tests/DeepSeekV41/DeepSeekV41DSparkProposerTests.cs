using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Generation.Speculative;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Tests.Speculative;
using System.Text.Json;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The DSpark proposer over the synthetic model. Its drafts reproduce upstream's at every decode position, with the state built the way the engine builds it,
/// and speculation with it reproduces plain greedy decoding token for token, through the rollbacks of rejected drafts and past the window.</summary>
public sealed class DeepSeekV41DSparkProposerTests
{
    private const int BlockDraft = 5;

    /// <summary>Counts the rounds that drafted something, so the identity test cannot pass without speculating.</summary>
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
    public void Proposer_Drafts_Match_Upstream_At_Every_Decode_Position()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu, DeepSeekV41DSparkFixture.TargetLayers);
        DeepSeekV41DSpark dspark = DeepSeekV41DSparkFixture.BuildDSpark(cpu);
        int[] ids = DeepSeekV41DSparkFixture.Ids;
        DeepSeekV41GenerationState state = new(model, DeepSeekV41DSparkFixture.MaxTokens, recordMainRows: true);
        DeepSeekV41DSparkProposer proposer = new(dspark, state);

        // the engine's own sequence: the prompt as one chunk, then one token at a time as the drafts are checked
        state.SyncTo(ids.AsSpan(0, 11));
        foreach ((int pos, JsonElement expected) in DeepSeekV41DSparkFixture.Drafts())
        {
            DraftBlock block = proposer.Propose(ids.AsSpan(0, pos + 1), BlockDraft);
            int[] upstream = DeepSeekV41DSparkFixture.Ints(expected.GetProperty("ids"));
            Assert.Equal(upstream[1..], block.Tokens);
        }
    }

    [Theory]
    [InlineData(6)]
    [InlineData(9)]
    [InlineData(11)]
    public void Speculation_With_The_DSpark_Proposer_Reproduces_Plain_Greedy_Decoding(int promptLength)
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu, DeepSeekV41DSparkFixture.TargetLayers);
        DeepSeekV41DSpark dspark = DeepSeekV41DSparkFixture.BuildDSpark(cpu);
        int[] prompt = DeepSeekV41DSparkFixture.Ids[..promptLength];
        const int count = 12;
        List<int> plain = DeepSeekV41DSparkFixture.PlainGreedy(model, prompt, count);

        DeepSeekV41GenerationState state = new(model, DeepSeekV41DSparkFixture.MaxTokens, recordMainRows: true);
        CountingProposer proposer = new(new DeepSeekV41DSparkProposer(dspark, state));
        List<int> speculative = [.. prompt];
        int produced = SpeculativeLoop.Generate(new DeepSeekV41SpeculativeScorer(model, state), proposer, SamplerChain.FromOptions(new SamplingOptions { Greedy = true }),
            SpeculativeTestSupport.Uniform(1), speculative, count, BlockDraft, model.VocabSize);

        Assert.Equal(count, produced);
        Assert.True(proposer.Drafted > 0, "the DSpark proposer never drafted");
        Assert.Equal(plain, speculative);
    }

    [Fact]
    public void A_Single_Token_Context_Drafts_From_An_Empty_Window()
    {
        // the first round after a one-token prompt: nothing is committed before the anchor, so the head's window is seeded with no rows
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu, DeepSeekV41DSparkFixture.TargetLayers);
        DeepSeekV41DSpark dspark = DeepSeekV41DSparkFixture.BuildDSpark(cpu);
        DeepSeekV41GenerationState state = new(model, DeepSeekV41DSparkFixture.MaxTokens, recordMainRows: true);
        DeepSeekV41DSparkProposer proposer = new(dspark, state);

        DraftBlock block = proposer.Propose(DeepSeekV41DSparkFixture.Ids.AsSpan(0, 1), BlockDraft);

        Assert.Equal(BlockDraft, block.Tokens.Length);
        Assert.Equal(1, state.Length);
    }

    [Fact]
    public void Asking_For_No_Tokens_Leaves_The_State_Alone()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu, DeepSeekV41DSparkFixture.TargetLayers);
        DeepSeekV41DSpark dspark = DeepSeekV41DSparkFixture.BuildDSpark(cpu);
        DeepSeekV41GenerationState state = new(model, DeepSeekV41DSparkFixture.MaxTokens, recordMainRows: true);
        DeepSeekV41DSparkProposer proposer = new(dspark, state);

        DraftBlock block = proposer.Propose(DeepSeekV41DSparkFixture.Ids.AsSpan(0, 11), 0);

        Assert.Empty(block.Tokens);
        Assert.Equal(0, state.Length);
    }

    [Fact]
    public void A_State_That_Does_Not_Record_The_Head_Input_Is_Refused()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu, DeepSeekV41DSparkFixture.TargetLayers);
        DeepSeekV41DSpark dspark = DeepSeekV41DSparkFixture.BuildDSpark(cpu);
        Assert.Throws<ArgumentException>(() => new DeepSeekV41DSparkProposer(dspark, new DeepSeekV41GenerationState(model, DeepSeekV41DSparkFixture.MaxTokens)));
    }

    [Fact]
    public void A_Scheduler_That_Verifies_Nothing_Keeps_Plain_Decoding_Exact()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu, DeepSeekV41DSparkFixture.TargetLayers);
        DeepSeekV41DSpark dspark = DeepSeekV41DSparkFixture.BuildDSpark(cpu);
        int[] prompt = DeepSeekV41DSparkFixture.Ids[..9];
        const int count = 12;
        List<int> plain = DeepSeekV41DSparkFixture.PlainGreedy(model, prompt, count);

        // one token costs 1000 steps a second and anything longer almost nothing, so the scheduler never verifies a drafted token
        DeepSeekV41GenerationState state = new(model, DeepSeekV41DSparkFixture.MaxTokens, recordMainRows: true);
        ConfidenceScheduler scheduler = new(new SpsProfile([1000, 1, 1, 1, 1, 1]), BlockDraft);
        CountingProposer proposer = new(new DeepSeekV41DSparkProposer(dspark, state, scheduler));
        List<int> speculative = [.. prompt];
        int produced = SpeculativeLoop.Generate(new DeepSeekV41SpeculativeScorer(model, state), proposer, SamplerChain.FromOptions(new SamplingOptions { Greedy = true }),
            SpeculativeTestSupport.Uniform(1), speculative, count, BlockDraft, model.VocabSize);

        Assert.Equal(count, produced);
        Assert.Equal(0, proposer.Drafted);
        Assert.Equal(plain, speculative);
    }

    [Fact]
    public void A_Scheduler_Built_For_A_Different_Block_Is_Refused()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu, DeepSeekV41DSparkFixture.TargetLayers);
        DeepSeekV41DSpark dspark = DeepSeekV41DSparkFixture.BuildDSpark(cpu);
        ConfidenceScheduler wrongBlock = new(new SpsProfile([100, 100, 100, 100, 100, 100, 100, 100]), BlockDraft + 2);
        Assert.Throws<ArgumentException>(() => new DeepSeekV41DSparkProposer(dspark,
            new DeepSeekV41GenerationState(model, DeepSeekV41DSparkFixture.MaxTokens, recordMainRows: true), wrongBlock));
    }

    [Fact]
    public void A_Flat_Profile_Keeps_The_Whole_Block_Of_The_Unscheduled_Proposer()
    {
        using CpuBackend cpu = new();
        DeepSeekV41HostModel model = DeepSeekV41HostModelTests.BuildModel(cpu, DeepSeekV41DSparkFixture.TargetLayers);
        DeepSeekV41DSpark dspark = DeepSeekV41DSparkFixture.BuildDSpark(cpu);
        int[] context = DeepSeekV41DSparkFixture.Ids[..11];

        DeepSeekV41GenerationState plainState = new(model, DeepSeekV41DSparkFixture.MaxTokens, recordMainRows: true);
        DraftBlock unscheduled = new DeepSeekV41DSparkProposer(dspark, plainState).Propose(context, BlockDraft);

        DeepSeekV41GenerationState flatState = new(model, DeepSeekV41DSparkFixture.MaxTokens, recordMainRows: true);
        ConfidenceScheduler scheduler = new(new SpsProfile([100, 100, 100, 100, 100, 100]), BlockDraft);
        DraftBlock scheduled = new DeepSeekV41DSparkProposer(dspark, flatState, scheduler).Propose(context, BlockDraft);

        Assert.Equal(unscheduled.Tokens, scheduled.Tokens);
        Assert.Equal(BlockDraft, scheduled.Tokens.Length);
    }
}

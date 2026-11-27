using HartsyInference.ThreeD.Models.UniRig;

namespace HartsyInference.ThreeD.Tests.UniRig;

public sealed class UniRigGrammarTests
{
    private static UniRigGrammar Started()
    {
        UniRigGrammar grammar = new();
        grammar.Advance(UniRigTokenizer.BosToken);
        grammar.Advance(UniRigTokenizer.ArticulationXlToken);
        return grammar;
    }

    private static HashSet<int> Allowed(UniRigGrammar grammar)
    {
        HashSet<int> set = [];
        for (int token = 0; token < UniRigTokenizer.VocabSize; token++)
        {
            if (grammar.Allows(token))
            {
                set.Add(token);
            }
        }

        return set;
    }

    [Fact]
    public void Start_AllowsOnlyBos()
    {
        UniRigGrammar grammar = new();
        Assert.Equal([UniRigTokenizer.BosToken], Allowed(grammar));
    }

    [Fact]
    public void AfterClass_AllowsPartsCoordinatesAndEos()
    {
        HashSet<int> allowed = Allowed(Started());
        Assert.Equal(256 + 3 + 1, allowed.Count);
        Assert.Contains(UniRigTokenizer.SpringToken, allowed);
        Assert.Contains(UniRigTokenizer.PartToken("body"), allowed);
        Assert.Contains(UniRigTokenizer.EosToken, allowed);
        Assert.DoesNotContain(UniRigTokenizer.BranchToken, allowed);
        Assert.DoesNotContain(UniRigTokenizer.BosToken, allowed);
        Assert.DoesNotContain(UniRigTokenizer.PadToken, allowed);
        Assert.DoesNotContain(UniRigTokenizer.ClsNoneToken, allowed);
    }

    [Fact]
    public void InsideACoordinateTriple_OnlyCoordinatesAreAllowed()
    {
        UniRigGrammar grammar = Started();
        grammar.Advance(10);
        Assert.Equal(256, Allowed(grammar).Count);
        grammar.Advance(20);
        Assert.Equal(256, Allowed(grammar).Count);
        grammar.Advance(30);
        HashSet<int> afterBone = Allowed(grammar);
        Assert.Contains(UniRigTokenizer.BranchToken, afterBone);
        Assert.Contains(UniRigTokenizer.EosToken, afterBone);
        Assert.Contains(UniRigTokenizer.SpringToken, afterBone);
        Assert.Contains(0, afterBone);
        Assert.Contains(255, afterBone);
    }

    [Fact]
    public void Branch_ParentCoordinatesAreCoordinateOnly_ThenUpstreamAllowsMarkersAgain()
    {
        UniRigGrammar grammar = Started();
        foreach (int c in new[] { 1, 2, 3 })
        {
            grammar.Advance(c);
        }

        grammar.Advance(UniRigTokenizer.BranchToken);
        Assert.Equal(256, Allowed(grammar).Count);
        grammar.Advance(4);
        Assert.Equal(256, Allowed(grammar).Count);
        grammar.Advance(5);
        Assert.Equal(256, Allowed(grammar).Count);
        grammar.Advance(6);

        // Upstream's state machine treats the parent triple like a finished bone, so EOS is allowed here
        // even though a branch bone still owes its own three coordinates. Kept as upstream has it.
        Assert.Contains(UniRigTokenizer.EosToken, Allowed(grammar));
        grammar.Advance(7);
        grammar.Advance(8);
        grammar.Advance(9);
        Assert.Contains(UniRigTokenizer.EosToken, Allowed(grammar));
    }

    [Fact]
    public void PartTokenAfterABone_ForcesAJointNext()
    {
        UniRigGrammar grammar = Started();
        foreach (int c in new[] { 1, 2, 3 })
        {
            grammar.Advance(c);
        }

        grammar.Advance(UniRigTokenizer.SpringToken);
        Assert.Equal(256, Allowed(grammar).Count);
    }

    [Fact]
    public void Advance_RejectsAForbiddenToken()
    {
        UniRigGrammar grammar = new();
        Assert.Throws<InvalidOperationException>(() => grammar.Advance(5));
        grammar.Advance(UniRigTokenizer.BosToken);
        grammar.Advance(UniRigTokenizer.ArticulationXlToken);
        grammar.Advance(1);
        Assert.Throws<InvalidOperationException>(() => grammar.Advance(UniRigTokenizer.EosToken));
    }

    [Fact]
    public void Tokenizer_Output_IsAcceptedByTheGrammarEndToEnd()
    {
        int[] tokens = UniRigTokenizer.Tokenize(new UniRigTokenizeInput
        {
            Joints = [0f, 0f, 0f, 0.1f, 0.2f, 0.3f, -0.4f, 0.5f, 0.6f],
            ParentJoints = [0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f],
            Branch = [false, false, true],
            Cls = "articulationxl",
            PartsBias = new Dictionary<int, string?> { [0] = "body", [1] = null },
        });
        UniRigGrammar grammar = new();
        foreach (int token in tokens)
        {
            grammar.Advance(token);
        }
    }

    [Fact]
    public void Step_MasksForbiddenIdsAndTracksHistory()
    {
        UniRigGrammar grammar = Started();
        UniRigGrammarStep step = new(grammar);
        float[] logits = new float[UniRigTokenizer.VocabSize];
        List<int> history = [];
        step.Apply(logits, history);
        Assert.True(float.IsNegativeInfinity(logits[UniRigTokenizer.BranchToken]));
        Assert.Equal(0f, logits[5]);

        history.AddRange([1, 2, 3]);
        logits = new float[UniRigTokenizer.VocabSize];
        step.Apply(logits, history);
        Assert.Equal(0f, logits[UniRigTokenizer.BranchToken]);
        Assert.Equal(0f, logits[UniRigTokenizer.EosToken]);
    }

    [Fact]
    public void Sampler_WithUniformLogits_AlwaysFollowsTheGrammarAndEndsAtEos()
    {
        UniRigSamplingOptions options = new() { Seed = 7, MaxNewTokens = 4000, Temperature = 1f, TopK = 0, TopP = 1f, RepetitionPenalty = 1f };
        int finished = 0;
        for (int seed = 0; seed < 20; seed++)
        {
            float[] logits = new float[UniRigTokenizer.VocabSize];
            int[] sequence;
            try
            {
                sequence = UniRigTokenSampler.Run(logits, _ => logits, options with { Seed = seed }, UniRigTokenizer.ArticulationXlToken, CancellationToken.None);
            }
            catch (HartsyInference.Core.Exceptions.HartsyInferenceException)
            {
                continue;
            }

            UniRigGrammar grammar = new();
            foreach (int token in sequence)
            {
                grammar.Advance(token);
            }

            Assert.Equal(UniRigTokenizer.EosToken, sequence[^1]);
            finished++;
        }

        Assert.True(finished >= 10, $"Only {finished} of 20 random walks reached EOS within the token cap.");
    }

    [Fact]
    public void LogSoftmaxThenRepetitionPenalty_MultipliesTheLogProbabilityOfSeenTokens()
    {
        float[] logits = [2f, 0f, 0f, 0f];
        new UniRigLogSoftmaxStep().Apply(logits, []);
        double sum = Math.Exp(2) + 3;
        Assert.Equal(2 - Math.Log(sum), logits[0], 5);
        Assert.Equal(-Math.Log(sum), logits[1], 5);

        float seenBefore = logits[0];
        new HartsyInference.LLM.Sampling.RepetitionPenaltyStep(3f).Apply(logits, [0]);
        Assert.Equal(seenBefore * 3f, logits[0], 5);
        Assert.Equal(-Math.Log(sum), logits[1], 5);
    }
}

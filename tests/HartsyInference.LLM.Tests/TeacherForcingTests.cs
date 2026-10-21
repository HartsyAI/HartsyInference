using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Teacher forcing is the regression gate's numerics probe for text: the head build scores the base build's greedy tokens.
/// It is only a probe if forcing a run's own tokens reproduces that run's scores exactly, and if forcing other tokens commits them.</summary>
public sealed class TeacherForcingTests
{
    private static (GenericTransformer Model, PrefixCacheTestModel Rig, TransformerConfig Cfg) Build()
    {
        PrefixCacheTestModel rig = new(0x7EAC4E2u);
        TransformerConfig cfg = PrefixCacheTestModel.Config();
        GenericTransformer model = new(cfg);
        model.LoadWeights(rig.Weights(cfg), "model");
        return (model, rig, cfg);
    }

    [Fact]
    public void ForcingTheGreedyStream_ReproducesItsScoresBitForBit()
    {
        (GenericTransformer model, PrefixCacheTestModel rig, TransformerConfig cfg) = Build();
        using GenericTransformer _ = model;
        using CpuBackend backend = new();
        PrefixCacheTestModel.StubTokenizer tokenizer = new();
        int[] prompt = [.. Enumerable.Range(0, 9).Select(_ => rig.NextToken(cfg.VocabSize))];
        SamplingOptions greedy = SamplingOptions.Default with { Greedy = true };

        GenerationResult plain = new TextGenerationPipeline(model, tokenizer, backend)
            .Generate(new GenerationRequest { RawTokenIds = prompt, MaxTokens = 12, Sampling = greedy });
        GenerationResult scored = new TextGenerationPipeline(model, tokenizer, backend)
            .Generate(new GenerationRequest { RawTokenIds = prompt, MaxTokens = 12, Sampling = greedy, RecordTokenScores = true });
        GenerationResult forced = new TextGenerationPipeline(model, tokenizer, backend)
            .Generate(new GenerationRequest { RawTokenIds = prompt, Sampling = greedy, ForcedTokenIds = [.. scored.TokenIds] });

        Assert.Equal(plain.TokenIds, scored.TokenIds);
        Assert.Equal(scored.TokenIds.Count, scored.TokenScores!.Count);
        Assert.Equal(scored.TokenScores, forced.TokenScores);
        foreach (TokenScore s in scored.TokenScores)
        {
            Assert.Equal(s.TopTokenId, s.TokenId);   // greedy: the emitted token is the argmax
            Assert.True(s.LogProb <= 0f && s.LogProb == s.TopLogProb);
        }
    }

    [Fact]
    public void ForcedTokens_AreCommittedInOrder_AndScoredBelowTheTopToken()
    {
        (GenericTransformer model, PrefixCacheTestModel rig, TransformerConfig cfg) = Build();
        using GenericTransformer _ = model;
        using CpuBackend backend = new();
        PrefixCacheTestModel.StubTokenizer tokenizer = new();
        int[] prompt = [.. Enumerable.Range(0, 7).Select(_ => rig.NextToken(cfg.VocabSize))];
        int[] chosen = [.. Enumerable.Range(0, 10).Select(_ => rig.NextToken(cfg.VocabSize))];

        GenerationResult forced = new TextGenerationPipeline(model, tokenizer, backend).Generate(new GenerationRequest
        {
            RawTokenIds = prompt, MaxTokens = 3, Sampling = SamplingOptions.Default with { Greedy = true }, ForcedTokenIds = chosen,
        });

        Assert.Equal(chosen, forced.TokenIds);              // the forced length wins over MaxTokens
        Assert.False(forced.StoppedOnStopToken);
        Assert.Equal(chosen, forced.TokenScores!.Select(s => s.TokenId));
        Assert.All(forced.TokenScores!, s => Assert.True(s.LogProb <= s.TopLogProb && s.TopLogProb <= 0f));
    }

    [Fact]
    public void AForcedStopToken_IsCommitted_AndDoesNotEndTheRun()
    {
        (GenericTransformer model, PrefixCacheTestModel rig, TransformerConfig cfg) = Build();
        using GenericTransformer _ = model;
        using CpuBackend backend = new();
        int[] prompt = [.. Enumerable.Range(0, 5).Select(_ => rig.NextToken(cfg.VocabSize))];
        int[] chosen = [rig.NextToken(cfg.VocabSize), PrefixCacheTestModel.EosId, rig.NextToken(cfg.VocabSize), PrefixCacheTestModel.EosId];

        GenerationResult forced = new TextGenerationPipeline(model, new PrefixCacheTestModel.StubTokenizer(), backend).Generate(new GenerationRequest
        {
            RawTokenIds = prompt, Sampling = SamplingOptions.Default with { Greedy = true }, ForcedTokenIds = chosen,
        });

        Assert.Equal(chosen, forced.TokenIds);
        Assert.False(forced.StoppedOnStopToken);
        Assert.Equal(chosen.Length, forced.TokenScores!.Count);
    }

    [Fact]
    public void AForcedIdOutsideTheVocabulary_IsRefused()
    {
        (GenericTransformer model, _, TransformerConfig cfg) = Build();
        using GenericTransformer owned = model;
        using CpuBackend backend = new();
        GenerationRequest request = new() { RawTokenIds = [1, 2, 3], ForcedTokenIds = [cfg.VocabSize] };
        Assert.Throws<ArgumentException>(() => new TextGenerationPipeline(model, new PrefixCacheTestModel.StubTokenizer(), backend).Generate(request));
    }
}

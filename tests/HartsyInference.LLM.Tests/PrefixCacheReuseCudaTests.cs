using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary><see cref="PrefixCacheReuseTests"/>' multi-turn conversation on CUDA, where the retained KV lives on the
/// device and every grow and shrink is a device-to-device copy. Two comparisons per turn: cached-and-resized against
/// cached-in-place (identical prefill chunking, only the buffer capacities differ — the test of the copy itself),
/// and cached-and-resized against a fresh prefill of the whole prompt (the end-to-end identity a caller relies on).</summary>
[Trait("Category", "GpuIntegration")]
public sealed class PrefixCacheReuseCudaTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _out = output;

    [Theory]
    [InlineData(true, 0f)]
    [InlineData(false, 0.7f)]
    public void GrowingConversation_ResizedOnDevice_MatchesInPlaceReuseAndFreshPrefill(bool greedy, float temperature)
    {
        if (!BackendGate.TryOpen("cuda", _out.WriteLine, out IBackend? gpu))
        {
            return;
        }
        using IBackend backend = gpu!;
        PrefixCacheTestModel rig = new(0x9B05688Cu);
        TransformerConfig cfg = PrefixCacheTestModel.Config(hiddenSize: 64, headDim: 16);
        Dictionary<string, Tensor> w = rig.Weights(cfg);
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        PrefixCacheTestModel.StubTokenizer tokenizer = new();
        SamplingOptions sampling = SamplingOptions.Default with { Greedy = greedy, Temperature = temperature, TopP = 0.95f, Seed = 0 };

        TextGenerationPipeline resizing = new(model, tokenizer, backend);
        TextGenerationPipeline inPlace = new(model, tokenizer, backend);
        using RetainedSequence resized = new();     // 3 tokens of headroom: grown and shrunk by copy every turn
        using RetainedSequence unresized = new();   // sized once for the whole conversation, never copied
        List<int> history = [];
        int previousLength = 0;

        for (int turn = 0; turn < 8; turn++)
        {
            for (int i = 0; i < 5; i++) history.Add(rig.NextToken(cfg.VocabSize));
            GenerationRequest request = new() { RawTokenIds = [.. history], MaxTokens = 8, Sampling = sampling };

            GenerationResult fresh = new TextGenerationPipeline(model, tokenizer, backend).Generate(request);
            GenerationResult viaCopies = resizing.Generate(request with { PrefixCacheHeadroomTokens = 3 }, resized);
            GenerationResult viaInPlace = inPlace.Generate(
                request with { PrefixCacheCapacityHint = 512, PrefixCacheHeadroomTokens = 512 }, unresized);
            _out.WriteLine($"turn {turn}: prompt {fresh.PromptTokens}, reused {viaCopies.ReusedPromptTokens}/{viaInPlace.ReusedPromptTokens}, "
                + $"kept capacity {resized.Cache?.Capacity}/{unresized.Cache?.Capacity}, tokens [{string.Join(",", fresh.TokenIds)}]");

            Assert.True(string.Join(",", viaInPlace.TokenIds) == string.Join(",", viaCopies.TokenIds),
                $"turn {turn}: resizing the retained cache changed the output — [{string.Join(",", viaCopies.TokenIds)}] vs in place "
                + $"[{string.Join(",", viaInPlace.TokenIds)}].");
            Assert.True(string.Join(",", fresh.TokenIds) == string.Join(",", viaCopies.TokenIds),
                $"turn {turn}: cached output differs from a fresh prefill — [{string.Join(",", viaCopies.TokenIds)}] vs fresh "
                + $"[{string.Join(",", fresh.TokenIds)}].");
            Assert.Equal(turn == 0 ? 0 : previousLength, viaCopies.ReusedPromptTokens);
            Assert.Equal(viaInPlace.ReusedPromptTokens, viaCopies.ReusedPromptTokens);
            ISequenceState kept = Assert.IsAssignableFrom<ISequenceState>(resized.Cache);
            Assert.InRange(kept.Capacity - kept.Length, 0, 3);
            previousLength = kept.Length;

            history.AddRange(fresh.TokenIds);
        }

        foreach (Tensor t in w.Values) t.Dispose();
    }
}

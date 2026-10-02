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
/// cached-in-place (identical prefill chunking, only the buffer capacities differ — the test of the copy itself,
/// fatal on the turn it fails), and cached against a fresh prefill of the whole prompt (the end-to-end identity a
/// caller relies on, collected over every turn and asserted at the end so one divergence still reports the rest).</summary>
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
        SamplingOptions sampling =
            SamplingOptions.Default with { Greedy = greedy, Temperature = temperature, TopP = 0.95f, Seed = 0 };

        TextGenerationPipeline resizing = new(model, tokenizer, backend);
        TextGenerationPipeline inPlace = new(model, tokenizer, backend);
        // 3 tokens of headroom: the 512-token first allocation is shrunk at turn 0, every later turn grows by copy, and
        // a turn that stops before MaxTokens is shrunk again.
        using RetainedSequence resized = new();
        using RetainedSequence unresized = new();   // sized once for the whole conversation, never copied
        List<int> history = [];
        List<string> freshMismatches = [];
        int previousLength = 0;

        for (int turn = 0; turn < 8; turn++)
        {
            for (int i = 0; i < 5; i++) history.Add(rig.NextToken(cfg.VocabSize));
            GenerationRequest request = new() { RawTokenIds = [.. history], MaxTokens = 12, Sampling = sampling };

            GenerationResult fresh = new TextGenerationPipeline(model, tokenizer, backend).Generate(request);
            GenerationResult viaCopies = resizing.Generate(
                request with { PrefixCacheCapacityHint = 512, PrefixCacheHeadroomTokens = 3 }, resized);
            GenerationResult viaInPlace = inPlace.Generate(
                request with { PrefixCacheCapacityHint = 512, PrefixCacheHeadroomTokens = 512 }, unresized);
            _out.WriteLine($"turn {turn}: prompt {fresh.PromptTokens}, reused {viaCopies.ReusedPromptTokens}"
                + $"/{viaInPlace.ReusedPromptTokens}, kept capacity {resized.Cache?.Capacity}/{unresized.Cache?.Capacity},"
                + $" tokens [{string.Join(",", fresh.TokenIds)}]");

            string copies = string.Join(",", viaCopies.TokenIds);
            string kept = string.Join(",", viaInPlace.TokenIds);
            Assert.True(kept == copies,
                $"turn {turn}: resizing the retained cache changed the output — [{copies}] vs in place [{kept}].");
            Assert.Equal(viaInPlace.ReusedPromptTokens, viaCopies.ReusedPromptTokens);
            ISequenceState resizedState = Assert.IsAssignableFrom<ISequenceState>(resized.Cache);
            Assert.InRange(resizedState.Capacity - resizedState.Length, 0, 3);
            string freshIds = string.Join(",", fresh.TokenIds);
            if (freshIds != copies)
            {
                freshMismatches.Add($"turn {turn}: cached [{copies}] vs fresh [{freshIds}]");
            }
            else if (freshMismatches.Count == 0)
            {
                // Only while the two histories agree: after a divergence the retained ids no longer prefix the prompt.
                Assert.Equal(turn == 0 ? 0 : previousLength, viaCopies.ReusedPromptTokens);
            }
            previousLength = resizedState.Length;

            history.AddRange(fresh.TokenIds);
        }

        Assert.True(freshMismatches.Count == 0,
            $"cached output differs from a fresh prefill on CUDA: {string.Join("; ", freshMismatches)}");
        foreach (Tensor t in w.Values) t.Dispose();
    }
}

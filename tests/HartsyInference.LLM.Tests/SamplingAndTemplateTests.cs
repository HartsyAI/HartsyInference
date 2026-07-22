using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Unit tests for the sampler chain, chat-template registry, and the GQA K/V repeat op.</summary>
public sealed unsafe class SamplingAndTemplateTests
{
    [Fact]
    public void Greedy_ReturnsArgmax()
    {
        SamplerChain chain = SamplerChain.FromOptions(SamplingOptions.GreedyPreset);
        float[] logits = [0.1f, 0.2f, 5.0f, -1.0f, 0.3f];
        Assert.Equal(2, chain.Next(logits, []));
    }

    [Fact]
    public void TopK1_CollapsesToArgmax()
    {
        SamplerChain chain = SamplerChain.FromOptions(new SamplingOptions { TopK = 1, Temperature = 1.0f, Seed = 7 });
        float[] logits = [0.1f, 9.0f, 0.2f, 0.3f];
        // Only the top-1 logit survives, so sampling is deterministic regardless of the RNG draw.
        Assert.Equal(1, chain.Next(logits, []));
    }

    [Fact]
    public void Sampling_IsDeterministicForSeed()
    {
        float[] Make() => [1.0f, 1.2f, 0.9f, 1.1f, 0.8f, 1.3f];
        SamplerChain a = SamplerChain.FromOptions(new SamplingOptions { Temperature = 1.0f, TopP = 0.9f, Seed = 42 });
        SamplerChain b = SamplerChain.FromOptions(new SamplingOptions { Temperature = 1.0f, TopP = 0.9f, Seed = 42 });
        for (int i = 0; i < 8; i++)
            Assert.Equal(a.Next(Make(), []), b.Next(Make(), []));
    }

    [Fact]
    public void RepetitionPenalty_LowersRepeatedTokenLogit()
    {
        // Positive logit of a token in history should shrink under penalty > 1.
        RepetitionPenaltyStep step = new(2.0f);
        float[] logits = [4.0f, 1.0f, 1.0f];
        step.Apply(logits, [0]);
        Assert.True(logits[0] < 4.0f);
        Assert.Equal(1.0f, logits[1], 5);
    }

    [Fact]
    public void ChatMlTemplate_SingleTurn_MatchesTokenizerEncodeChat()
    {
        using Qwen2Tokenizer tok = new();
        const string prompt = "Hello, who are you?";
        int[] viaTemplate = ChatMlTemplate.EncodeSingleTurn(tok, prompt, systemPrompt: null);
        int[] viaTokenizer = tok.EncodeChat(prompt);
        Assert.Equal(viaTokenizer, viaTemplate);
    }

    [Fact]
    public void ChatTemplateRegistry_ResolvesChatmlAndQwen()
    {
        ChatTemplateRegistry reg = ChatTemplateRegistry.Default;
        Assert.Equal("chatml", reg.Get("chatml").Name);
        Assert.Equal("chatml", reg.Get("qwen").Name);
        Assert.True(reg.TryGet("CHATML", out _)); // case-insensitive
    }

    [Fact]
    public void RepeatKvHeads_FollowsBlockLayout()
    {
        using CpuBackend cpu = new();
        IBackend backend = cpu;
        const int b = 1, hkv = 2, group = 3, len = 2, d = 2;
        using Tensor input = new(new TensorShape(b, hkv, len, d), DType.F32);
        float* pi = (float*)input.DataPointer;
        for (long i = 0; i < input.ElementCount; i++) pi[i] = i;
        using Tensor output = new(new TensorShape(b, hkv * group, len, d), DType.F32);
        backend.RepeatKvHeads(output, input, hkv, group);

        float* po = (float*)output.DataPointer;
        long perHead = (long)len * d;
        for (int qh = 0; qh < hkv * group; qh++)
        {
            int srcHead = qh / group; // block pattern: heads 0..group-1 -> kv0, etc.
            for (long e = 0; e < perHead; e++)
                Assert.Equal(pi[srcHead * perHead + e], po[qh * perHead + e], 6);
        }
    }
}

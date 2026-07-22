using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Correctness gate for <see cref="TextGenerationPipeline"/>'s prompt-lookup speculative decode path
/// (Phase 5b): it must be a PURE speedup — every output, token for token, must be byte-identical to the plain
/// greedy decode loop with speculative decoding turned off. A tiny random-weight CPU model under pure greedy
/// decoding reliably falls into short repeating cycles (a well-known failure mode of untrained models with no
/// repetition penalty), which is exactly the workload that exercises the draft-accept path — so these tests
/// aren't just checking the near-certain "no draft found" fallback, they exercise real multi-token accepts,
/// partial-accept rollback (draft mismatches partway through), and the free bonus-token path.</summary>
public sealed class SpeculativeDecodeTests
{
    private static uint _rng = 0x51ED270Bu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static unsafe Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static unsafe Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    private static TransformerConfig Cfg() => new()
    {
        HiddenSize = 16, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
        IntermediateSize = 32, VocabSize = 24, MaxPositionEmbeddings = 256, AttentionBias = true, QkNorm = false,
    };

    private static Dictionary<string, Tensor> Weights(TransformerConfig c)
    {
        int h = c.HiddenSize, qDim = c.QDim, kvDim = c.KvDim;
        Dictionary<string, Tensor> w = new() { ["model.embed_tokens.weight"] = F2(c.VocabSize, h), ["model.norm.weight"] = Ones(h) };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F2(qDim, h);
            w[$"{p}.self_attn.k_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.v_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.o_proj.weight"] = F2(h, qDim);
            w[$"{p}.self_attn.q_proj.bias"] = F1(qDim);
            w[$"{p}.self_attn.k_proj.bias"] = F1(kvDim);
            w[$"{p}.self_attn.v_proj.bias"] = F1(kvDim);
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
        }
        return w;
    }

    /// <summary>Minimal tokenizer stub: tests drive prompts via <see cref="GenerationRequest.RawTokenIds"/>
    /// (bypassing Encode/chat-template entirely), so only Decode/StopIds need real behavior.</summary>
    private sealed class StubTokenizer : ILlmTokenizer
    {
        public int[] Encode(string text, bool addSpecial) => throw new NotSupportedException();
        public int[] EncodeOrdinary(string text) => throw new NotSupportedException();
        public string Decode(IReadOnlyList<int> ids) => string.Join(",", ids);
        public int? SpecialId(string token) => null;
        public int? BosId => null;
        public int? EosId => 23;
        public IReadOnlyList<int> StopIds => [23];
        public string? BosToken => null;
        public string? EosToken => null;
    }

    private static GenerationRequest Req(int[] promptIds, int maxTokens, SamplingOptions sampling, bool? specDecode) => new()
    {
        RawTokenIds = promptIds,
        MaxTokens = maxTokens,
        Sampling = sampling,
        SpeculativeDecode = specDecode,
    };

    private static (int[] promptIds, TransformerConfig cfg, Dictionary<string, Tensor> weights) Setup(int promptLen, uint seed)
    {
        _rng = seed;
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        int[] prompt = new int[promptLen];
        for (int i = 0; i < promptLen; i++) { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; prompt[i] = (int)(_rng % (uint)(cfg.VocabSize - 1)); }
        return (prompt, cfg, w);
    }

    [Theory]
    [InlineData(3, 40, 0xA5A5u)]
    [InlineData(1, 60, 0xC001u)]
    [InlineData(9, 50, 0xF00Du)]
    public void GreedyNoPenalty_MatchesPlainDecodeExactly(int promptLen, int maxTokens, uint seed)
    {
        (int[] prompt, TransformerConfig cfg, Dictionary<string, Tensor> w) = Setup(promptLen, seed);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        SamplingOptions sampling = SamplingOptions.Default with { Greedy = true };

        TextGenerationPipeline plainPipeline = new(model, tokenizer, backend);
        GenerationResult reference = plainPipeline.Generate(Req(prompt, maxTokens, sampling, specDecode: false));

        TextGenerationPipeline specPipeline = new(model, tokenizer, backend);
        GenerationResult actual = specPipeline.Generate(Req(prompt, maxTokens, sampling, specDecode: true));

        Assert.Equal(reference.StoppedOnStopToken, actual.StoppedOnStopToken);
        Assert.Equal(string.Join(",", reference.TokenIds), string.Join(",", actual.TokenIds));
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Theory]
    [InlineData(1.15f, 30, 0xBEEFu)]
    [InlineData(1.4f, 45, 0x1234u)]
    public void GreedyWithRepetitionPenalty_MatchesPlainDecodeExactly(float penalty, int maxTokens, uint seed)
    {
        // Repetition penalty makes the accepted token at each drafted position depend on the exact history
        // built up so far — the sharpest test that GenerateSpeculative replays history in the same
        // left-to-right order the eager loop does (not e.g. computed once per round then reused stale).
        (int[] prompt, TransformerConfig cfg, Dictionary<string, Tensor> w) = Setup(4, seed);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        SamplingOptions sampling = SamplingOptions.Default with { Greedy = true, RepetitionPenalty = penalty };

        TextGenerationPipeline plainPipeline = new(model, tokenizer, backend);
        GenerationResult reference = plainPipeline.Generate(Req(prompt, maxTokens, sampling, specDecode: false));

        TextGenerationPipeline specPipeline = new(model, tokenizer, backend);
        GenerationResult actual = specPipeline.Generate(Req(prompt, maxTokens, sampling, specDecode: true));

        Assert.Equal(string.Join(",", reference.TokenIds), string.Join(",", actual.TokenIds));
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void MaxTokensBoundary_MatchesPlainDecodeExactly(int maxTokens)
    {
        // Small MaxTokens values stress the per-round budget clamp (draft length capped so a round can never
        // emit more tokens than remain) at its tightest — 1 token forces k=0 every round, 2-3 exercise the
        // "draft would overflow budget, get clipped" path directly.
        (int[] prompt, TransformerConfig cfg, Dictionary<string, Tensor> w) = Setup(5, 0x7777u);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        SamplingOptions sampling = SamplingOptions.Default with { Greedy = true };

        TextGenerationPipeline plainPipeline = new(model, tokenizer, backend);
        GenerationResult reference = plainPipeline.Generate(Req(prompt, maxTokens, sampling, specDecode: false));

        TextGenerationPipeline specPipeline = new(model, tokenizer, backend);
        GenerationResult actual = specPipeline.Generate(Req(prompt, maxTokens, sampling, specDecode: true));

        Assert.Equal(string.Join(",", reference.TokenIds), string.Join(",", actual.TokenIds));
        Assert.True(actual.TokenIds.Count <= maxTokens);
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void NonGreedyRequest_IsIgnoredByTheOptInGate()
    {
        // SpeculativeDecode=true on a non-greedy request must be a no-op (dispatch gate requires Greedy) — a
        // stochastic multinomial draw can't be reproduced out of order, so this must fall through to the
        // ordinary loop rather than silently produce wrong (non-reproducible) output.
        (int[] prompt, TransformerConfig cfg, Dictionary<string, Tensor> w) = Setup(3, 0x9999u);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        SamplingOptions sampling = SamplingOptions.Default with { Greedy = false, Seed = 42 };

        TextGenerationPipeline p1 = new(model, tokenizer, backend);
        GenerationResult a = p1.Generate(Req(prompt, 20, sampling, specDecode: false));
        TextGenerationPipeline p2 = new(model, tokenizer, backend);
        GenerationResult b = p2.Generate(Req(prompt, 20, sampling, specDecode: true));

        Assert.Equal(string.Join(",", a.TokenIds), string.Join(",", b.TokenIds));
        foreach (Tensor t in w.Values) t.Dispose();
    }
}

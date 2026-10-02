using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Tests;

/// <summary>Random weights and prompt tokens for the prefix-cache identity tests, from a per-instance xorshift
/// stream: the CPU and CUDA classes run in parallel and must not share random state.</summary>
internal sealed class PrefixCacheTestModel(uint seed)
{
    /// <summary>End-of-sequence id of <see cref="StubTokenizer"/>.</summary>
    public const int EosId = 36;

    private uint _state = seed;

    /// <summary>A two-layer GQA decoder small enough to run every comparison on the CPU in milliseconds.</summary>
    public static TransformerConfig Config(int hiddenSize = 16, int headDim = 4) => new()
    {
        HiddenSize = hiddenSize, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = headDim,
        IntermediateSize = 2 * hiddenSize, VocabSize = EosId + 1, MaxPositionEmbeddings = 512, AttentionBias = true,
        QkNorm = false,
    };

    /// <summary>A prompt token that is never the end-of-sequence id.</summary>
    public int NextToken(int vocab) => (int)(NextRaw() % (uint)(vocab - 1));

    /// <summary>Fresh random weights for <paramref name="c"/>; the caller disposes them.</summary>
    public Dictionary<string, Tensor> Weights(TransformerConfig c)
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

    /// <summary>A tensor of random values in [-0.1, 0.1].</summary>
    public unsafe Tensor Random(TensorShape shape)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            p[i] = ((NextRaw() & 0xFFFF) / 65535f - 0.5f) * 0.2f;
        }
        return t;
    }

    private uint NextRaw()
    {
        _state ^= _state << 13;
        _state ^= _state >> 17;
        _state ^= _state << 5;
        return _state;
    }

    private Tensor F2(int a, int b) => Random(new TensorShape(a, b));

    private Tensor F1(int a) => Random(new TensorShape(a));

    private static unsafe Tensor Ones(int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++)
        {
            p[i] = 1f;
        }
        return t;
    }

    /// <summary>Bypasses Encode/chat templates entirely: every test drives prompts through
    /// <c>GenerationRequest.RawTokenIds</c>.</summary>
    internal sealed class StubTokenizer : ILlmTokenizer
    {
        public int[] Encode(string text, bool addSpecial) => throw new NotSupportedException();
        public int[] EncodeOrdinary(string text) => throw new NotSupportedException();
        public string Decode(IReadOnlyList<int> ids) => string.Join(",", ids);
        public int? SpecialId(string token) => null;
        public int? BosId => null;
        public int? EosId => PrefixCacheTestModel.EosId;
        public IReadOnlyList<int> StopIds => [PrefixCacheTestModel.EosId];
        public string? BosToken => null;
        public string? EosToken => null;
    }
}

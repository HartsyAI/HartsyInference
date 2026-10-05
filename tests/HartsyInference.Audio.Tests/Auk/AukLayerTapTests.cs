using HartsyInference.Audio.Models.Auk;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>Layer-tap index semantics on a tiny headless Qwen2 body: tap(i) equals the oracle forward over layers [0, i+1) with no final norm.</summary>
public sealed class AukLayerTapTests
{
    private const int Hidden = 32;
    private const int Heads = 4;
    private const int KvHeads = 2;
    private const int Inter = 32;
    private const int NumLayers = 4;
    private const int T = 5;

    private static Qwen2Config Config() => new()
    {
        HiddenSize = Hidden, NumHiddenLayers = NumLayers, NumAttentionHeads = Heads, NumKeyValueHeads = KvHeads,
        IntermediateSize = Inter, VocabSize = 8, MaxPositionEmbeddings = 64, AttentionBias = false, TieWordEmbeddings = false,
    };

    private static Tensor Rand(Random rng, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5) * 0.6f;
        return t;
    }

    private static Tensor Ones(int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        t.AsSpan<float>().Fill(1f);
        return t;
    }

    private static Dictionary<string, Tensor> Weights(List<Tensor> owned)
    {
        Random rng = new(3);
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
        void Add(string key, Tensor t) { owned.Add(t); w[key] = t; }
        Add("model.norm.weight", Ones(Hidden));
        for (int l = 0; l < NumLayers; l++)
        {
            string p = $"model.layers.{l}";
            Add($"{p}.input_layernorm.weight", Ones(Hidden));
            Add($"{p}.post_attention_layernorm.weight", Ones(Hidden));
            Add($"{p}.self_attn.q_proj.weight", Rand(rng, Hidden, Hidden));
            Add($"{p}.self_attn.k_proj.weight", Rand(rng, KvHeads * 8, Hidden));
            Add($"{p}.self_attn.v_proj.weight", Rand(rng, KvHeads * 8, Hidden));
            Add($"{p}.self_attn.o_proj.weight", Rand(rng, Hidden, Hidden));
            Add($"{p}.mlp.gate_proj.weight", Rand(rng, Inter, Hidden));
            Add($"{p}.mlp.up_proj.weight", Rand(rng, Inter, Hidden));
            Add($"{p}.mlp.down_proj.weight", Rand(rng, Hidden, Inter));
        }
        return w;
    }

    private static Tensor Embeds()
    {
        Random rng = new(99);
        Tensor t = new(new TensorShape(1, T, Hidden), DType.F32);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static float[] Forward(Qwen2Model model, CpuBackend backend, int? endLayer, bool norm, Action<int, Tensor>? tap)
    {
        using Tensor embeds = Embeds();
        IKvCache cache = model.CreateDecodeCache(64);
        using Tensor result = model.ForwardEmbeds(backend, embeds, 1, T, 0, cache, 0, endLayer, norm, tap);
        return result.AsSpan<float>().ToArray();
    }

    [Fact]
    public void Tap_EqualsPartialStackOracle_AndNullTapIsIdentical_AndFusionConsumesIt()
    {
        List<Tensor> owned = [];
        try
        {
            CpuBackend backend = new();
            using Qwen2Model model = new(Config());
            model.LoadWeightsHeadless(Weights(owned), "model");

            List<float[]> taps = [];
            List<int> indices = [];
            float[] full = Forward(model, backend, null, true, (i, h) => { indices.Add(i); taps.Add(h.AsSpan<float>().ToArray()); });
            Assert.Equal([0, 1, 2, 3], indices);
            Assert.Equal(full, Forward(model, backend, null, true, null));

            for (int i = 0; i < NumLayers; i++)
            {
                float[] oracle = Forward(model, backend, i + 1, false, null);
                Assert.Equal(oracle, taps[i]);
            }
            Assert.NotEqual(taps[0], taps[1]);

            using AukLayerFusion fusion = new(backend, [0.5f, -0.2f, 0.1f, 0.9f], 1.5f, Hidden);
            fusion.Begin(T);
            Forward(model, backend, null, true, fusion.OnLayer);
            // Fusion path above leaves the last layer incomplete by design; finish it with the final-normed output.
            using Tensor finalNormed = new(new TensorShape(1, T, Hidden), DType.F32);
            full.CopyTo(finalNormed.AsSpan<float>());
            using Tensor fused = fusion.Complete(finalNormed);
            Assert.Equal(T * Hidden, (int)fused.Shape.ElementCount);
        }
        finally
        {
            foreach (Tensor t in owned) t.Dispose();
        }
    }
}

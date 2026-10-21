using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.TextEncoders;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A sliding layer attends only to the last <see cref="LlamaStyleEncoderConfig.SlidingWindow"/> tokens (Gemma 3's
/// local layers). The encoder used to give those layers the plain causal mask, which only differs once a prompt is longer
/// than the window.</summary>
public sealed class LlamaStyleEncoderSlidingWindowTests
{
    private const int Hidden = 16, Heads = 2, HeadDim = 8, Vocab = 8, Window = 2, Seq = 5;

    [Theory]
    [InlineData("sliding_attention", true)]
    [InlineData("full_attention", false)]   // control: a full layer still sees token 0 from token 4
    public void ASlidingLayerSeesOnlyItsWindow(string layerType, bool windowed)
    {
        LlamaStyleEncoderConfig config = new()
        {
            HiddenSize = Hidden, NumLayers = 1, NumQueryHeads = Heads, NumKvHeads = Heads, HeadDim = HeadDim,
            IntermediateSize = 2 * Hidden, VocabSize = Vocab, HasFinalNorm = false,
            LayerAttentionTypes = [layerType], SlidingWindow = Window,
        };
        using CpuBackend backend = new();
        using LlamaStyleEncoder encoder = new(config);
        Dictionary<string, Tensor> weights = Weights();
        encoder.LoadWeights(weights);

        using Tensor a = encoder.Encode(backend, [[1, 2, 3, 4, 5]]);
        using Tensor b = encoder.Encode(backend, [[6, 2, 3, 4, 5]]);
        float[] Row(Tensor t, int token) => t.AsReadOnlySpan<float>().Slice(token * Hidden, Hidden).ToArray();

        // Token 4's window is tokens 3 and 4, so token 0 cannot reach it; token 1's window holds token 0.
        if (windowed) Assert.Equal(Row(a, 4), Row(b, 4));
        else Assert.NotEqual(Row(a, 4), Row(b, 4));
        Assert.NotEqual(Row(a, 1), Row(b, 1));
        foreach (Tensor t in weights.Values) t.Dispose();
    }

    private static Dictionary<string, Tensor> Weights()
    {
        Random rng = new(7);
        Tensor R(params long[] dims)
        {
            Tensor t = new(new TensorShape(dims), DType.F32);
            Span<float> p = t.AsSpan<float>();
            for (int i = 0; i < p.Length; i++) p[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            return t;
        }
        const string l = "model.layers.0";
        return new()
        {
            ["model.embed_tokens.weight"] = R(Vocab, Hidden),
            [$"{l}.input_layernorm.weight"] = R(Hidden),
            [$"{l}.post_attention_layernorm.weight"] = R(Hidden),
            [$"{l}.self_attn.q_proj.weight"] = R(Heads * HeadDim, Hidden),
            [$"{l}.self_attn.k_proj.weight"] = R(Heads * HeadDim, Hidden),
            [$"{l}.self_attn.v_proj.weight"] = R(Heads * HeadDim, Hidden),
            [$"{l}.self_attn.o_proj.weight"] = R(Hidden, Heads * HeadDim),
            [$"{l}.self_attn.q_norm.weight"] = R(HeadDim),
            [$"{l}.self_attn.k_norm.weight"] = R(HeadDim),
            [$"{l}.mlp.gate_proj.weight"] = R(2 * Hidden, Hidden),
            [$"{l}.mlp.up_proj.weight"] = R(2 * Hidden, Hidden),
            [$"{l}.mlp.down_proj.weight"] = R(Hidden, 2 * Hidden),
        };
    }
}

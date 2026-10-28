using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The streaming encoder layer where the hidden width differs from heads * head_dim, as on small and medium.
/// The attention merge buffer must be sized to the attention output, not to the hidden width.</summary>
public sealed class MoonshineStreamingEncoderLayerTests
{
    private const int Intermediate = 24;
    private const int SeqLen = 9;
    private const string Prefix = "layer";

    // (hidden, heads, headDim). The first has hidden wider than the attention output, as on medium (768 vs 640);
    // the second has it narrower.
    [Theory]
    [InlineData(16, 2, 6)]
    [InlineData(12, 2, 8)]
    public void Forward_WhenAttentionWidthDiffersFromHiddenWidth_ReturnsTheHiddenShape(
        int hidden, int heads, int headDim)
    {
        int inner = heads * headDim;
        MoonshineConfig cfg = new()
        {
            EncoderHiddenSize = hidden,
            EncoderNumHeads = heads,
            EncoderHeadDim = headDim,
            EncoderIntermediateSize = Intermediate,
        };
        Random random = new(7);
        Dictionary<string, Tensor> weights = Weights(random, hidden, inner);
        using IBackend backend = new CpuBackend();
        MoonshineStreamingEncoderLayer layer = new(cfg, (Left: 4, Right: 4));
        layer.LoadWeights(weights, Prefix);
        Tensor input = RandomTensor(random, new TensorShape(1, SeqLen, hidden));

        Tensor output = layer.Forward(backend, input, SeqLen);

        try
        {
            Assert.Equal(SeqLen * hidden, output.ElementCount);
            float[] values = output.AsSpan<float>().ToArray();
            Assert.All(values, value => Assert.True(float.IsFinite(value)));
        }
        finally
        {
            output.Dispose();
            input.Dispose();
            foreach (Tensor weight in weights.Values)
            {
                weight.Dispose();
            }
        }
    }

    private static Dictionary<string, Tensor> Weights(Random random, int hidden, int inner)
    {
        string p = Prefix;
        return new Dictionary<string, Tensor>
        {
            [$"{p}.input_layernorm.gamma"] = RandomTensor(random, new TensorShape(hidden)),
            [$"{p}.self_attn.q_proj.weight"] = RandomTensor(random, new TensorShape(inner, hidden)),
            [$"{p}.self_attn.k_proj.weight"] = RandomTensor(random, new TensorShape(inner, hidden)),
            [$"{p}.self_attn.v_proj.weight"] = RandomTensor(random, new TensorShape(inner, hidden)),
            [$"{p}.self_attn.o_proj.weight"] = RandomTensor(random, new TensorShape(hidden, inner)),
            [$"{p}.post_attention_layernorm.gamma"] = RandomTensor(random, new TensorShape(hidden)),
            [$"{p}.mlp.fc1.weight"] = RandomTensor(random, new TensorShape(Intermediate, hidden)),
            [$"{p}.mlp.fc1.bias"] = RandomTensor(random, new TensorShape(Intermediate)),
            [$"{p}.mlp.fc2.weight"] = RandomTensor(random, new TensorShape(hidden, Intermediate)),
            [$"{p}.mlp.fc2.bias"] = RandomTensor(random, new TensorShape(hidden)),
        };
    }

    private static Tensor RandomTensor(Random random, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)(random.NextDouble() * 0.2 - 0.1);
        }
        return tensor;
    }
}

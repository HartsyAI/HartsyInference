using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests.Stt;

/// <summary>Streaming encoder layer whose hidden width is wider than heads * head_dim, as on medium.</summary>
public sealed class MoonshineStreamingEncoderLayerTests
{
    private const int Intermediate = 24;
    private const int SeqLen = 9;
    private const string Prefix = "layer";

    // Fails on a pre-fix merge buffer through LinearTransB's element-count check.
    [Fact]
    public void Forward_WhenHiddenWidthIsWiderThanAttentionWidth_ReturnsTheHiddenShape()
    {
        const int hidden = 16;
        const int heads = 2;
        const int headDim = 6;
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
        Tensor input = RandomTensor(random, new TensorShape(1, SeqLen, hidden));
        using IBackend backend = new CpuBackend();
        Tensor? output = null;

        try
        {
            MoonshineStreamingEncoderLayer layer = new(cfg, (Left: 4, Right: 4));
            layer.LoadWeights(weights, Prefix);
            output = layer.Forward(backend, input, SeqLen);

            Assert.Equal(new TensorShape(1, SeqLen, hidden), output.Shape);
            float[] values = output.AsSpan<float>().ToArray();
            Assert.All(values, value => Assert.True(float.IsFinite(value)));
        }
        finally
        {
            output?.Dispose();
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

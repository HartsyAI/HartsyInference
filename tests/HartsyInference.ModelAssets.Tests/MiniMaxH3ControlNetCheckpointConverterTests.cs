using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.Core.Exceptions;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Locks the upstream VideoX-Fun-to-native key and row-layout conversion used by H3 Fun ControlNet.</summary>
public unsafe class MiniMaxH3ControlNetCheckpointConverterTests
{
    [Fact]
    public void ConvertFusesQkvAndSwapsDiffusersSwiGluHalves()
    {
        Dictionary<string, Tensor> source = new Dictionary<string, Tensor>
        {
            ["controlnet.control_blocks.0.attn.to_q.weight"] = Matrix(2, 2, 1f),
            ["controlnet.control_blocks.0.attn.to_k.weight"] = Matrix(2, 2, 5f),
            ["controlnet.control_blocks.0.attn.to_v.weight"] = Matrix(2, 2, 9f),
            ["controlnet.control_blocks.0.ff.net.0.proj.weight"] = Matrix(4, 2, 13f),
            ["controlnet.control_blocks.0.ff.net.2.weight"] = Matrix(2, 2, 21f),
            ["controlnet.control_blocks.0.attn.norm_q.weight"] = Matrix(1, 2, 25f),
            ["controlnet.control_blocks.0.attn.to_out.0.weight"] = Matrix(2, 2, 27f),
        };
        Dictionary<string, Tensor>? converted = null;
        try
        {
            converted = MiniMaxH3ControlNetCheckpointConverter.Convert(source);

            Assert.Equal(
                [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f],
                Values(converted["control_blocks.0.attn.qkv_proj.weight"]));
            Assert.Equal(
                [17f, 18f, 19f, 20f, 13f, 14f, 15f, 16f],
                Values(converted["control_blocks.0.mlp.fc1.weight"]));
            Assert.True(converted.ContainsKey("control_blocks.0.mlp.fc2.weight"));
            Assert.True(converted.ContainsKey("control_blocks.0.attn.q_norm.weight"));
            Assert.True(converted.ContainsKey("control_blocks.0.attn.out_proj.weight"));
            Assert.DoesNotContain(converted.Keys, key => key.Contains(".to_q.", StringComparison.Ordinal));
        }
        finally
        {
            DisposeDistinct(source.Values.Concat(
                converted is null ? Enumerable.Empty<Tensor>() : converted.Values));
        }
    }

    [Fact]
    public void ConvertRefusesAnOrphanSplitProjection()
    {
        Dictionary<string, Tensor> source = new Dictionary<string, Tensor>
        {
            ["control_blocks.0.attn.to_k.weight"] = Matrix(2, 2, 1f),
        };
        try
        {
            HartsyInferenceException failure = Assert.Throws<HartsyInferenceException>(
                () => MiniMaxH3ControlNetCheckpointConverter.Convert(source));
            Assert.Contains("without a matching split Q projection", failure.Message, StringComparison.Ordinal);
        }
        finally
        {
            DisposeDistinct(source.Values);
        }
    }

    private static Tensor Matrix(int rows, int columns, float first)
    {
        Tensor tensor = new Tensor(new TensorShape(rows, columns), DType.F32);
        float* pointer = (float*)tensor.DataPointer;
        for (int index = 0; index < rows * columns; index++)
        {
            pointer[index] = first + index;
        }
        return tensor;
    }

    private static float[] Values(Tensor tensor)
    {
        float[] values = new float[checked((int)tensor.ElementCount)];
        float* pointer = (float*)tensor.DataPointer;
        for (int index = 0; index < values.Length; index++)
        {
            values[index] = pointer[index];
        }
        return values;
    }

    private static void DisposeDistinct(IEnumerable<Tensor> tensors)
    {
        foreach (Tensor tensor in tensors.Distinct())
        {
            tensor.Dispose();
        }
    }
}

using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.MiniMaxH3;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Locks the affine weight and mandatory DC-bias transform used by the local pruned-branch converter.</summary>
public unsafe class MiniMaxH3ControlNetPrunedConverterTests
{
    [Fact]
    public void RebaseProjectionAppliesCurveMapAndDcBias()
    {
        using Tensor denseWeight = Values(new TensorShape(2, 3), [1f, 2f, 3f, -1f, 0.5f, 4f]);
        using Tensor denseBias = Values(new TensorShape(2), [0.25f, -0.75f]);
        using Tensor intercept = Values(new TensorShape(3), [0.5f, -1f, 2f]);
        using Tensor projection = Values(new TensorShape(3, 2), [1f, 2f, 3f, 4f, 5f, 6f]);
        using MiniMaxH3PddAffineBasis basis = new MiniMaxH3PddAffineBasis(intercept, projection, 1e-6);

        (Tensor weight, Tensor bias) = MiniMaxH3ControlNetPrunedConverter.RebaseProjection(
            denseWeight, denseBias, basis);
        using (weight)
        using (bias)
        {
            Assert.Equal([22f, 28f, 20.5f, 24f], Read(weight));
            Assert.Equal([4.75f, 6.25f], Read(bias));
        }
    }

    private static Tensor Values(TensorShape shape, IReadOnlyList<float> values)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        Assert.Equal(tensor.ElementCount, values.Count);
        float* pointer = (float*)tensor.DataPointer;
        for (int index = 0; index < values.Count; index++)
        {
            pointer[index] = values[index];
        }
        return tensor;
    }

    private static float[] Read(Tensor tensor)
    {
        float[] values = new float[checked((int)tensor.ElementCount)];
        float* pointer = (float*)tensor.DataPointer;
        for (int index = 0; index < values.Length; index++)
        {
            values[index] = pointer[index];
        }
        return values;
    }
}

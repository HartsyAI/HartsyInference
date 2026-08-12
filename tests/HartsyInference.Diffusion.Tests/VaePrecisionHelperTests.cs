using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed unsafe class VaePrecisionHelperTests
{
    [Fact]
    public void CastVaeWeights_MixedDtypes_BorrowsMatchingTensor_AndReturnsOwnedCast()
    {
        using Tensor matching = Values(DType.F32, [1.25f, -2.5f, 3.75f]);
        using Tensor needsCast = Values(DType.F16, [0.5f, -1.5f, 2.25f]);
        Dictionary<string, Tensor> source = new()
        {
            ["matching"] = matching,
            ["needs_cast"] = needsCast,
        };

        Dictionary<string, Tensor>? cast = null;
        try
        {
            cast = VaePrecisionHelper.CastVaeWeights(source, DType.F32);

            Assert.Same(matching, cast["matching"]);
            Assert.NotSame(needsCast, cast["needs_cast"]);
            Assert.Equal(DType.F32, cast["needs_cast"].DType);
            Assert.Equal([0.5f, -1.5f, 2.25f], Snapshot(cast["needs_cast"]));

            cast["needs_cast"].Dispose();
            cast.Remove("needs_cast");

            Assert.Equal([1.25f, -2.5f, 3.75f], Snapshot(matching));
            Assert.Equal([0.5f, -1.5f, 2.25f], Snapshot(needsCast));
        }
        finally
        {
            if (cast is not null)
            {
                foreach (Tensor tensor in cast.Values)
                {
                    if (!ReferenceEquals(tensor, matching) && !ReferenceEquals(tensor, needsCast))
                        tensor.Dispose();
                }
            }
        }
    }

    private static Tensor Values(DType dtype, float[] values)
    {
        using Tensor f32 = new(new TensorShape(values.Length), DType.F32);
        values.CopyTo(new Span<float>(f32.DataPointer, values.Length));
        return dtype == DType.F32 ? f32.To(DeviceKind.Cpu) : f32.CastTo(dtype);
    }

    private static float[] Snapshot(Tensor tensor)
    {
        using Tensor f32 = tensor.DType == DType.F32 ? tensor.To(DeviceKind.Cpu) : tensor.CastTo(DType.F32);
        float[] values = new float[tensor.ElementCount];
        new ReadOnlySpan<float>(f32.DataPointer, values.Length).CopyTo(values);
        return values;
    }
}

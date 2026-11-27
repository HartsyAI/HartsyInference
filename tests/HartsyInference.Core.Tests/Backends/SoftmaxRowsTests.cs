using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

public sealed class SoftmaxRowsTests
{
    [Fact]
    public void SoftmaxRows_NormalizesEachRow_AndIsShiftInvariant()
    {
        float[] x = { 1f, 2f, 3f, 1000f, 1001f, 1002f };
        using Tensor input = Make(x, 2, 3);
        using Tensor output = new(new TensorShape(2, 3), DType.F32);
        IBackend backend = new CpuBackend();
        backend.SoftmaxRows(output, input, 2, 3);

        float[] y = output.AsReadOnlySpan<float>().ToArray();
        float[] expectedRow = { 0.09003057f, 0.24472847f, 0.66524096f };
        for (int r = 0; r < 2; r++)
        {
            Assert.Equal(1f, y[r * 3] + y[r * 3 + 1] + y[r * 3 + 2], 5);
            for (int c = 0; c < 3; c++) Assert.Equal(expectedRow[c], y[r * 3 + c], 5);
        }
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}

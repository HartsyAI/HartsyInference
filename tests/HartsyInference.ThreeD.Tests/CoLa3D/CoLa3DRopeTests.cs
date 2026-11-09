using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.CoLa3D;
using Xunit;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

public sealed class CoLa3DRopeTests
{
    [Fact]
    public void Plan_UsesOneFrequencyScheduleOnEachOfThreeAxes()
    {
        const float tolerance = 1e-5f;
        CoLa3DRopeTable table;
        using (table = CoLa3DRope.Build(CoLa3DRope.Plan(64), [3f, 5f, 7f], tokens: 1))
        {
            ReadOnlySpan<float> cos = table.Cos.AsReadOnlySpan<float>();
            ReadOnlySpan<float> sin = table.Sin.AsReadOnlySpan<float>();
            Assert.Equal(30, cos.Length);

            // Pair 0 turns with x at the highest frequency; pair 10 is the first y pair; pair 20 the first z pair.
            AssertAngle(cos, sin, 0, 3.0, tolerance);
            AssertAngle(cos, sin, 10, 5.0, tolerance);
            AssertAngle(cos, sin, 20, 7.0, tolerance);
            AssertAngle(cos, sin, 1, 3.0 * Math.Pow(10000.0, -0.1), tolerance);
        }

        Assert.Equal(60, CoLa3DRope.RotaryDim(64));
    }

    [Fact]
    public void Apply_RotatesEachInterleavedPairAndLeavesTheIdentityPairAlone()
    {
        using CpuBackend backend = new();
        using CoLa3DRopeTable table = CoLa3DRope.Build(CoLa3DRope.Plan(8), [1f, 2f, 3f], tokens: 1);
        using Tensor x = new(new TensorShape(1, 1, 1, 8), DType.F32);
        float[] input = [0.5f, -1f, 2f, 0.25f, -0.75f, 1.5f, 0.3f, -0.6f];
        input.AsSpan().CopyTo(x.AsSpan<float>());

        CoLa3DRope.Apply(backend, x, table, CoLa3DRope.RotaryDim(8));

        float[] expected = (float[])input.Clone();
        double[] angles = [1.0, 2.0, 3.0];
        for (int pair = 0; pair < 3; pair++)
        {
            float c = (float)Math.Cos(angles[pair]), s = (float)Math.Sin(angles[pair]);
            float a = input[2 * pair], b = input[2 * pair + 1];
            expected[2 * pair] = a * c - b * s;
            expected[2 * pair + 1] = a * s + b * c;
        }

        ReadOnlySpan<float> actual = x.AsReadOnlySpan<float>();
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 5);
    }

    private static void AssertAngle(ReadOnlySpan<float> cos, ReadOnlySpan<float> sin, int pair, double angle, float tolerance)
    {
        Assert.Equal((float)Math.Cos(angle), cos[pair], tolerance);
        Assert.Equal((float)Math.Sin(angle), sin[pair], tolerance);
    }
}

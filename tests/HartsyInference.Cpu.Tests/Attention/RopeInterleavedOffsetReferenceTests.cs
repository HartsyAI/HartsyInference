using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using static HartsyInference.Cpu.Tests.Attention.AttentionFixtures;

namespace HartsyInference.Cpu.Tests.Attention;

public sealed class RopeInterleavedOffsetReferenceTests
{
    private static readonly System.Text.Json.JsonElement Fx = Load("rope_interleaved_offset.json");

    private static void Check(string input, string rotated, string inverse, params long[] shape)
    {
        int rd = Fx.GetProperty("rotaryDim").GetInt32(), off = Fx.GetProperty("dimOffset").GetInt32();
        int len = Fx.GetProperty("length").GetInt32();
        using Tensor cos = F32(Floats(Fx.GetProperty("cos")), 1, len, rd / 2);
        using Tensor sin = F32(Floats(Fx.GetProperty("sin")), 1, len, rd / 2);
        using Tensor negSin = F32(Floats(Fx.GetProperty("sin")).Select(v => -v).ToArray(), 1, len, rd / 2);
        using Tensor x = F32(Floats(Fx.GetProperty(input)), shape);
        using CpuBackend cpu = new();
        cpu.ApplyRopeInterleaved(x, cos, sin, rd, off);
        Assert.True(MaxAbsDiff(Floats(Fx.GetProperty(rotated)), ReadF32(x)) < 1e-6f, "forward");
        cpu.ApplyRopeInterleaved(x, cos, negSin, rd, off);
        Assert.True(MaxAbsDiff(Floats(Fx.GetProperty(inverse)), ReadF32(x)) < 1e-6f, "inverse via negated sin");
        Assert.True(MaxAbsDiff(Floats(Fx.GetProperty(input)), ReadF32(x)) < 1e-5f, "round trip");
    }

    [Fact]
    public void Rank4_Offset_Rotation_And_Its_Negated_Sin_Inverse_Match_Upstream() =>
        Check("q", "qRotated", "qInverse", 1, Fx.GetProperty("length").GetInt32(), Fx.GetProperty("heads").GetInt32(), Fx.GetProperty("dim").GetInt32());

    [Fact]
    public void Elements_Outside_The_Rotary_Slice_Are_Untouched()
    {
        int rd = Fx.GetProperty("rotaryDim").GetInt32(), off = Fx.GetProperty("dimOffset").GetInt32(), len = Fx.GetProperty("length").GetInt32();
        int dim = Fx.GetProperty("dim").GetInt32();
        float[] before = Floats(Fx.GetProperty("k"));
        using Tensor cos = F32(Floats(Fx.GetProperty("cos")), 1, len, rd / 2), sin = F32(Floats(Fx.GetProperty("sin")), 1, len, rd / 2);
        using Tensor x = F32(before, 1, len, dim);
        using CpuBackend cpu = new();
        cpu.ApplyRopeInterleaved(x, cos, sin, rd, off);
        float[] after = ReadF32(x);
        for (int i = 0; i < after.Length; i++)
            if (i % dim < off || i % dim >= off + rd) Assert.Equal(before[i], after[i]);
    }

    [Fact]
    public void Invalid_Operands_Throw()
    {
        using Tensor cos = F32(new float[4], 1, 1, 4), sin = F32(new float[4], 1, 1, 4), x = F32(new float[16], 1, 1, 16);
        using CpuBackend cpu = new();
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.ApplyRopeInterleaved(x, cos, sin, 8, 9));
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.ApplyRopeInterleaved(x, cos, sin, 8, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.ApplyRopeInterleaved(x, cos, sin, 7, 0));
    }
}

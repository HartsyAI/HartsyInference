using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using static HartsyInference.Cpu.Tests.Attention.AttentionFixtures;

namespace HartsyInference.Cpu.Tests.Attention;

public sealed class HcReferenceTests
{
    private static readonly System.Text.Json.JsonElement Fx = Load("hc_mix.json");

    private static float[] Named(System.Text.Json.JsonElement c, string n) => Floats(c.GetProperty(n));

    [Fact]
    public void Split_Sinkhorn_Matches_The_Torch_Port_For_1_3_And_20_Iterations()
    {
        int hc = Fx.GetProperty("hc").GetInt32(), t = Fx.GetProperty("tokens").GetInt32();
        float eps = (float)Fx.GetProperty("eps").GetDouble();
        using Tensor mixes = F32(Floats(Fx.GetProperty("mixes")), t, (2 + hc) * hc);
        using Tensor scale = F32(Floats(Fx.GetProperty("scale")), 3), bias = F32(Floats(Fx.GetProperty("bias")), (2 + hc) * hc);
        using CpuBackend cpu = new();
        foreach (System.Text.Json.JsonElement c in Fx.GetProperty("cases").EnumerateArray())
        {
            using Tensor pre = Empty(DType.F32, t, hc), post = Empty(DType.F32, t, hc), comb = Empty(DType.F32, t, hc, hc);
            cpu.HcSplitSinkhorn(pre, post, comb, mixes, scale, bias, hc, c.GetProperty("iters").GetInt32(), eps);
            Assert.True(MaxAbsDiff(Named(c, "pre"), ReadF32(pre)) < 1e-6f);
            Assert.True(MaxAbsDiff(Named(c, "post"), ReadF32(post)) < 1e-6f);
            Assert.True(MaxAbsDiff(Named(c, "comb"), ReadF32(comb)) < 1e-6f, $"iters {c.GetProperty("iters")}");
        }
    }

    [Fact]
    public void Pre_And_Post_Mix_Match_The_Upstream_Block_Functions()
    {
        int hc = Fx.GetProperty("hc").GetInt32(), t = Fx.GetProperty("tokens").GetInt32(), d = Fx.GetProperty("dim").GetInt32();
        using Tensor streams = F32(Floats(Fx.GetProperty("streams")), t, hc, d), x = F32(Floats(Fx.GetProperty("x")), t, d);
        using CpuBackend cpu = new();
        foreach (System.Text.Json.JsonElement c in Fx.GetProperty("cases").EnumerateArray())
        {
            using Tensor pre = F32(Named(c, "pre"), t, hc), post = F32(Named(c, "post"), t, hc), comb = F32(Named(c, "comb"), t, hc, hc);
            using Tensor collapsed = Empty(DType.F32, t, d), expanded = Empty(DType.F32, t, hc, d);
            cpu.HcPreMix(collapsed, streams, pre);
            cpu.HcPostMix(expanded, x, streams, post, comb);
            Assert.True(MaxAbsDiff(Named(c, "preMix"), ReadF32(collapsed)) < 1e-5f);
            Assert.True(MaxAbsDiff(Named(c, "postMix"), ReadF32(expanded)) < 1e-5f);
        }
    }

    [Fact]
    public void Invalid_Operands_Throw()
    {
        using Tensor a = F32(new float[24], 1, 24), s = F32(new float[3], 3), b = F32(new float[24], 24);
        using Tensor p = Empty(DType.F32, 1, 4), c = Empty(DType.F32, 1, 4, 4);
        using CpuBackend cpu = new();
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.HcSplitSinkhorn(p, p, c, a, s, b, 4, 0, 1e-6f));
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.HcSplitSinkhorn(p, p, c, a, s, b, 9, 20, 1e-6f));
        using Tensor wrongPost = Empty(DType.F32, 1, 3);
        Assert.Throws<ArgumentException>(() => cpu.HcSplitSinkhorn(p, wrongPost, c, a, s, b, 4, 20, 1e-6f));
    }
}

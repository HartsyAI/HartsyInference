using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed unsafe class LoraBakerTests
{
    [Fact]
    public void Apply_MergesAPeftPairWithAlphaOverRankAndRemovesNothingElse()
    {
        using Tensor w = Filled([2, 3], 1, 2, 3, 4, 5, 6);
        using Tensor a = Filled([1, 3], 1, 0, -1);
        using Tensor b = Filled([2, 1], 2, 3);
        using Tensor bias = Filled([2], 7, 8);
        Dictionary<string, Tensor> weights = new() { ["enc.q.weight"] = w, ["enc.q.bias"] = bias };
        List<LoraBaker.Patch> patches = LoraBaker.Group([new("ad.q.lora_A.weight", a), new("ad.q.lora_B.weight", b)]);
        List<Tensor> owned = [];

        int changed = LoraBaker.Apply(weights, patches, root => "enc" + root[2..], new LoraBaker.Options { Alpha = 2 }, owned);

        // scale = alpha / rank = 2; ΔW = [[2,0,-2],[3,0,-3]].
        Assert.Equal(1, changed);
        Assert.Equal([5f, 2, -1, 10, 5, 0], Values(weights["enc.q.weight"]));
        Assert.Same(bias, weights["enc.q.bias"]);
        Dispose(owned);
    }

    [Fact]
    public void Apply_KeepsTheWeightDtypeAndRoundsBf16ToNearestEven()
    {
        using Tensor w = new(new TensorShape(1, 1), DType.BF16);
        ((ushort*)w.DataPointer)[0] = 0x3F80; // 1.0
        using Tensor a = Filled([1, 1], 1);
        using Tensor b = Filled([1, 1], 1f / 256); // 1 + 2^-8 is a bf16 tie; even is 1.0
        Dictionary<string, Tensor> weights = new() { ["m.weight"] = w };
        List<Tensor> owned = [];

        LoraBaker.Apply(weights, LoraBaker.Group([new("m.lora_down.weight", a), new("m.lora_up.weight", b)]), r => r, new LoraBaker.Options(), owned);

        Assert.Equal(DType.BF16, weights["m.weight"].DType);
        Assert.Equal((ushort)0x3F80, ((ushort*)weights["m.weight"].DataPointer)[0]);
        Dispose(owned);
    }

    [Fact]
    public void MatMulFma_AccumulatesInOrderWithOneRoundingPerStep()
    {
        // 1 + 2^-24 is lost when the product is rounded before the add, and kept by a fused multiply-add.
        float tiny = MathF.Pow(2, -24);
        using Tensor up = Filled([1, 2], 1, tiny);
        using Tensor down = Filled([2, 1], 1, 1 + MathF.Pow(2, -23));
        using Tensor product = LoraBaker.MatMulFma(up, down);
        float expected = MathF.FusedMultiplyAdd(tiny, 1 + MathF.Pow(2, -23), 1);
        Assert.Equal(expected, Values(product)[0]);
    }

    [Fact]
    public void Apply_BakesAConvAdapterAndAKohyaAlphaAndBiasDiff()
    {
        using Tensor w = Filled([2, 1, 1, 2], 0, 0, 0, 0);
        using Tensor bias = Filled([2], 1, 1);
        using Tensor down = Filled([1, 1, 1, 2], 1, 2);
        using Tensor up = Filled([2, 1, 1, 1], 1, -1);
        using Tensor alpha = Filled([], 0.5f);
        using Tensor diffB = Filled([2], 3, 4);
        Dictionary<string, Tensor> weights = new() { ["unet.conv_in.weight"] = w, ["unet.conv_in.bias"] = bias };
        List<LoraBaker.Patch> patches = LoraBaker.Group([
            new("lora_unet_conv_in.lora_down.weight", down), new("lora_unet_conv_in.lora_up.weight", up),
            new("lora_unet_conv_in.alpha", alpha), new("lora_unet_conv_in.diff_b", diffB)]);
        List<Tensor> owned = [];

        LoraBaker.Apply(weights, patches, LoraBaker.AutoTargets(weights.Keys, patches.Select(p => p.Root)), new LoraBaker.Options(), owned);

        Assert.Equal([0.5f, 1, -0.5f, -1], Values(weights["unet.conv_in.weight"]));
        Assert.Equal([4f, 5], Values(weights["unet.conv_in.bias"]));
        Dispose(owned);
    }

    [Theory]
    [InlineData("base_model.model.layers.{0}.attn.q_proj")]
    [InlineData("layers.{0}.attn.q_proj")]
    [InlineData("lora_unet_layers_{0}_attn_q_proj")]
    public void AutoTargets_InfersOnePrefixMappingForTheWholeAdapter(string spelling)
    {
        string[] weights = ["model.layers.0.attn.q_proj.weight", "model.layers.1.attn.q_proj.weight", "model.norm.weight", "vision.proj.weight"];
        Func<string, string?> target = LoraBaker.AutoTargets(weights, [string.Format(spelling, 0), string.Format(spelling, 1)]);
        Assert.Equal("model.layers.0.attn.q_proj", target(string.Format(spelling, 0)));
        Assert.Equal("model.layers.1.attn.q_proj", target(string.Format(spelling, 1)));
        // Same tail, wrong place: not a module under the inferred prefix, so it is unresolved rather than guessed.
        Assert.Null(target(string.Format(spelling, 9)));
    }

    [Fact]
    public void AutoTargets_RefusesAnAmbiguousMatch()
    {
        Assert.Throws<InvalidDataException>(() => LoraBaker.AutoTargets(["text.proj.weight", "vision.proj.weight"], ["proj"]));
    }

    [Fact]
    public void Group_And_Apply_NameWhatIsMissing()
    {
        using Tensor a = Filled([1, 2], 1, 1);
        Assert.Contains("incomplete", Assert.Throws<InvalidDataException>(() => LoraBaker.Group([new("x.lora_A.weight", a)])).Message);

        using Tensor b = Filled([2, 1], 1, 1);
        List<LoraBaker.Patch> patches = LoraBaker.Group([new("x.lora_A.weight", a), new("x.lora_B.weight", b)]);
        Assert.Contains("match no weight", Assert.Throws<InvalidDataException>(() =>
            LoraBaker.Apply(new Dictionary<string, Tensor>(), patches, _ => null, new LoraBaker.Options(), [])).Message);

        using Tensor w = Filled([3, 3], 0, 0, 0, 0, 0, 0, 0, 0, 0);
        Assert.Contains("delta", Assert.Throws<InvalidDataException>(() =>
            LoraBaker.Apply(new Dictionary<string, Tensor> { ["x.weight"] = w }, patches, r => r, new LoraBaker.Options(), [])).Message);
    }

    [Fact]
    public void Apply_RefusesAQuantizedWeight()
    {
        using Tensor w = new(new TensorShape(1, 1), DType.I8);
        using Tensor a = Filled([1, 1], 1);
        using Tensor b = Filled([1, 1], 1);
        Assert.Throws<NotSupportedException>(() => LoraBaker.Apply(new Dictionary<string, Tensor> { ["m.weight"] = w },
            LoraBaker.Group([new("m.lora_A.weight", a), new("m.lora_B.weight", b)]), r => r, new LoraBaker.Options(), []));
    }

    [Fact]
    public void ScaleFor_UsesSqrtRankUnderRsLoraAndFoldsStrength()
    {
        Assert.Equal(2f, LoraBaker.ScaleFor(128, 64, new LoraBaker.Options()));
        Assert.Equal(16f, LoraBaker.ScaleFor(128, 64, new LoraBaker.Options { RsLora = true }));
        Assert.Equal(1f, LoraBaker.ScaleFor(128, 64, new LoraBaker.Options { Strength = 0.5f }));
    }

    private static Tensor Filled(long[] shape, params float[] values)
    {
        Tensor t = new(new TensorShape(shape.Length == 0 ? [1] : shape), DType.F32);
        values.AsSpan().CopyTo(new Span<float>((void*)t.DataPointer, values.Length));
        return t;
    }

    private static float[] Values(Tensor t)
    {
        if (t.DType == DType.F32)
            return new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.Shape.ElementCount).ToArray();
        float[] result = new float[t.Shape.ElementCount];
        fixed (float* dst = result)
            SafeTensorsMerger.Convert((byte*)t.DataPointer, t.DType, (byte*)dst, DType.F32, result.Length);
        return result;
    }

    private static void Dispose(List<Tensor> owned)
    {
        foreach (Tensor t in owned)
            t.Dispose();
    }
}

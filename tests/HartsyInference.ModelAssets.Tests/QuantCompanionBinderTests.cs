using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.Quant;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class QuantCompanionBinderTests
{
    private static QuantBindingSet Bind(QuantTestShard shard, QuantFlavor flavor)
    {
        ShardedSafeTensorSet set = shard.Open();
        return QuantCompanionBinder.Bind(set.Inventory, flavor);
    }

    private static HartsyInferenceException BindFails(QuantTestShard shard, QuantFlavor flavor)
    {
        ShardedSafeTensorSet set = shard.Open();
        return Assert.Throws<HartsyInferenceException>(() => QuantCompanionBinder.Bind(set.Inventory, flavor));
    }

    [Fact]
    public void Official_BindsDenseFp8And32x32AndExpertMxfp4_1x32()
    {
        using QuantTestShard shard = new QuantTestShard()
            .Add("l.wo_a.weight", "F8_E4M3", 256, 128).Add("l.wo_a.scale", "F8_E8M0", 8, 4)
            .Add("l.e.w1.weight", "I8", 64, 32).Add("l.e.w1.scale", "F8_E8M0", 64, 2)
            .Add("l.gate.weight", "BF16", 16, 16)
            .Add("l.engram.weight", "F8_E4M3", 4, 64).Add("l.engram.scale", "F8_E8M0", 4, 2);

        QuantBindingSet bound = Bind(shard, QuantFlavor.Official);

        QuantBinding dense = bound.Bindings["l.wo_a.weight"];
        Assert.Equal(QuantEncoding.Fp8E4M3BlockE8M0, dense.Encoding);
        Assert.Equal(new BlockGeometry(32, 32), dense.Geometry);
        Assert.Equal((256L, 128L), (dense.LogicalRows, dense.LogicalCols));
        QuantBinding expert = bound.Bindings["l.e.w1.weight"];
        Assert.Equal(QuantEncoding.Mxfp4E8M0, expert.Encoding);
        Assert.Equal(new BlockGeometry(1, 32), expert.Geometry);
        Assert.Equal((64L, 64L), (expert.LogicalRows, expert.LogicalCols));
        Assert.Equal(new BlockGeometry(1, 32), bound.Bindings["l.engram.weight"].Geometry);
        Assert.False(bound.Bindings.ContainsKey("l.gate.weight"));
    }

    [Fact]
    public void Official_V3F32ScaleInv_IsRefusedBecauseTheCodecsDecodeE8M0Only()
    {
        using QuantTestShard shard = new QuantTestShard()
            .Add("m.weight", "F8_E4M3", 256, 384).Add("m.weight_scale_inv", "F32", 2, 3);

        HartsyInferenceException ex = BindFails(shard, QuantFlavor.Official);

        Assert.Contains("F8_E8M0 or raw U8", ex.Message);
        Assert.Contains("m.weight_scale_inv", ex.Message);
    }

    [Fact]
    public void Official_V3StyleE8M0ScaleInv_Binds128x128()
    {
        using QuantTestShard shard = new QuantTestShard()
            .Add("m.weight", "F8_E4M3", 256, 384).Add("m.weight_scale_inv", "F8_E8M0", 2, 3);

        QuantBinding binding = Bind(shard, QuantFlavor.Official).Bindings["m.weight"];

        Assert.Equal(new BlockGeometry(128, 128), binding.Geometry);
        Assert.Equal("m.weight_scale_inv", binding.ScaleKey);
    }

    [Fact]
    public void Official_TileEquivalentGeometries_PickTheCanonical32x32()
    {
        // A 32x32 weight with a [1,1] scale fits 32x32 and 128x128, which tile it identically.
        using QuantTestShard shard = new QuantTestShard().Add("m.weight", "F8_E4M3", 32, 32).Add("m.scale", "F8_E8M0", 1, 1);

        Assert.Equal(new BlockGeometry(32, 32), Bind(shard, QuantFlavor.Official).Bindings["m.weight"].Geometry);
    }

    [Fact]
    public void Official_SingleRowWeight_PicksTheCanonicalGeometry()
    {
        using QuantTestShard shard = new QuantTestShard().Add("m.weight", "F8_E4M3", 1, 64).Add("m.scale", "F8_E8M0", 1, 2);

        Assert.Equal(new BlockGeometry(32, 32), Bind(shard, QuantFlavor.Official).Bindings["m.weight"].Geometry);
    }

    [Fact]
    public void Official_NormScaleWithoutASiblingWeight_IsNotAnOrphan()
    {
        using QuantTestShard shard = new QuantTestShard().Add("ls.scale", "F32", 8).Add("m.weight", "BF16", 4, 4);

        Assert.Empty(Bind(shard, QuantFlavor.Official).Bindings);
    }

    [Fact]
    public void Nvfp4_PerTensorFp8WithGlobalScale_ClaimsItsCompanions()
    {
        using QuantTestShard shard = new QuantTestShard().Add("m.weight", "F8_E4M3", 16, 16)
            .Add("m.weight_scale", "F32", 1).Add("m.weight_scale_2", "F32", 1);

        Assert.Equal(["m.weight"], Bind(shard, QuantFlavor.NvidiaNvfp4).PerTensorFp8);
    }

    [Fact]
    public void Official_NoMatchingGeometry_Throws()
    {
        using QuantTestShard shard = new QuantTestShard().Add("m.weight", "F8_E4M3", 64, 64).Add("m.scale", "F8_E8M0", 3, 5);

        Assert.Contains("matches no geometry", BindFails(shard, QuantFlavor.Official).Message);
    }

    [Fact]
    public void Official_WeightWithoutScale_Throws()
    {
        using QuantTestShard shard = new QuantTestShard().Add("m.weight", "F8_E4M3", 32, 32).Add("k.weight", "I8", 4, 16);

        HartsyInferenceException ex = BindFails(shard, QuantFlavor.Official);

        Assert.Contains("'m.weight'", ex.Message);
        Assert.Contains("'k.weight'", ex.Message);
    }

    [Fact]
    public void Official_ScaleBesideAnUnquantizedWeight_Throws()
    {
        using QuantTestShard shard = new QuantTestShard().Add("n.weight", "BF16", 4, 4).Add("n.scale", "F8_E8M0", 1, 1);

        HartsyInferenceException ex = BindFails(shard, QuantFlavor.Official);

        Assert.Contains("'n.scale'", ex.Message);
    }

    [Fact]
    public void Official_WeightScaleInvWithoutWeight_Throws()
    {
        using QuantTestShard shard = new QuantTestShard().Add("gone.weight_scale_inv", "F8_E8M0", 2, 2);

        Assert.Contains("'gone.weight_scale_inv'", BindFails(shard, QuantFlavor.Official).Message);
    }

    [Fact]
    public void Official_BothScaleNames_IsAmbiguous()
    {
        using QuantTestShard shard = new QuantTestShard().Add("m.weight", "F8_E4M3", 32, 32)
            .Add("m.scale", "F8_E8M0", 1, 1).Add("m.weight_scale_inv", "F32", 1, 1);

        Assert.Contains("2 scale companions", BindFails(shard, QuantFlavor.Official).Message);
    }

    [Fact]
    public void Official_UnsupportedScaleDType_Throws()
    {
        using QuantTestShard shard = new QuantTestShard().Add("m.weight", "F8_E4M3", 32, 64).Add("m.scale", "BF16", 1, 2);

        Assert.Contains("decode only F8_E8M0 or raw U8", BindFails(shard, QuantFlavor.Official).Message);
    }

    [Fact]
    public void Official_ReportsEveryProblemInOneException()
    {
        using QuantTestShard shard = new QuantTestShard().Add("a.weight", "F8_E4M3", 32, 32).Add("b.weight", "I8", 8, 16)
            .Add("c.weight_scale_inv", "F8_E8M0", 1, 1);

        HartsyInferenceException ex = BindFails(shard, QuantFlavor.Official);

        Assert.Contains("3 problems", ex.Message);
    }

    [Fact]
    public void Quark_RequiresU8Scales()
    {
        using QuantTestShard bad = new QuantTestShard().Add("e.weight", "U8", 8, 32).Add("e.weight_scale", "F8_E8M0", 8, 2);
        Assert.Contains("raw U8", BindFails(bad, QuantFlavor.AmdQuark).Message);

        using QuantTestShard good = new QuantTestShard().Add("e.weight", "U8", 8, 32).Add("e.weight_scale", "U8", 8, 2);
        QuantBinding binding = Bind(good, QuantFlavor.AmdQuark).Bindings["e.weight"];
        Assert.Equal(QuantEncoding.Mxfp4E8M0, binding.Encoding);
        Assert.Equal(DType.U8, binding.ScaleDType);
    }

    [Fact]
    public void Nvfp4_BindsPackedWeightWithGlobalAndInputScale()
    {
        using QuantTestShard shard = new QuantTestShard()
            .Add("e.weight", "U8", 8, 32).Add("e.weight_scale", "F8_E4M3", 8, 4)
            .Add("e.weight_scale_2", "F32", 1).Add("e.input_scale", "F32", 1)
            .Add("d.weight", "F8_E4M3", 16, 16).Add("d.weight_scale", "F32", 1).Add("d.input_scale", "F32", 1);

        QuantBindingSet bound = Bind(shard, QuantFlavor.NvidiaNvfp4);

        QuantBinding binding = bound.Bindings["e.weight"];
        Assert.Equal(QuantEncoding.Nvfp4, binding.Encoding);
        Assert.Equal(new BlockGeometry(1, 16), binding.Geometry);
        Assert.Equal((8L, 64L), (binding.LogicalRows, binding.LogicalCols));
        Assert.Equal("e.weight_scale_2", binding.GlobalScaleKey);
        Assert.Equal("e.input_scale", binding.InputScaleKey);
        Assert.Equal(["d.weight"], bound.PerTensorFp8);
    }

    [Fact]
    public void Nvfp4_ExpertWithoutGlobalScale_Throws()
    {
        using QuantTestShard shard = new QuantTestShard().Add("e.weight", "U8", 8, 32).Add("e.weight_scale", "F8_E4M3", 8, 4);

        Assert.Contains("e.weight_scale_2", BindFails(shard, QuantFlavor.NvidiaNvfp4).Message);
    }

    [Fact]
    public void Mlx_BindsGroup64AndChecksPackedWidth()
    {
        using QuantTestShard good = new QuantTestShard().Add("m.weight", "U32", 8, 16)
            .Add("m.scales", "F32", 8, 2).Add("m.biases", "F32", 8, 2);
        QuantBinding binding = Bind(good, QuantFlavor.Mlx).Bindings["m.weight"];
        Assert.Equal(QuantEncoding.AffineInt4, binding.Encoding);
        Assert.Equal((8L, 128L), (binding.LogicalRows, binding.LogicalCols));
        Assert.Equal("m.biases", binding.BiasKey);

        // Group size 32: [8,4] scales over in=128.
        using QuantTestShard gs32 = new QuantTestShard().Add("m.weight", "U32", 8, 16)
            .Add("m.scales", "F32", 8, 4).Add("m.biases", "F32", 8, 4);
        Assert.Contains("group size 64", BindFails(gs32, QuantFlavor.Mlx).Message);
    }

    [Fact]
    public void Mlx_MissingBiases_Throws()
    {
        using QuantTestShard shard = new QuantTestShard().Add("m.weight", "U32", 8, 16).Add("m.scales", "F32", 8, 2);

        Assert.Contains("m.biases", BindFails(shard, QuantFlavor.Mlx).Message);
    }

    // Shapes below are the real derivative headers divided by 32 along each axis, keeping every dtype, key and ratio.
    [Fact]
    public void Nvfp4_HybridCheckpoint_BindsOfficialLayoutPartsAlongsideNvfp4Experts()
    {
        using QuantTestShard shard = new QuantTestShard()
            .Add("layers.0.attn.wq_a.weight", "F8_E4M3", 40, 160).Add("layers.0.attn.wq_a.scale", "F8_E8M0", 2, 5)
            .Add("layers.0.ffn.shared_experts.w1.weight", "F8_E4M3", 72, 160).Add("layers.0.ffn.shared_experts.w1.scale", "F8_E8M0", 3, 5)
            .Add("layers.0.ffn.experts.0.w1.weight", "U8", 72, 80).Add("layers.0.ffn.experts.0.w1.weight_scale", "F8_E4M3", 72, 10)
            .Add("layers.0.ffn.experts.0.w1.weight_scale_2", "F32").Add("layers.0.ffn.experts.0.w1.input_scale", "F32")
            .Add("layers.0.ffn.gate.weight", "BF16", 12, 160)
            .Add("mtp.0.ffn.experts.0.w1.weight", "I8", 72, 80).Add("mtp.0.ffn.experts.0.w1.scale", "F8_E8M0", 72, 5);

        QuantBindingSet bound = Bind(shard, QuantFlavor.NvidiaNvfp4);

        Assert.Equal(QuantEncoding.Fp8E4M3BlockE8M0, bound.Bindings["layers.0.attn.wq_a.weight"].Encoding);
        Assert.Equal(new BlockGeometry(32, 32), bound.Bindings["layers.0.attn.wq_a.weight"].Geometry);
        Assert.Equal(new BlockGeometry(32, 32), bound.Bindings["layers.0.ffn.shared_experts.w1.weight"].Geometry);
        QuantBinding expert = bound.Bindings["layers.0.ffn.experts.0.w1.weight"];
        Assert.Equal(QuantEncoding.Nvfp4, expert.Encoding);
        Assert.Equal(new BlockGeometry(1, 16), expert.Geometry);
        Assert.Equal((72L, 160L), (expert.LogicalRows, expert.LogicalCols));
        Assert.Equal("layers.0.ffn.experts.0.w1.weight_scale_2", expert.GlobalScaleKey);
        Assert.Equal(QuantEncoding.Mxfp4E8M0, bound.Bindings["mtp.0.ffn.experts.0.w1.weight"].Encoding);
        Assert.Empty(bound.PerTensorFp8);
    }

    [Fact]
    public void Nvfp4_ToWeightInfo_CarriesTheGlobalAndInputScaleTensors()
    {
        using QuantTestShard shard = new QuantTestShard()
            .Add("e.weight", "U8", 8, 32).Add("e.weight_scale", "F8_E4M3", 8, 4)
            .Add("e.weight_scale_2", "F32").Add("e.input_scale", "F32");
        ShardedSafeTensorSet set = shard.Open();

        QuantWeightInfo info = QuantCompanionBinder.Bind(set.Inventory, QuantFlavor.NvidiaNvfp4).Bindings["e.weight"].ToWeightInfo(set.GetTensor);

        Assert.Equal("recipe-nvfp4", info.Format);
        Assert.Equal(DType.F8E4M3, info.Recipe!.ScaleDType);
        Assert.Equal(DType.F32, info.Recipe.GlobalScale!.DType);
        Assert.NotNull(info.Recipe.InputScale);
    }

    [Fact]
    public void Nvfp4_ExpertWithMalformedGlobalScale_OrWrongScaleType_Throws()
    {
        using QuantTestShard wrongScale = new QuantTestShard().Add("e.weight", "U8", 8, 32).Add("e.weight_scale", "F32", 8, 4)
            .Add("e.weight_scale_2", "F32");
        Assert.Contains("F8_E4M3", BindFails(wrongScale, QuantFlavor.NvidiaNvfp4).Message);

        using QuantTestShard wrongGeometry = new QuantTestShard().Add("e.weight", "U8", 8, 32).Add("e.weight_scale", "F8_E4M3", 8, 2)
            .Add("e.weight_scale_2", "F32");
        Assert.Contains("e.weight", BindFails(wrongGeometry, QuantFlavor.NvidiaNvfp4).Message);
    }

    [Fact]
    public void Quark_RealLayout_BindsFp4ExpertsAndSharedExpertsAndFp8AttentionWithE8m0Scales()
    {
        using QuantTestShard shard = new QuantTestShard()
            .Add("layers.0.attn.wq_a.weight", "F8_E4M3", 40, 160).Add("layers.0.attn.wq_a.weight_scale", "F8_E8M0", 2, 5)
            .Add("layers.0.ffn.experts.0.w1.weight", "U8", 72, 80).Add("layers.0.ffn.experts.0.w1.weight_scale", "U8", 72, 5)
            .Add("layers.0.ffn.shared_experts.w1.weight", "U8", 72, 80).Add("layers.0.ffn.shared_experts.w1.weight_scale", "U8", 72, 5)
            .Add("layers.1.engram.wkv.weight", "F8_E4M3", 800, 192).Add("layers.1.engram.wkv.weight_scale", "F8_E8M0", 25, 6);

        QuantBindingSet bound = Bind(shard, QuantFlavor.AmdQuark);

        QuantBinding attention = bound.Bindings["layers.0.attn.wq_a.weight"];
        Assert.Equal(QuantEncoding.Fp8E4M3BlockE8M0, attention.Encoding);
        Assert.Equal(DType.F8E8M0, attention.ScaleDType);
        Assert.Equal(QuantEncoding.Mxfp4E8M0, bound.Bindings["layers.0.ffn.shared_experts.w1.weight"].Encoding);
        Assert.Equal(new BlockGeometry(1, 32), bound.Bindings["layers.0.ffn.experts.0.w1.weight"].Geometry);
        QuantBinding w1 = bound.Bindings["layers.0.ffn.experts.0.w1.weight"];
        Assert.Equal((72L, 160L), (w1.LogicalRows, w1.LogicalCols));
        Assert.Equal(QuantEncoding.Fp8E4M3BlockE8M0, bound.Bindings["layers.1.engram.wkv.weight"].Encoding);
    }

    [Fact]
    public void Quark_Fp4WeightWithBf16OrF8Scale_OrMissingScale_Throws()
    {
        using QuantTestShard bf16 = new QuantTestShard().Add("e.weight", "U8", 8, 32).Add("e.weight_scale", "BF16", 8, 2);
        Assert.Contains("e.weight_scale", BindFails(bf16, QuantFlavor.AmdQuark).Message);

        using QuantTestShard missing = new QuantTestShard().Add("e.weight", "U8", 8, 32);
        Assert.Contains("e.weight", BindFails(missing, QuantFlavor.AmdQuark).Message);
    }

    [Fact]
    public void Mlx_MixedPrecision_InfersFourAndEightBitFromTheScaleShape()
    {
        using QuantTestShard shard = new QuantTestShard()
            .Add("layers.4.ffn.experts.3.w2.weight", "U32", 160, 16).Add("layers.4.ffn.experts.3.w2.scales", "F32", 160, 2)
                .Add("layers.4.ffn.experts.3.w2.biases", "F32", 160, 2)
            .Add("layers.5.attn.wkv.weight", "U32", 16, 32).Add("layers.5.attn.wkv.scales", "F32", 16, 2)
                .Add("layers.5.attn.wkv.biases", "F32", 16, 2)
            .Add("layers.1.engram.wkv.weight", "U32", 800, 48).Add("layers.1.engram.wkv.scales", "F32", 800, 3)
                .Add("layers.1.engram.wkv.biases", "F32", 800, 3)
            .Add("layers.14.engram.embed.weight", "U32", 40, 32).Add("layers.14.engram.embed.scales", "F32", 40, 4)
                .Add("layers.14.engram.embed.biases", "F32", 40, 4)
            .Add("layers.4.ffn.gate.weight", "BF16", 12, 160).Add("layers.4.ffn.gate.bias", "F32", 12);

        QuantBindingSet bound = Bind(shard, QuantFlavor.Mlx);

        QuantBinding fourBit = bound.Bindings["layers.14.engram.embed.weight"];
        Assert.Equal(QuantEncoding.AffineInt4, fourBit.Encoding);
        Assert.Equal((40L, 256L), (fourBit.LogicalRows, fourBit.LogicalCols));
        QuantBinding eightBit = bound.Bindings["layers.5.attn.wkv.weight"];
        Assert.Equal(QuantEncoding.AffineInt8, eightBit.Encoding);
        Assert.Equal((16L, 128L), (eightBit.LogicalRows, eightBit.LogicalCols));
        Assert.Equal(QuantEncoding.AffineInt8, bound.Bindings["layers.1.engram.wkv.weight"].Encoding);
        Assert.False(bound.Bindings.ContainsKey("layers.4.ffn.gate.weight"));
    }

    [Fact]
    public void Mlx_Bf16ScalesAndScaleShapeMismatches_Throw()
    {
        using QuantTestShard bf16 = new QuantTestShard().Add("m.weight", "U32", 8, 16).Add("m.scales", "BF16", 8, 2).Add("m.biases", "BF16", 8, 2);
        Assert.Contains("F32", BindFails(bf16, QuantFlavor.Mlx).Message);

        using QuantTestShard mixedDtype = new QuantTestShard().Add("m.weight", "U32", 8, 16).Add("m.scales", "F32", 8, 2)
            .Add("m.biases", "F16", 8, 2);
        Assert.Contains("one shape and dtype", BindFails(mixedDtype, QuantFlavor.Mlx).Message);

        using QuantTestShard rowMismatch = new QuantTestShard().Add("m.weight", "U32", 8, 16).Add("m.scales", "F32", 4, 2)
            .Add("m.biases", "F32", 4, 2);
        Assert.Contains("m.weight", BindFails(rowMismatch, QuantFlavor.Mlx).Message);

        using QuantTestShard oddGroups = new QuantTestShard().Add("m.weight", "U32", 8, 16).Add("m.scales", "F32", 8, 3).Add("m.biases", "F32", 8, 3);
        Assert.Contains("group size 64", BindFails(oddGroups, QuantFlavor.Mlx).Message);
    }

    [Fact]
    public void Mlx_ToWeightInfo_CarriesScaleAndBiasUnderARecipeFormat()
    {
        using QuantTestShard shard = new QuantTestShard()
            .Add("a.weight", "U32", 8, 16).Add("a.scales", "F32", 8, 2).Add("a.biases", "F32", 8, 2)
            .Add("b.weight", "U32", 8, 32).Add("b.scales", "F32", 8, 2).Add("b.biases", "F32", 8, 2);
        ShardedSafeTensorSet set = shard.Open();
        QuantBindingSet bound = QuantCompanionBinder.Bind(set.Inventory, QuantFlavor.Mlx);

        QuantWeightInfo four = bound.Bindings["a.weight"].ToWeightInfo(set.GetTensor);
        QuantWeightInfo eight = bound.Bindings["b.weight"].ToWeightInfo(set.GetTensor);

        Assert.Equal("recipe-affine-int4", four.Format);
        Assert.Equal("recipe-affine-int8", eight.Format);
        Assert.Equal(new TensorShape(8, 2), four.Recipe!.Bias!.Shape);
        Assert.Equal(DType.F32, four.Recipe.Scale!.DType);
        Assert.Equal(128L, eight.Recipe!.LogicalCols);
    }

    // Shapes below are read from the real sfxnz/DeepSeek-V4.1-Flash-EXL3 headers (branch 2.0bpw-mcg-viterbi-lmhead-mxfp8 @ 982b7045).
    private static QuantTestShard RealExpert(string name = "layers.3.ffn.experts.0.w1", int inTiles = 320, int outTiles = 144) =>
        new QuantTestShard().Add(name + ".trellis", "I16", inTiles, outTiles, 32).Add(name + ".suh", "F16", inTiles * 16)
            .Add(name + ".svh", "F16", outTiles * 16).Add(name + ".mcg", "I32");

    [Fact]
    public void Exl3_BindsARealExpertShapeRowsAreOutputColumnsAreInput()
    {
        using QuantTestShard shard = RealExpert();

        QuantBinding binding = Bind(shard, QuantFlavor.Exl3).Bindings["layers.3.ffn.experts.0.w1.trellis"];

        Assert.Equal(QuantEncoding.Exl3Trellis, binding.Encoding);
        Assert.Equal((2304L, 5120L), (binding.LogicalRows, binding.LogicalCols));
        Assert.Equal(2, binding.Exl3!.Bits);
        Assert.Equal("layers.3.ffn.experts.0.w1.suh", binding.Exl3.Suh);
        Assert.Equal("layers.3.ffn.experts.0.w1.mcg", binding.Exl3.Mcg);
    }

    [Fact]
    public void Exl3_BindsW2WithSwappedGeometry()
    {
        using QuantTestShard shard = RealExpert("layers.3.ffn.experts.0.w2", inTiles: 144, outTiles: 320);

        QuantBinding binding = Bind(shard, QuantFlavor.Exl3).Bindings["layers.3.ffn.experts.0.w2.trellis"];

        Assert.Equal((5120L, 2304L), (binding.LogicalRows, binding.LogicalCols));
    }

    [Theory]
    [InlineData(".mcg")]
    [InlineData(".suh")]
    [InlineData(".svh")]
    public void Exl3_MissingCompanion_NamesIt(string suffix)
    {
        QuantTestShard shard = new();
        foreach ((string name, string dtype, long[] dims) in new (string, string, long[])[]
                 {
                     ("x.trellis", "I16", [8, 8, 32]), ("x.suh", "F16", [128]), ("x.svh", "F16", [128]), ("x.mcg", "I32", []),
                 })
            if (name != "x" + suffix) shard.Add(name, dtype, dims);
        using QuantTestShard built = shard;

        Assert.Contains("x" + suffix, BindFails(built, QuantFlavor.Exl3).Message);
    }

    [Fact]
    public void Exl3_OtherThanTwoBits_ThrowsNamingKeyAndShape()
    {
        using QuantTestShard shard = new QuantTestShard().Add("x.trellis", "I16", 8, 8, 48).Add("x.suh", "F16", 128)
            .Add("x.svh", "F16", 128).Add("x.mcg", "I32");

        string message = BindFails(shard, QuantFlavor.Exl3).Message;

        Assert.Contains("'x.trellis'", message);
        Assert.Contains("3 bits", message);
        Assert.Contains("[8, 8, 48]", message);
    }

    [Fact]
    public void Exl3_GeometryMismatches_AreEachReported()
    {
        // suh must equal in (tiles*16), svh must equal out, and both dims must be Hadamard blocks.
        using QuantTestShard shard = new QuantTestShard()
            .Add("a.trellis", "I16", 8, 8, 32).Add("a.suh", "F16", 64).Add("a.svh", "F16", 128).Add("a.mcg", "I32")
            .Add("b.trellis", "I16", 8, 8, 32).Add("b.suh", "F16", 128).Add("b.svh", "F16", 256).Add("b.mcg", "I32")
            .Add("c.trellis", "I16", 9, 8, 32).Add("c.suh", "F16", 144).Add("c.svh", "F16", 128).Add("c.mcg", "I32");

        string message = BindFails(shard, QuantFlavor.Exl3).Message;

        Assert.Contains("'a.suh'", message);
        Assert.Contains("F16 [128] (in)", message);
        Assert.Contains("'b.svh'", message);
        Assert.Contains("F16 [128] (out)", message);
        Assert.Contains("'c.trellis'", message);
        Assert.Contains("multiples of 128", message);
        Assert.Contains("3 problems", message);
    }

    [Theory]
    [InlineData("I32", "F16", "F16", "I32")]
    [InlineData("I16", "F32", "F16", "I32")]
    [InlineData("I16", "F16", "BF16", "I32")]
    [InlineData("I16", "F16", "F16", "F32")]
    public void Exl3_WrongDTypes_AreRefused(string trellis, string suh, string svh, string mcg)
    {
        using QuantTestShard shard = new QuantTestShard().Add("x.trellis", trellis, 8, 8, 32).Add("x.suh", suh, 128)
            .Add("x.svh", svh, 128).Add("x.mcg", mcg);

        Assert.Contains("EXL3", BindFails(shard, QuantFlavor.Exl3).Message);
    }

    [Fact]
    public void Exl3_RankTwoTrellis_IsRefused()
    {
        using QuantTestShard shard = new QuantTestShard().Add("x.trellis", "I16", 8, 32).Add("x.suh", "F16", 128)
            .Add("x.svh", "F16", 128).Add("x.mcg", "I32");

        Assert.Contains("[in/16, out/16, 16*bits]", BindFails(shard, QuantFlavor.Exl3).Message);
    }

    [Fact]
    public void Exl3_BindsTheNonRoutedFp8AndMxfp8LmHeadOfTheRealLayout()
    {
        // The real matrices are [2304,5120] with 32x32 scales [72,160] and lm_head [129280,5120] with 1x32 scales [129280,160] (U8);
        // the same scale patterns are used at a size that fits a zero-filled test file.
        using QuantTestShard shard = new QuantTestShard()
            .Add("layers.0.attn.wo_a.weight", "F8_E4M3", 256, 512).Add("layers.0.attn.wo_a.scale", "F8_E8M0", 8, 16)
            .Add("lm_head.weight", "F8_E4M3", 256, 512).Add("lm_head.weight_scale", "U8", 256, 16)
            .Add("layers.0.attn_norm.weight", "BF16", 5120)
            .Add("layers.3.ffn.experts.0.w1.trellis", "I16", 320, 144, 32).Add("layers.3.ffn.experts.0.w1.suh", "F16", 5120)
            .Add("layers.3.ffn.experts.0.w1.svh", "F16", 2304).Add("layers.3.ffn.experts.0.w1.mcg", "I32");

        QuantBindingSet bound = Bind(shard, QuantFlavor.Exl3);

        Assert.Equal(new BlockGeometry(32, 32), bound.Bindings["layers.0.attn.wo_a.weight"].Geometry);
        QuantBinding head = bound.Bindings["lm_head.weight"];
        Assert.Equal(QuantEncoding.Fp8E4M3BlockE8M0, head.Encoding);
        Assert.Equal(new BlockGeometry(1, 32), head.Geometry);
        Assert.Equal("lm_head.weight_scale", head.ScaleKey);
        Assert.Equal(3, bound.Bindings.Count);
    }

    [Fact]
    public void Exl3_Fp8WeightWithoutAScale_IsRefused()
    {
        using QuantTestShard shard = new QuantTestShard().Add("lm_head.weight", "F8_E4M3", 128, 64);

        Assert.Contains("'lm_head.weight'", BindFails(shard, QuantFlavor.Exl3).Message);
    }

    [Fact]
    public void Exl3_OrphanSuhOrScale_IsRefused()
    {
        using QuantTestShard shard = new QuantTestShard().Add("y.suh", "F16", 128).Add("z.weight_scale", "U8", 4, 4);

        string message = BindFails(shard, QuantFlavor.Exl3).Message;

        Assert.Contains("'y.suh'", message);
        Assert.Contains("'z.weight_scale'", message);
    }

    [Fact]
    public void ToWeightInfo_MaterializesBorrowedTensorsUnderARecipeFormat()
    {
        using QuantTestShard shard = new QuantTestShard().Add("w.weight", "F8_E4M3", 64, 64).Add("w.scale", "F8_E8M0", 2, 2);
        ShardedSafeTensorSet set = shard.Open();

        QuantWeightInfo info = QuantCompanionBinder.Bind(set.Inventory, QuantFlavor.Official).Bindings["w.weight"].ToWeightInfo(set.GetTensor);

        Assert.Equal("recipe-fp8-block-e8m0", info.Format);
        Assert.Equal(new TensorShape(2, 2), info.Recipe!.Scale!.Shape);
        Assert.Equal(DType.F8E8M0, info.Recipe.Scale.DType);
    }
}

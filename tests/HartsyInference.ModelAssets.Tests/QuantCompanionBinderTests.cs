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
    public void Official_V3WeightScaleInv_Binds128x128()
    {
        using QuantTestShard shard = new QuantTestShard()
            .Add("m.weight", "F8_E4M3", 256, 384).Add("m.weight_scale_inv", "F32", 2, 3);

        QuantBinding binding = Bind(shard, QuantFlavor.Official).Bindings["m.weight"];

        Assert.Equal(new BlockGeometry(128, 128), binding.Geometry);
        Assert.Equal(DType.F32, binding.ScaleDType);
        Assert.Equal("m.weight_scale_inv", binding.ScaleKey);
    }

    [Fact]
    public void Official_AmbiguousGeometry_Throws()
    {
        // A 32x32 weight with a [1,1] scale fits both the 32x32 and 128x128 geometries.
        using QuantTestShard shard = new QuantTestShard().Add("m.weight", "F8_E4M3", 32, 32).Add("m.scale", "F8_E8M0", 1, 1);

        HartsyInferenceException ex = BindFails(shard, QuantFlavor.Official);

        Assert.Contains("ambiguous", ex.Message);
        Assert.Contains("m.scale", ex.Message);
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
    public void Official_ScaleWithoutWeight_Throws()
    {
        using QuantTestShard shard = new QuantTestShard().Add("gone.scale", "F8_E8M0", 2, 2).Add("n.weight", "BF16", 4, 4).Add("n.scale", "F8_E8M0", 1, 1);

        HartsyInferenceException ex = BindFails(shard, QuantFlavor.Official);

        Assert.Contains("'gone.scale'", ex.Message);
        Assert.Contains("'n.scale'", ex.Message);
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

        Assert.Contains("scale dtype must be one of", BindFails(shard, QuantFlavor.Official).Message);
    }

    [Fact]
    public void Official_ReportsEveryProblemInOneException()
    {
        using QuantTestShard shard = new QuantTestShard().Add("a.weight", "F8_E4M3", 32, 32).Add("b.weight", "I8", 8, 16)
            .Add("c.scale", "F8_E8M0", 1, 1);

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
            .Add("m.scales", "BF16", 8, 2).Add("m.biases", "BF16", 8, 2);
        QuantBinding binding = Bind(good, QuantFlavor.Mlx).Bindings["m.weight"];
        Assert.Equal(QuantEncoding.AffineInt4, binding.Encoding);
        Assert.Equal((8L, 128L), (binding.LogicalRows, binding.LogicalCols));
        Assert.Equal("m.biases", binding.BiasKey);

        // Group size 32: [8,4] scales over in=128.
        using QuantTestShard gs32 = new QuantTestShard().Add("m.weight", "U32", 8, 16)
            .Add("m.scales", "BF16", 8, 4).Add("m.biases", "BF16", 8, 4);
        Assert.Contains("group size 64", BindFails(gs32, QuantFlavor.Mlx).Message);
    }

    [Fact]
    public void Mlx_MissingBiases_Throws()
    {
        using QuantTestShard shard = new QuantTestShard().Add("m.weight", "U32", 8, 16).Add("m.scales", "BF16", 8, 2);

        Assert.Contains("m.biases", BindFails(shard, QuantFlavor.Mlx).Message);
    }

    [Fact]
    public void Exl3_RequiresMcg()
    {
        using QuantTestShard bad = new QuantTestShard().Add("x.trellis", "I32", 4, 8, 32).Add("x.suh", "F16", 64).Add("x.svh", "F16", 128);
        Assert.Contains("x.mcg", BindFails(bad, QuantFlavor.Exl3).Message);

        using QuantTestShard good = new QuantTestShard().Add("x.trellis", "I32", 4, 8, 32)
            .Add("x.suh", "F16", 64).Add("x.svh", "F16", 128).Add("x.mcg", "U32", 1);
        QuantBinding binding = Bind(good, QuantFlavor.Exl3).Bindings["x.trellis"];
        Assert.Equal(2, binding.Exl3!.Bits);
        Assert.Equal((128L, 64L), (binding.LogicalRows, binding.LogicalCols));
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

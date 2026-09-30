using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Quant;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Producer tensor names to the canonical (official) names, using names copied from the real headers.</summary>
public sealed class HfKeyMapperTests
{
    [Theory]
    [InlineData("layers.3.ffn.experts.17.w1.weight")]
    [InlineData("layers.3.ffn.experts.17.w1.scale")]
    [InlineData("mtp.2.ffn.experts.9.w2.weight")]
    [InlineData("embed.weight")]
    [InlineData("head.weight")]
    [InlineData("layers.14.engram.embed.weight")]
    public void OfficialAndMlx_KeepNamesAsTheyAre(string key)
    {
        Assert.Equal(key, new OfficialV41KeyMapper().MapToCanonical(key));
        Assert.Equal(key, new MlxV41KeyMapper().MapToCanonical(key));
    }

    [Fact]
    public void OfficialAndMlx_StripNothing()
    {
        Assert.Empty(new OfficialV41KeyMapper().StrippedComponents);
        Assert.Empty(new MlxV41KeyMapper().StrippedComponents);
        Assert.Empty(new Exl3V41KeyMapper().StrippedComponents);
    }

    [Theory]
    [InlineData("lm_head.weight", "head.weight")]
    [InlineData("lm_head.scale", "head.scale")]
    [InlineData("layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.weight")]
    [InlineData("head.weight", "head.weight")]
    public void Exl3_RenamesOnlyTheLmHead(string key, string expected) =>
        Assert.Equal(expected, new Exl3V41KeyMapper().MapToCanonical(key));

    [Fact]
    public void ForSafeTensors_PicksTheMapperByFlavor()
    {
        Assert.IsType<MlxV41KeyMapper>(HfKeyMappers.ForSafeTensors(QuantFlavor.Mlx));
        Assert.IsType<Exl3V41KeyMapper>(HfKeyMappers.ForSafeTensors(QuantFlavor.Exl3));
        Assert.IsType<OfficialV41KeyMapper>(HfKeyMappers.ForSafeTensors(QuantFlavor.Official));
        Assert.IsType<OfficialV41KeyMapper>(HfKeyMappers.ForSafeTensors(QuantFlavor.NvidiaNvfp4));
        Assert.IsType<OfficialV41KeyMapper>(HfKeyMappers.ForSafeTensors(QuantFlavor.AmdQuark));
        Assert.IsType<OfficialV41KeyMapper>(HfKeyMappers.ForSafeTensors(null));
    }

    [Theory]
    [InlineData("token_embd.weight", "embed.weight")]
    [InlineData("output_norm.weight", "norm.weight")]
    [InlineData("output.weight", "head.weight")]
    [InlineData("blk.0.attn_q_a.weight", "layers.0.attn.wq_a.weight")]
    [InlineData("blk.39.attn_output_b.weight", "layers.39.attn.wo_b.weight")]
    [InlineData("blk.2.attn_compressor_kv.weight", "layers.2.attn.compressor.wkv.weight")]
    [InlineData("blk.2.indexer.proj.weight", "layers.2.attn.indexer.weights_proj.weight")]
    [InlineData("blk.5.attn_sinks.weight", "layers.5.attn.attn_sink")]
    [InlineData("blk.5.hc_ffn_base.weight", "layers.5.hc_ffn_base")]
    [InlineData("blk.7.ffn_gate_inp.weight", "layers.7.ffn.gate.weight")]
    [InlineData("blk.7.exp_probs_b.bias", "layers.7.ffn.gate.bias")]
    [InlineData("blk.7.ffn_gate_shexp.weight", "layers.7.ffn.shared_experts.w1.weight")]
    [InlineData("blk.7.ffn_down_exps.weight", "layers.7.ffn.experts.w2.weight")]
    [InlineData("blk.7.ffn_up_exps.weight", "layers.7.ffn.experts.w3.weight")]
    [InlineData("blk.1.engram_embd.weight", "layers.1.engram.embed.weight")]
    [InlineData("blk.14.engram_kv.weight", "layers.14.engram.wkv.weight")]
    [InlineData("blk.14.engram_q_norm.weight", "layers.14.engram.q_weight")]
    public void DwarfStar_MapsGgufNamesToCanonical(string key, string expected) =>
        Assert.Equal(expected, new DwarfStarV41KeyMapper().MapToCanonical(key));

    [Theory]
    [InlineData("blk.40.attn_q_a.weight")]
    [InlineData("blk.42.ffn_gate_exps.weight")]
    [InlineData("mtp.0.attn.wq_a.weight")]
    [InlineData("blk.3.no_such_tensor.weight")]
    [InlineData("blk.x.attn_q_a.weight")]
    [InlineData("blk.3")]
    [InlineData("something.else")]
    public void DwarfStar_MapsDraftAndUnknownNamesToNull(string key) =>
        Assert.Null(new DwarfStarV41KeyMapper().MapToCanonical(key));

    [Fact]
    public void DwarfStar_ReportsTheDraftAsStrippedAndHonoursTheBackboneDepth()
    {
        DwarfStarV41KeyMapper mapper = new(backboneLayers: 4);

        Assert.Equal(["mtp"], mapper.StrippedComponents);
        Assert.Equal("layers.3.attn_norm.weight", mapper.MapToCanonical("blk.3.attn_norm.weight"));
        Assert.Null(mapper.MapToCanonical("blk.4.attn_norm.weight"));
    }

    [Fact]
    public void DwarfStar_RejectsANonPositiveDepthAndNullKeys()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DwarfStarV41KeyMapper(0));
        Assert.Throws<ArgumentNullException>(() => new DwarfStarV41KeyMapper().MapToCanonical(null!));
        Assert.Throws<ArgumentNullException>(() => new OfficialV41KeyMapper().MapToCanonical(null!));
    }

    [Fact]
    public void DwarfStar_EngramRowLayoutConstantsAddUp()
    {
        Assert.Equal(264, DwarfStarV41KeyMapper.EngramRowBytes);
        Assert.Equal(256, DwarfStarV41KeyMapper.EngramPayloadBytes);
        Assert.Equal(8, DwarfStarV41KeyMapper.EngramScaleBytes);
    }
}

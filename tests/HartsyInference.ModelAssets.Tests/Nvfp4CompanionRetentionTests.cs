using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers <c>keepNvfp4Companions</c>: the opt-out that leaves an nvfp4 group exactly as the file wrote it,
/// for a consumer that dequantizes the format itself. MiniMax-H3's conditioning tower is that consumer — its
/// <c>Nvfp4Linear</c> reads <c>.weight_scale</c>/<c>.weight_scale_2</c> as KEYS and dequantizes one slice per forward,
/// so the default fold-and-widen turned a 15.7 GB mmap into a full materialization and took the DiT's VRAM headroom
/// with it. Every test here pairs the retained case with the folded one, because "the companions are present" is only
/// meaningful against a run of the same fixture where they are not.</summary>
public sealed unsafe class Nvfp4CompanionRetentionTests
{
    /// <summary>E4M3 byte for 1.0.</summary>
    private const byte One = 0x38;

    private static Tensor Filled(TensorShape shape, DType dtype, byte value)
    {
        Tensor t = new Tensor(shape, dtype);
        byte* p = (byte*)t.DataPointer;
        long bytes = dtype.ComputeByteCount(shape.ElementCount);
        for (long i = 0; i < bytes; i++) p[i] = value;
        return t;
    }

    private static Tensor Scalar(float value)
    {
        Tensor t = new Tensor(new TensorShape(1), DType.F32);
        ((float*)t.DataPointer)[0] = value;
        return t;
    }

    /// <summary>A <c>.comfy_quant</c> descriptor blob, which ships as UTF-8 JSON in a U8 tensor.</summary>
    private static Tensor Utf8Blob(string json)
    {
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(json);
        Tensor t = new Tensor(new TensorShape(bytes.Length), DType.U8);
        bytes.CopyTo(t.AsSpan<byte>());
        return t;
    }

    /// <summary>A complete nvfp4 group: U8 nibble bank, rank-2 F8E4M3 block scales, F32 scalar global scale.</summary>
    private static Dictionary<string, Tensor> Nvfp4Group(string prefix, Dictionary<string, Tensor>? into = null)
    {
        Dictionary<string, Tensor> w = into ?? new Dictionary<string, Tensor>();
        w[$"{prefix}.weight"] = Filled(new TensorShape(2, 8), DType.U8, 0x22);
        w[$"{prefix}.weight_scale"] = Filled(new TensorShape(2, 1), DType.F8E4M3, One);
        w[$"{prefix}.weight_scale_2"] = Scalar(1f);
        return w;
    }

    [Fact]
    public void KeptGroup_StaysPackedWithItsCompanions_WhereTheFoldedOneIsWidened()
    {
        Dictionary<string, Tensor> kept = CheckpointConvertUtils.ApplyFp8ScaledDequant(
            Nvfp4Group("blk"), keepNvfp4Companions: true);
        Dictionary<string, Tensor> folded = CheckpointConvertUtils.ApplyFp8ScaledDequant(Nvfp4Group("blk"));

        // Retained: the bytes the file holds, untouched, plus both companions under their own keys.
        Assert.Equal(DType.U8, kept["blk.weight"].DType);
        Assert.Equal(8, kept["blk.weight"].Shape[1]);
        Assert.Equal(DType.F8E4M3, kept["blk.weight_scale"].DType);
        Assert.Equal(DType.F32, kept["blk.weight_scale_2"].DType);

        // Negative control on the same fixture: without the flag the weight is widened 4x and the keys are gone.
        // Asserting only the retained side would pass against an implementation that never drops anything.
        Assert.Equal(DType.F16, folded["blk.weight"].DType);
        Assert.Equal(16, folded["blk.weight"].Shape[1]);
        Assert.False(folded.ContainsKey("blk.weight_scale"));
        Assert.False(folded.ContainsKey("blk.weight_scale_2"));
    }

    [Fact]
    public void KeptGroup_CarriesItsAwqPreQuantScale()
    {
        // The AWQ case is why this flag exists rather than `residentNvfp4`: QuantInfo has no pre_quant_scale field,
        // so TryAttachResident refuses these layers and they take the eager path. H3's published encoder is AWQ.
        Dictionary<string, Tensor> source = Nvfp4Group("blk");
        source["blk.pre_quant_scale"] = Filled(new TensorShape(16), DType.F32, 0);

        Dictionary<string, Tensor> kept = CheckpointConvertUtils.ApplyFp8ScaledDequant(
            source, keepNvfp4Companions: true);
        Dictionary<string, Tensor> resident = CheckpointConvertUtils.ApplyFp8ScaledDequant(
            Nvfp4Group("blk"), residentNvfp4: true);

        Assert.Equal(DType.U8, kept["blk.weight"].DType);
        Assert.True(kept.ContainsKey("blk.pre_quant_scale"));
        Assert.True(kept.ContainsKey("blk.weight_scale"));
        // Contrast: residentNvfp4 keeps the weight packed too, but by moving the scales onto the tensor — the
        // companion keys are still dropped, which is exactly what Nvfp4Linear cannot read.
        Assert.False(resident.ContainsKey("blk.weight_scale"));
    }

    [Fact]
    public void KeptGroup_DoesNotChangeFp8Folding()
    {
        Dictionary<string, Tensor> source = Nvfp4Group("nv");
        Tensor fp8 = Filled(new TensorShape(2, 4), DType.F8E4M3, One);
        source["fp8blk.weight"] = fp8;
        source["fp8blk.scale_weight"] = Scalar(0.25f);

        Dictionary<string, Tensor> result = CheckpointConvertUtils.ApplyFp8ScaledDequant(
            source, keepNvfp4Companions: true);

        Assert.Equal(0.25f, result["fp8blk.weight"].Fp8ScaleFactor);
        Assert.False(result.ContainsKey("fp8blk.scale_weight"));
        Assert.Equal(DType.U8, result["nv.weight"].DType);
    }

    [Fact]
    public void KeptGroup_DoesNotChangeInt8Folding()
    {
        // int8_tensorwise shares the `.weight_scale` suffix with nvfp4's block scales. The retention rule is keyed on
        // the WEIGHT's dtype, so an I8 weight still folds onto QuantInfo — where MiniMaxH3TextEncoder reads it for
        // the published int8_convrot build, which is the build this flag must not regress. The descriptor is not
        // optional padding here: without it the fold refuses the weight outright rather than guessing whether it was
        // ConvRot-rotated, so a fixture missing it would prove nothing about retention.
        const int Rows = 4, Cols = 256;
        Dictionary<string, Tensor> source = Nvfp4Group("nv");
        source["int8blk.weight"] = Filled(new TensorShape(Rows, Cols), DType.I8, 1);
        source["int8blk.weight_scale"] = Filled(new TensorShape(Rows, 1), DType.F32, 0);
        source["int8blk.comfy_quant"] = Utf8Blob(
            "{\"format\": \"int8_tensorwise\", \"convrot\": true, \"convrot_groupsize\": 256, \"per_row\": true}");

        Dictionary<string, Tensor> result = CheckpointConvertUtils.ApplyFp8ScaledDequant(
            source, keepNvfp4Companions: true);

        Assert.NotNull(result["int8blk.weight"].QuantInfo);
        Assert.NotNull(result["int8blk.weight"].QuantInfo!.RowScale);
        Assert.False(result.ContainsKey("int8blk.weight_scale"));
        Assert.False(result.ContainsKey("int8blk.comfy_quant"));
        Assert.Equal(DType.U8, result["nv.weight"].DType);
    }

    [Fact]
    public void IncompleteGroup_TakesItsNormalPath_RatherThanBeingStranded()
    {
        // A U8 weight with no `.weight_scale_2` is not nvfp4. Retaining its `.weight_scale` would leave a companion
        // in a dictionary the caller's own loader does not expect, so the rule must require all three conditions —
        // the same three the eager branch checks.
        Dictionary<string, Tensor> source = new()
        {
            ["partial.weight"] = Filled(new TensorShape(2, 8), DType.U8, 0x22),
            ["partial.weight_scale"] = Filled(new TensorShape(2, 1), DType.F8E4M3, One),
            ["other.weight"] = Filled(new TensorShape(2, 4), DType.F8E4M3, One),
            ["other.scale_weight"] = Scalar(1f),
        };

        Dictionary<string, Tensor> result = CheckpointConvertUtils.ApplyFp8ScaledDequant(
            source, keepNvfp4Companions: true);

        Assert.Equal(DType.U8, result["partial.weight"].DType); // untouched, as it is with the flag off
        Assert.False(result.ContainsKey("partial.weight_scale"));
    }

    [Fact]
    public void KeptGroup_StillDropsItsComfyQuantDescriptor()
    {
        // Only the two scale companions are the packed format's payload. `.comfy_quant` is a format declaration this
        // pass consumes, and leaving it behind would pollute the dictionary the encoder iterates.
        Dictionary<string, Tensor> source = Nvfp4Group("blk");
        source["blk.comfy_quant"] = Filled(new TensorShape(8), DType.U8, (byte)'{');

        Dictionary<string, Tensor> result = CheckpointConvertUtils.ApplyFp8ScaledDequant(
            source, keepNvfp4Companions: true);

        Assert.True(result.ContainsKey("blk.weight_scale"));
        Assert.False(result.ContainsKey("blk.comfy_quant"));
    }
}

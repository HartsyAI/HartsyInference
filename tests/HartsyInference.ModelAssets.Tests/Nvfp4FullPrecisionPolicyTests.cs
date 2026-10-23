using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers the per-layer <c>full_precision_matrix_mult</c> policy on resident nvfp4 weights. A flagged layer must
/// not be eligible for the native block-scaled GEMM (it takes the dense path), and every unflagged layer must stay
/// eligible. The policy is the layer's own <c>.comfy_quant</c> blob when it has one, else the file-level config.</summary>
public sealed unsafe class Nvfp4FullPrecisionPolicyTests
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

    private static Tensor Utf8Blob(string json)
    {
        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(json);
        Tensor t = new Tensor(new TensorShape(bytes.Length), DType.U8);
        bytes.CopyTo(t.AsSpan<byte>());
        return t;
    }

    /// <summary>A complete nvfp4 group with an optional per-layer <c>.comfy_quant</c> blob.</summary>
    private static void AddGroup(Dictionary<string, Tensor> w, string prefix, string? blobJson)
    {
        w[$"{prefix}.weight"] = Filled(new TensorShape(2, 8), DType.U8, 0x22);
        w[$"{prefix}.weight_scale"] = Filled(new TensorShape(128, 4), DType.F8E4M3, One);
        w[$"{prefix}.weight_scale_2"] = Scalar(1f);
        if (blobJson is not null) w[$"{prefix}.comfy_quant"] = Utf8Blob(blobJson);
    }

    private static Dictionary<string, Tensor> Resident(Dictionary<string, Tensor> w,
        IReadOnlyDictionary<string, string>? fileMetadata = null)
        => CheckpointConvertUtils.ApplyFp8ScaledDequant(w, residentNvfp4: true, fileMetadata: fileMetadata);

    [Fact]
    public void BlobFlaggedLayer_TakesDense_AndUnflaggedLayersStayNative()
    {
        Dictionary<string, Tensor> w = new Dictionary<string, Tensor>();
        AddGroup(w, "a", "{\"format\": \"nvfp4\"}");
        AddGroup(w, "b", "{\"format\": \"nvfp4\", \"full_precision_matrix_mult\": true}");
        AddGroup(w, "c", "{\"format\": \"nvfp4\", \"params\": {\"full_precision_matrix_mult\": false}}");

        Dictionary<string, Tensor> result = Resident(w);

        Assert.True(result["a.weight"].QuantInfo!.NativeGemmEligible);
        Assert.False(result["b.weight"].QuantInfo!.NativeGemmEligible);
        Assert.True(result["b.weight"].QuantInfo!.FullPrecisionMatMul);
        Assert.True(result["c.weight"].QuantInfo!.NativeGemmEligible);
        foreach (KeyValuePair<string, Tensor> kvp in result) Assert.Equal(DType.F4E2M1, kvp.Value.DType);
        Assert.False(result.ContainsKey("b.comfy_quant"));
    }

    [Fact]
    public void FileLevelFlag_AppliesOnlyWhereNoBlobExists()
    {
        Dictionary<string, Tensor> w = new Dictionary<string, Tensor>();
        AddGroup(w, "a", null);
        AddGroup(w, "b", "{\"format\": \"nvfp4\"}");
        AddGroup(w, "c", null);
        AddGroup(w, "d", null);
        Dictionary<string, string> metadata = new Dictionary<string, string>
        {
            ["_quantization_metadata"] = "{\"format_version\": \"1.0\", \"layers\": {"
                + "\"a\": {\"format\": \"nvfp4\", \"full_precision_matrix_mult\": true}, "
                + "\"b\": {\"format\": \"nvfp4\", \"full_precision_matrix_mult\": true}, "
                + "\"c\": {\"format\": \"nvfp4\"}, "
                + "\"d\": {\"format\": \"nvfp4\", \"params\": {\"full_precision_matrix_mult\": true}}}}",
        };

        Dictionary<string, Tensor> result = Resident(w, metadata);

        Assert.False(result["a.weight"].QuantInfo!.NativeGemmEligible);
        Assert.True(result["b.weight"].QuantInfo!.NativeGemmEligible, "the per-layer blob is authoritative over the file-level mirror.");
        Assert.True(result["c.weight"].QuantInfo!.NativeGemmEligible);
        Assert.False(result["d.weight"].QuantInfo!.NativeGemmEligible, "the nested params spelling is read the same as the top-level one.");
    }

    [Fact]
    public void LtxConvert_FileLevelMark_ReachesResidentPath_AndBlobStillWins()
    {
        const string flagged = "model.diffusion_model.file.a";
        const string blobbed = "model.diffusion_model.blob.b";
        Dictionary<string, Tensor> w = new Dictionary<string, Tensor>();
        AddGroup(w, flagged, null);
        AddGroup(w, blobbed, "{\"format\": \"nvfp4\"}");
        Dictionary<string, string> metadata = new Dictionary<string, string>
        {
            ["_quantization_metadata"] = "{\"format_version\": \"1.0\", \"layers\": {"
                + $"\"{flagged}\": {{\"format\": \"nvfp4\", \"full_precision_matrix_mult\": true}}, "
                + $"\"{blobbed}\": {{\"format\": \"nvfp4\", \"full_precision_matrix_mult\": true}}}}}}",
        };

        LtxVideo2CheckpointConverter.ConvertedWeights c = LtxVideo2CheckpointConverter.Convert(w, residentNvfp4: true, fileMetadata: metadata);

        Assert.False(c.Transformer["file.a.weight"].QuantInfo!.NativeGemmEligible,
            "a file-level mark with no per-layer blob must reach the LTX-2 resident path.");
        Assert.True(c.Transformer["blob.b.weight"].QuantInfo!.NativeGemmEligible,
            "the per-layer blob is authoritative over the file-level mirror on the LTX-2 path too.");
    }

    [Fact]
    public void LtxConvert_WithoutFileMetadata_FileLevelMarkIsNotApplied()
    {
        const string flagged = "model.diffusion_model.file.a";
        Dictionary<string, Tensor> w = new Dictionary<string, Tensor>();
        AddGroup(w, flagged, null);

        LtxVideo2CheckpointConverter.ConvertedWeights c = LtxVideo2CheckpointConverter.Convert(w, residentNvfp4: true);

        Assert.True(c.Transformer["file.a.weight"].QuantInfo!.NativeGemmEligible,
            "control: with no metadata passed, the converter has no file-level policy to apply.");
    }
}

using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Nvfp4;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>An NVFP4 group comes in two layouts: Comfy (blocked scales, high nibble first) and ModelOpt (row-major scales,
/// low nibble first, what this repo's repacker wrote before it emitted Comfy layout). Reading one as the other scores about
/// 0.00 against the source weights, which is how the repacked MiniMax-H3 build was mis-read, so each path here is checked
/// against the weights it was quantized from, never against another decoder.</summary>
public sealed class Nvfp4LayoutTests
{
    private const int Rows = 200;   // not a multiple of 128: the converted scale tensor is padded
    private const int Cols = 512;

    [Fact]
    public void LegacyModelOptFile_DecodesToItsSource()
    {
        float[] source = Source();
        using TempFile file = Write(ModelOptNvfp4Codec.QuantizeToCompanions("layer.weight", source, Rows, Cols), metadata: null);
        using CheckpointSource opened = CheckpointSource.Open(file.Path);
        Tensor weight = opened.Weights["layer.weight"];
        Assert.Equal(DType.F16, weight.DType);
        Assert.True(Pearson(source, weight) > 0.99, $"read as ModelOpt layout: r = {Pearson(source, weight):F4}");
    }

    /// <summary>The negative control: the same bytes through the Comfy decode, which is what the loader did before.</summary>
    [Fact]
    public void LegacyModelOptFile_ReadAsComfyLayout_IsNoise()
    {
        float[] source = Source();
        using TempFile file = Write(ModelOptNvfp4Codec.QuantizeToCompanions("layer.weight", source, Rows, Cols), metadata: null);
        using CheckpointSource raw = CheckpointSource.Open(file.Path, new CheckpointOpenOptions { FoldQuantCompanions = false });
        float global = raw.Weights["layer.weight_scale_2"].AsReadOnlySpan<float>()[0];
        // The old decode read the row-major bytes as the blocked tile; zero-pad them to the tile's rows so it stays in bounds.
        Tensor rowMajor = raw.Weights["layer.weight_scale"];
        using Tensor padded = new(new TensorShape((rowMajor.Shape[0] + 127) / 128 * 128, rowMajor.Shape[1]), DType.F8E4M3);
        rowMajor.AsReadOnlySpan<byte>().CopyTo(padded.AsSpan<byte>());
        using Tensor misread = CheckpointConvertUtils.DequantNvfp4ToF16(raw.Weights["layer.weight"], padded, global);
        Assert.True(Math.Abs(Pearson(source, misread)) < 0.2, $"the old decode should be noise: r = {Pearson(source, misread):F4}");
    }

    [Fact]
    public void LegacyModelOptFile_StaysResidentInComfyLayout()
    {
        float[] source = Source();
        using TempFile file = Write(ModelOptNvfp4Codec.QuantizeToCompanions("layer.weight", source, Rows, Cols), metadata: null);
        using CheckpointSource opened = CheckpointSource.Open(file.Path, new CheckpointOpenOptions { ResidentNvfp4 = true });
        Tensor weight = opened.Weights["layer.weight"];
        Assert.Equal(DType.F4E2M1, weight.DType);
        Tensor blockScale = weight.QuantInfo!.BlockScale!;
        Assert.Equal(new TensorShape(256, Cols / 16), blockScale.Shape);   // padded to the 128-row tile
        using Tensor decoded = Nvfp4ResidentCodec.DequantToBf16(weight, blockScale, weight.QuantInfo.GlobalScale!);
        Assert.True(Pearson(source, decoded) > 0.99, $"resident decode: r = {Pearson(source, decoded):F4}");
    }

    /// <summary>The repacker's writer: Comfy layout plus <c>_quantization_metadata</c>, so the bytes on disk need no
    /// conversion and the loader leaves them alone.</summary>
    [Fact]
    public void ComfyWriterOutput_IsComfyLayoutOnDisk_AndDecodesToItsSource()
    {
        float[] source = Source();
        Dictionary<string, string> metadata = new() { ["_quantization_metadata"] = Nvfp4Codec.QuantizationMetadata(["layer"]) };
        using TempFile file = Write(Nvfp4Codec.QuantizeToComfyCompanions("layer.weight", source, Rows, Cols), metadata);

        using CheckpointSource raw = CheckpointSource.Open(file.Path, new CheckpointOpenOptions { FoldQuantCompanions = false });
        float global = raw.Weights["layer.weight_scale_2"].AsReadOnlySpan<float>()[0];
        using Tensor onDisk = CheckpointConvertUtils.DequantNvfp4ToF16(raw.Weights["layer.weight"], raw.Weights["layer.weight_scale"], global);
        Assert.True(Pearson(source, onDisk) > 0.99, $"bytes on disk as Comfy layout: r = {Pearson(source, onDisk):F4}");

        using CheckpointSource opened = CheckpointSource.Open(file.Path);
        Assert.True(Pearson(source, opened.Weights["layer.weight"]) > 0.99);
    }

    private static float[] Source()
    {
        float[] values = new float[Rows * Cols];
        Random rng = new(23);
        // Row-dependent magnitudes, so a scale read from the wrong row or block shows up.
        for (int r = 0; r < Rows; r++)
        {
            float rowScale = 0.05f + 2f * (float)rng.NextDouble();
            for (int c = 0; c < Cols; c++) values[r * Cols + c] = (float)(rng.NextDouble() * 2 - 1) * rowScale;
        }
        return values;
    }

    private static double Pearson(float[] source, Tensor decoded)
    {
        using Tensor f32 = decoded.DType == DType.F32 ? decoded : decoded.CastTo(DType.F32);
        ReadOnlySpan<float> d = f32.AsReadOnlySpan<float>();
        double ms = 0, md = 0;
        for (int i = 0; i < source.Length; i++) { ms += source[i]; md += d[i]; }
        ms /= source.Length; md /= source.Length;
        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < source.Length; i++)
        {
            double a = source[i] - ms, b = d[i] - md;
            sxy += a * b; sxx += a * a; syy += b * b;
        }
        return sxy / Math.Sqrt(sxx * syy);
    }

    private static TempFile Write(List<KeyValuePair<string, Tensor>> tensors, Dictionary<string, string>? metadata)
    {
        TempFile file = new();
        try
        {
            SafeTensorsMerger.Write(file.Path, tensors, metadata: metadata);
        }
        finally
        {
            foreach (KeyValuePair<string, Tensor> tensor in tensors) tensor.Value.Dispose();
        }
        return file;
    }

    private sealed class TempFile : IDisposable
    {
        private readonly string _dir = Directory.CreateTempSubdirectory("nvfp4-layout-").FullName;
        public string Path => System.IO.Path.Combine(_dir, "model.safetensors");
        public void Dispose() => Directory.Delete(_dir, recursive: true);
    }
}

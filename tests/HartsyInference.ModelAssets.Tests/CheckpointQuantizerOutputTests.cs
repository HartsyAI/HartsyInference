using System.Security.Cryptography;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Metadata;
using HartsyInference.ModelAssets.Nvfp4;
using HartsyInference.ModelAssets.Quant;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>What a quantized copy of a safetensors checkpoint carries and writes: the source's identity, a digest that
/// describes the copy's own payload, and for NVFP4 the ComfyUI companions, the layer marker and values that decode back
/// to the source within the format's error. Synthetic BF16 weights; no real checkpoint is read.</summary>
public sealed class CheckpointQuantizerOutputTests : IDisposable
{
    /// <summary>1024 x 1024 is 2^20 elements, the smallest weight the quantizers touch.</summary>
    private const int Side = 1024;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-quant-" + Guid.NewGuid().ToString("N"));

    public CheckpointQuantizerOutputTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static float[] Wave(int count) => Enumerable.Range(0, count).Select(i => MathF.Sin(i * 0.013f) * 2f).ToArray();

    private static Tensor Bf16(float[] values, params long[] shape)
    {
        using Tensor f32 = new(new TensorShape(shape), DType.F32);
        values.AsSpan().CopyTo(f32.AsSpan<float>());
        return f32.CastTo(DType.BF16);
    }

    private static Dictionary<string, string> Identity() => new(StringComparer.Ordinal)
    {
        ["modelspec.architecture"] = "iris-z-small",
        ["modelspec.license"] = "other",
        ["modelspec.title"] = "Iris-Z-small (bf16)",
        ["modelspec.hash_sha256"] = "0x" + new string('0', 64),
        ["hartsy.engine_id"] = "iris-z-small",
    };

    /// <summary>One eligible weight, one rank-1 norm and one weight below the element gate, all BF16.</summary>
    private string WriteSource(Dictionary<string, string>? metadata)
    {
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal)
        {
            ["layer.weight"] = Bf16(Wave(Side * Side), Side, Side),
            ["layer.norm.weight"] = Bf16(Wave(Side), Side),
            ["tiny.weight"] = Bf16(Wave(4 * Side), 4, Side),
        };
        string path = Path.Combine(_dir, "source.safetensors");
        SafeTensorsWriter.Save(path, tensors, metadata);
        foreach (Tensor tensor in tensors.Values)
        {
            tensor.Dispose();
        }
        return path;
    }

    private static string PayloadSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        Span<byte> lengthPrefix = stackalloc byte[8];
        stream.ReadExactly(lengthPrefix);
        stream.Seek(8 + BitConverter.ToInt64(lengthPrefix), SeekOrigin.Begin);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static QuantizationJob Job(string source, string output, QuantizationTargetKind kind) => new()
    {
        SourcePath = source,
        OutputPath = output,
        Target = new QuantizationTarget(kind),
    };

    [Fact]
    public void Fp8ScaledCopyKeepsTheIdentityAndDigestsItsOwnPayload()
    {
        string source = WriteSource(Identity());
        string output = Path.Combine(_dir, "out", "fp8.safetensors");

        CheckpointQuantizer.Quantize(Job(source, output, QuantizationTargetKind.Fp8Scaled));

        CheckpointHeader header = CheckpointHeader.Read(output);
        Assert.Equal("iris-z-small", header.Metadata["modelspec.architecture"]);
        Assert.Equal("other", header.Metadata["modelspec.license"]);
        Assert.Equal("Iris-Z-small (fp8-scaled)", header.Metadata["modelspec.title"]);
        Assert.Equal("fp8-scaled", header.Metadata["hartsy.quant"]);
        Assert.Equal(DType.F8E4M3, header.Descriptors["layer.weight"].DType);
        Assert.Equal("0x" + PayloadSha256(output), header.Metadata[ArtifactMetadata.HashKey]);
    }

    [Fact]
    public void FP8CopyOfAnUnidentifiedSourceWritesNoMetadata()
    {
        string source = WriteSource(metadata: null);
        string output = Path.Combine(_dir, "out", "fp8.safetensors");

        CheckpointQuantizer.Quantize(Job(source, output, QuantizationTargetKind.Fp8Scaled));

        Assert.Empty(CheckpointHeader.Read(output).Metadata);
    }

    [Fact]
    public void NvfpCopyWritesComfyCompanionsAndTheLayerMarker()
    {
        string source = WriteSource(Identity());
        string output = Path.Combine(_dir, "out", "nvfp4.safetensors");

        QuantizationReport report = CheckpointQuantizer.Quantize(Job(source, output, QuantizationTargetKind.Nvfp4));

        Assert.Equal(1, report.QuantizedCount);
        CheckpointHeader header = CheckpointHeader.Read(output);
        // Two FP4 values per byte; one E4M3 block scale per 16 values; one F32 tensor scale.
        Assert.Equal(Side * Side / 2, header.Descriptors["layer.weight"].ByteLength);
        Assert.Equal(Side * (Side / Nvfp4Codec.GroupSize), header.Descriptors["layer.weight_scale"].ByteLength);
        Assert.Equal(sizeof(float), header.Descriptors["layer.weight_scale_2"].ByteLength);
        // Ineligible tensors stay BF16, and the norm keeps its name and shape.
        Assert.Equal(DType.BF16, header.Descriptors["tiny.weight"].DType);
        Assert.Equal(DType.BF16, header.Descriptors["layer.norm.weight"].DType);
        Assert.Contains("\"layer\"", header.Metadata["_quantization_metadata"]);
        Assert.Contains("nvfp4", header.Metadata["_quantization_metadata"]);
        Assert.Equal("nvfp4", header.Metadata["hartsy.quant"]);
    }

    [Fact]
    public void NvfpCopyDecodesBackToTheSourceWithinTheFormatsError()
    {
        float[] values = Wave(Side * Side);
        string source = WriteSource(Identity());
        string output = Path.Combine(_dir, "out", "nvfp4.safetensors");
        CheckpointQuantizer.Quantize(Job(source, output, QuantizationTargetKind.Nvfp4));

        // The default open decodes each ComfyUI NVFP4 group back to a dense weight and folds its scales in, which is
        // the path a CPU load takes. The scale companions must be gone from the decoded view.
        using CheckpointSource copy = CheckpointSource.Open(output);
        Assert.False(copy.Weights.ContainsKey("layer.weight_scale"));
        using Tensor dense = copy.Weights["layer.weight"].CastTo(DType.F32);
        float[] decoded = dense.AsSpan<float>().ToArray();

        // E2M1 steps are at most 2 blocks-scale units apart at the top of the range, so a value is off by at most a sixth of
        // its block's maximum. The block scales are themselves E4M3, which adds a little more; 0.2 of the tensor's maximum
        // leaves room for that without letting a wrong layout through.
        float maxAbs = values.Max(MathF.Abs);
        Assert.Equal(values.Length, decoded.Length);
        float worst = 0f;
        for (int i = 0; i < values.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(decoded[i] - values[i]));
        }
        Assert.True(worst <= 0.2f * maxAbs, $"worst abs error {worst} exceeds 0.2 x max {maxAbs}");
    }
}

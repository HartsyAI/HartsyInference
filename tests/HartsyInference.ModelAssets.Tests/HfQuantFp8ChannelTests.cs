using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>compressed-tensors FP8 with per-row (<c>channel</c>) and per-tensor scales, as the MOSS-VL FP8 release ships it, read
/// through <see cref="HfQuantCheckpoint"/> and loaded through <see cref="HfQuantWeightCache"/>. Synthetic files, no weights.</summary>
public sealed class HfQuantFp8ChannelTests : IDisposable
{
    // E4M3 codes with exact values: 0x38 = 1, 0x40 = 2, 0x30 = 0.5, 0xB8 = -1.
    private static readonly byte[] Codes = [0x38, 0x40, 0x30, 0xB8, 0x40, 0x38, 0xB8, 0x30];
    private static readonly float[] Values = [1f, 2f, 0.5f, -1f, 2f, 1f, -1f, 0.5f];
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-fp8ch-" + Guid.NewGuid().ToString("N"));

    public HfQuantFp8ChannelTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static Tensor Fp8(int rows, int cols)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F8E4M3);
        Span<byte> bytes = t.AsSpan<byte>();
        for (int i = 0; i < bytes.Length; i++) bytes[i] = Codes[i % Codes.Length];
        return t;
    }

    private static Tensor Bf16(float[] values, params long[] shape)
    {
        using Tensor f32 = new(new TensorShape(shape), DType.F32);
        values.AsSpan().CopyTo(f32.AsSpan<float>());
        return f32.CastTo(DType.BF16);
    }

    private string Write(Dictionary<string, Tensor> tensors)
    {
        string path = Path.Combine(_dir, "model.safetensors");
        SafeTensorsWriter.Save(path, tensors);
        foreach (Tensor t in tensors.Values) t.Dispose();
        return _dir;
    }

    [Fact]
    public void ARowScaledFp8WeightDecodesAsValueTimesItsRowScale()
    {
        const int rows = 4, cols = 64;
        Write(new Dictionary<string, Tensor>
        {
            ["layer.q_proj.weight"] = Fp8(rows, cols),
            ["layer.q_proj.weight_scale"] = Bf16([0.5f, 1f, 2f, 4f], rows, 1),
            ["layer.q_proj.input_scale"] = Bf16([1f], 1),
            ["layer.norm.weight"] = Bf16([1f, 1f, 1f, 1f], 4),
        });
        using HfQuantCheckpoint checkpoint = HfQuantCheckpoint.Open(_dir);

        Assert.Equal(HfWeightStorage.Fp8Channel, checkpoint.Weights["layer.q_proj.weight"].Storage);
        Assert.DoesNotContain("layer.q_proj.weight_scale", checkpoint.Weights.Keys);
        Assert.DoesNotContain("layer.q_proj.input_scale", checkpoint.Weights.Keys);
        Assert.Equal(HfWeightStorage.Plain, checkpoint.Weights["layer.norm.weight"].Storage);

        using Tensor f32 = checkpoint.DequantizeF32("layer.q_proj.weight");
        ReadOnlySpan<float> got = f32.AsReadOnlySpan<float>();
        float[] scales = [0.5f, 1f, 2f, 4f];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                Assert.Equal(Values[(r * cols + c) % Values.Length] * scales[r], got[r * cols + c]);

        // FP8 values on a power-of-two scale sit exactly on Q8_0's grid for these blocks, so the 8-bit repack round-trips.
        using Tensor q8 = checkpoint.Produce("layer.q_proj.weight", DType.Q8_0);
        Assert.Equal(DType.Q8_0, q8.DType);
    }

    [Fact]
    public void ATensorScaledFp8WeightUsesTheOneScaleEverywhere()
    {
        Write(new Dictionary<string, Tensor>
        {
            ["w.weight"] = Fp8(2, 32),
            ["w.weight_scale"] = Bf16([3f], 1),
        });
        using HfQuantCheckpoint checkpoint = HfQuantCheckpoint.Open(_dir);
        using Tensor f32 = checkpoint.DequantizeF32("w.weight");
        ReadOnlySpan<float> got = f32.AsReadOnlySpan<float>();
        for (int i = 0; i < got.Length; i++) Assert.Equal(Values[i % Values.Length] * 3f, got[i]);
    }

    [Fact]
    public void AScaleOfTheWrongShapeIsRefusedByName()
    {
        Write(new Dictionary<string, Tensor>
        {
            ["w.weight"] = Fp8(4, 32),
            ["w.weight_scale"] = Bf16([1f, 1f], 2, 1),
        });
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => HfQuantCheckpoint.Open(_dir));
        Assert.Contains("w.weight_scale", error.Message);
    }

    [Fact]
    public void TheWeightCacheBorrowsPlainWeightsConvertsTheRestOnceAndRemapsItAfter()
    {
        Write(new Dictionary<string, Tensor>
        {
            ["a.weight"] = Fp8(4, 64),
            ["a.weight_scale"] = Bf16([1f, 1f, 1f, 1f], 4, 1),
            ["b.weight"] = Bf16(Enumerable.Range(0, 64).Select(i => i / 64f).ToArray(), 1, 64),
            ["skip.weight"] = Bf16([1f], 1),
        });
        string cache = Path.Combine(_dir, "_gguf-cache", "test.gguf");
        DType Target(HfQuantWeight w) => w.Storage == HfWeightStorage.Plain ? w.StoredDType : DType.Q8_0;

        DateTime written;
        {
            HfQuantCheckpoint checkpoint = HfQuantCheckpoint.Open(_dir);
            (IReadOnlyDictionary<string, Tensor> weights, IDisposable owner) =
                HfQuantWeightCache.Load(checkpoint, Target, cache, "test", "[test]", include: name => name != "skip.weight");
            using (owner)
            {
                Assert.Equal(["a.weight", "b.weight"], weights.Keys.Order().ToArray());
                Assert.Equal(DType.Q8_0, weights["a.weight"].DType);
                Assert.Equal(DType.BF16, weights["b.weight"].DType);
                Assert.Equal(new TensorShape(4, 64), weights["a.weight"].Shape);
            }
            written = File.GetLastWriteTimeUtc(cache);
        }
        {
            HfQuantCheckpoint checkpoint = HfQuantCheckpoint.Open(_dir);
            (IReadOnlyDictionary<string, Tensor> weights, IDisposable owner) =
                HfQuantWeightCache.Load(checkpoint, Target, cache, "test", "[test]", include: name => name != "skip.weight");
            using (owner) Assert.Equal(DType.Q8_0, weights["a.weight"].DType);
        }
        Assert.Equal(written, File.GetLastWriteTimeUtc(cache));
    }

    [Fact]
    public void AConfigWithQuantizationIsDetected()
    {
        File.WriteAllText(Path.Combine(_dir, "config.json"), """{"model_type":"x","quantization_config":{"quant_method":"bitsandbytes"}}""");
        Assert.True(HfQuantCheckpoint.DeclaresQuantization(_dir));
        File.WriteAllText(Path.Combine(_dir, "config.json"), """{"model_type":"x","text_config":{"hidden_size":4}}""");
        Assert.False(HfQuantCheckpoint.DeclaresQuantization(_dir));
    }
}

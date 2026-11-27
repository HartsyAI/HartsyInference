using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Writes the Fibo 2.0 int8 transformer as a bf16 checkpoint with every int8 weight dequantized, for the quantization diagnosis.
/// The int8 companions (<c>.weight_scale</c>, <c>.comfy_quant</c>) are attached to their weights first, exactly as the loader does
/// (<see cref="CheckpointConvertUtils.AttachInt8QuantInfo"/>), then the engine's CPU widening (<see cref="QuantizedWeightPolicy"/>) writes
/// each one as BF16 = int8 times its row scale. It is not an independent check of the dequantization math (diagnosis test B).</summary>
public sealed class Fibo2Int8DequantExportTests(Xunit.Abstractions.ITestOutputHelper output)
{
    /// <summary>Exports <paramref name="input"/> to <paramref name="output"/> and returns the tensor counts. Refuses to write when no int8
    /// weight was attached, or when an I8 tensor or a quantization companion remains in the output.</summary>
    public static (int Written, int Quantized) Export(string input, string outPath)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(input);
        Dictionary<string, Tensor> loaded = loader.GetAllTensors().ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
        Dictionary<string, Tensor> weights = CheckpointConvertUtils.AttachInt8QuantInfo(loaded);

        int quantized = weights.Values.Count(t => t.QuantInfo is not null);
        if (quantized == 0)
            throw new InvalidDataException($"{input}: no int8 weight carries QuantInfo after attaching the companions; refusing to write an unchanged copy.");

        using CpuBackend cpu = new();
        using QuantizedWeightPolicy.PreparedWeights prepared = QuantizedWeightPolicy.PrepareForBackend(weights, cpu, DType.BF16);

        List<string> leftovers = weights.Where(pair => pair.Value.DType == DType.I8 || pair.Key.EndsWith(".comfy_quant", StringComparison.Ordinal)
                || pair.Key.EndsWith(".weight_scale", StringComparison.Ordinal))
            .Select(pair => pair.Key).ToList();
        if (leftovers.Count > 0)
            throw new InvalidDataException($"{input}: {leftovers.Count} int8 or companion tensors remain after dequantization, e.g. {leftovers[0]}.");

        SafeTensorsWriter.Save(outPath, weights);
        return (weights.Count, quantized);
    }

    /// <summary>Reported as skipped, never passed, when either variable is unset or the input is not a file.</summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    public sealed class ExportFactAttribute : FactAttribute
    {
        public ExportFactAttribute()
        {
            string? input = Environment.GetEnvironmentVariable("HARTSY_FIBO2_INT8_FILE");
            string? target = Environment.GetEnvironmentVariable("HARTSY_FIBO2_DEQUANT_OUT");
            if (string.IsNullOrWhiteSpace(input) || !File.Exists(input))
                Skip = "SKIPPED: HARTSY_FIBO2_INT8_FILE is unset or not an existing file; the dequant export did not run.";
            else if (string.IsNullOrWhiteSpace(target) || File.Exists(target))
                Skip = "SKIPPED: HARTSY_FIBO2_DEQUANT_OUT is unset or already exists; the dequant export did not run.";
        }
    }

    [ExportFact]
    [Trait("Category", "Export")]
    public void WriteDequantizedBf16Checkpoint()
    {
        string input = Environment.GetEnvironmentVariable("HARTSY_FIBO2_INT8_FILE")!;
        string target = Environment.GetEnvironmentVariable("HARTSY_FIBO2_DEQUANT_OUT")!;

        (int written, int quantized) = Export(input, target);

        output.WriteLine($"wrote {written} tensors ({quantized} int8 weights dequantized to BF16) to {target}");
        Assert.True(quantized > 0);
        Assert.True(File.Exists(target));
    }

    /// <summary>A small file with the same key layout as the int8 checkpoint: an int8 weight, its F32 row scale, its comfy_quant descriptor
    /// and an unquantized BF16 norm. The export must dequantize the weight to int8 times the row scale and drop the companions.</summary>
    [Fact]
    public void SyntheticInt8_ExportsAsInt8TimesRowScaleInBf16()
    {
        string dir = Path.Combine(Path.GetTempPath(), "fibo2-export-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            sbyte[] q = [1, -2, 3, 4, -5, 6, 7, -8, 9, 10, -11, 12, -13, 14, 15, -16, 17, -18, 19, 20, -21, 22, 23, -24, 25, -26, 27, 28, -29, 30, 31, -32];
            float[] scale = [0.5f, 0.25f, 2f, 0.125f];
            Tensor weight = new(new TensorShape(4, 8), DType.I8);
            for (int i = 0; i < q.Length; i++) weight.AsSpan<sbyte>()[i] = q[i];
            Tensor rowScale = new(new TensorShape(4), DType.F32);
            for (int i = 0; i < 4; i++) rowScale.AsSpan<float>()[i] = scale[i];
            byte[] descriptor = Encoding.UTF8.GetBytes("{\"format\":\"int8_tensorwise\",\"params\":{}}");
            Tensor blob = new(new TensorShape(descriptor.Length), DType.U8);
            descriptor.AsSpan().CopyTo(blob.AsSpan<byte>());
            Tensor norm = new(new TensorShape(8), DType.BF16);
            for (int i = 0; i < 8; i++) norm.AsSpan<ushort>()[i] = TensorCasts.F32ToBf16Bits(1f);

            string input = Path.Combine(dir, "in.safetensors");
            string target = Path.Combine(dir, "out.safetensors");
            SafeTensorsWriter.Save(input, new Dictionary<string, Tensor>(StringComparer.Ordinal)
            {
                ["blk.weight"] = weight, ["blk.weight_scale"] = rowScale, ["blk.comfy_quant"] = blob, ["norm.weight"] = norm,
            });
            weight.Dispose(); rowScale.Dispose(); blob.Dispose(); norm.Dispose();

            (int written, int quantized) = Export(input, target);
            Assert.Equal(1, quantized);
            Assert.Equal(2, written);

            using SafeTensorsLoader check = new();
            check.Load(target);
            Dictionary<string, Tensor> outTensors = check.GetAllTensors().ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
            Assert.Equal(new[] { "blk.weight", "norm.weight" }, outTensors.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray());
            Tensor dequant = outTensors["blk.weight"];
            Assert.Equal(DType.BF16, dequant.DType);
            ReadOnlySpan<ushort> bits = dequant.AsReadOnlySpan<ushort>();
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 8; c++)
                {
                    float expected = q[r * 8 + c] * scale[r];
                    float actual = BitConverter.Int32BitsToSingle(bits[r * 8 + c] << 16);
                    float bf16Expected = BitConverter.Int32BitsToSingle(TensorCasts.F32ToBf16Bits(expected) << 16);
                    Assert.Equal(bf16Expected, actual);
                }
            }
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}

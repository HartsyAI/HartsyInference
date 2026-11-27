using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Writes the Fibo 2.0 int8 transformer as a bf16 checkpoint with every int8 weight widened, for the quantization diagnosis.
/// It uses the engine's own widening on the CPU backend (QuantizedWeightPolicy), so it isolates the GPU int8 kernel from the weights it
/// reads: the dequantized checkpoint run on the bf16 path is the reference for the GPU int8 path. It is not an independent check of the
/// dequantization math; that needs ComfyUI's definition (diagnosis test B).
/// Runs only when HARTSY_FIBO2_INT8_FILE names the int8 transformer and HARTSY_FIBO2_DEQUANT_OUT names a path that does not yet exist.</summary>
public sealed class Fibo2Int8DequantExportTests(Xunit.Abstractions.ITestOutputHelper output)
{
    /// <summary>Reported as skipped, never passed, when either variable is unset or the input is not a file.</summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
    public sealed class ExportFactAttribute : FactAttribute
    {
        public ExportFactAttribute()
        {
            string? input = Environment.GetEnvironmentVariable("HARTSY_FIBO2_INT8_FILE");
            string? output = Environment.GetEnvironmentVariable("HARTSY_FIBO2_DEQUANT_OUT");
            if (string.IsNullOrWhiteSpace(input) || !File.Exists(input))
                Skip = "SKIPPED: HARTSY_FIBO2_INT8_FILE is unset or not an existing file; the dequant export did not run.";
            else if (string.IsNullOrWhiteSpace(output) || File.Exists(output))
                Skip = "SKIPPED: HARTSY_FIBO2_DEQUANT_OUT is unset or already exists; the dequant export did not run.";
        }
    }

    [ExportFact]
    [Trait("Category", "Export")]
    public void WriteDequantizedBf16Checkpoint()
    {
        string input = Environment.GetEnvironmentVariable("HARTSY_FIBO2_INT8_FILE")!;
        string outPath = Environment.GetEnvironmentVariable("HARTSY_FIBO2_DEQUANT_OUT")!;

        using SafeTensorsLoader loader = new();
        loader.Load(input);
        Dictionary<string, Tensor> weights = loader.GetAllTensors().ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
        int packed = weights.Values.Count(t => t.QuantInfo is not null);

        using CpuBackend cpu = new();
        using QuantizedWeightPolicy.PreparedWeights prepared = QuantizedWeightPolicy.PrepareForBackend(weights, cpu, DType.BF16);
        SafeTensorsWriter.Save(outPath, weights);

        output.WriteLine($"wrote {weights.Count} tensors ({packed} quantized, widened to BF16) to {outPath}");
        Assert.True(File.Exists(outPath));
    }
}

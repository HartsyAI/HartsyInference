using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Cuda;
using SharpInference.ModelHandler.Gguf;
using SharpInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace SharpInference.Diffusion.Tests;

/// <summary>Real-world validation of the Phase D wiring: takes a single Q4_K weight from the actual `city96/FLUX.1-schnell-gguf` checkpoint, runs <see cref="CudaBackend.Linear"/> with it, and verifies output sanity (no NaN/Inf, reasonable magnitudes). Exercises the full path: GGUF mmap → Tensor with Q4_K dtype → GpuTransferHelper.CopyToDevice → Linear's CastIfNeeded → GPU dequant kernel → cuBLAS GEMM.
///
/// <para>Supplements the synthetic <see cref="SharpInference.Cuda.Tests.CudaLinearQuantTests"/> with a real ggml-quantized weight, confirming the codecs read on-disk city96 layouts correctly (not just hand-built test data).</para></summary>
public sealed class FluxGgufLinearTests
{
    private readonly ITestOutputHelper _output;
    public FluxGgufLinearTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void Linear_RealQ4_K_FromCity96Gguf_ProducesSaneOutput()
    {
        if (!File.Exists(TestPaths.Flux.SchnellQ4KS))
        {
            _output.WriteLine($"SKIPPED: GGUF not found at {TestPaths.Flux.SchnellQ4KS}");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable.");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        using GgufModelLoader.LoadedGgufModel loaded = GgufModelLoader.Load(TestPaths.Flux.SchnellQ4KS);
        sw.Stop();
        _output.WriteLine($"GGUF mmap loaded in {sw.ElapsedMilliseconds}ms ({loaded.Weights.Count} tensors).");

        // Find a 2D Q4_K weight that's small enough to be quick + has a typical Flux shape.
        // Flux double_blocks have 3072x3072 attn projection weights.
        Tensor? selected = null;
        string selectedName = "";
        foreach (KeyValuePair<string, Tensor> kv in loaded.Weights)
        {
            if (kv.Value.DType != DType.Q4_K) continue;
            if (kv.Value.Shape.Rank != 2) continue;
            long rows = kv.Value.Shape[0];
            long cols = kv.Value.Shape[1];
            if (rows != 3072 || cols != 3072) continue;
            if (!kv.Key.Contains("img_attn", StringComparison.Ordinal) && !kv.Key.Contains("img_mod", StringComparison.Ordinal)) continue;
            selected = kv.Value;
            selectedName = kv.Key;
            break;
        }

        if (selected is null)
        {
            // Fallback: any 2D Q4_K weight.
            foreach (KeyValuePair<string, Tensor> kv in loaded.Weights)
            {
                if (kv.Value.DType == DType.Q4_K && kv.Value.Shape.Rank == 2)
                {
                    selected = kv.Value;
                    selectedName = kv.Key;
                    break;
                }
            }
        }
        Assert.NotNull(selected);
        _output.WriteLine($"Selected weight: '{selectedName}' shape={selected.Shape} dtype={selected.DType} bytes={selected.DType.ComputeByteCount(selected.ElementCount)}");

        int outDim = (int)selected.Shape[0];
        int inDim = (int)selected.Shape[1];
        const int batch = 4;

        Tensor input = new Tensor(new TensorShape(batch, inDim), DType.F16);
        Tensor output = new Tensor(new TensorShape(batch, outDim), DType.F16);
        try
        {
            Random rng = new Random(42);
            Half* ip = (Half*)input.DataPointer;
            for (int i = 0; i < batch * inDim; i++) ip[i] = (Half)((rng.NextDouble() * 2.0 - 1.0) * 0.5);

            string assemblyDir = Path.GetDirectoryName(typeof(FluxGgufLinearTests).Assembly.Location)!;
            string ptxDir = Path.Combine(assemblyDir, "Ptx");
            using CudaBackend backend = Directory.Exists(ptxDir)
                ? new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir)
                : new CudaBackend();

            sw.Restart();
            backend.Linear(output, input, selected, bias: null);
            backend.Sync();
            sw.Stop();
            _output.WriteLine($"Linear (batch={batch}, [{outDim},{inDim}] @ Q4_K weight) executed in {sw.ElapsedMilliseconds}ms.");

            Half* op = (Half*)output.DataPointer;
            int nanCount = 0, infCount = 0;
            float minVal = float.MaxValue, maxVal = float.MinValue;
            double sumAbs = 0;
            int total = batch * outDim;
            for (int i = 0; i < total; i++)
            {
                float v = (float)op[i];
                if (float.IsNaN(v)) { nanCount++; continue; }
                if (float.IsInfinity(v)) { infCount++; continue; }
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
                sumAbs += MathF.Abs(v);
            }
            float meanAbs = (float)(sumAbs / (total - nanCount - infCount));
            _output.WriteLine($"Output stats: min={minVal:F4} max={maxVal:F4} mean_abs={meanAbs:F4}, NaN={nanCount}, Inf={infCount}");

            Assert.Equal(0, nanCount);
            Assert.Equal(0, infCount);
            Assert.True(meanAbs > 0.001f, $"output mean_abs {meanAbs} too small — likely all zero");
            Assert.True(meanAbs < 100f, $"output mean_abs {meanAbs} too large — likely overflow");
        }
        finally
        {
            input.Dispose();
            output.Dispose();
        }
    }
}

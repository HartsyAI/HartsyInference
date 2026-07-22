using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>
/// Smoke tests for the Mistral-Small-3 distill packaged with Flux.2 Dev. Validates that the
/// shared <see cref="LlamaStyleEncoder"/> covers the Mistral architecture (no per-head Q/K norm,
/// no final RMSNorm, GQA 32:8, BF16 embed table + FP8-scaled projections) and that the FP8
/// scale companions are correctly folded by <see cref="CheckpointConvertUtils.ApplyFp8ScaledDequant"/>.
/// </summary>
[Trait("Category", "Integration")]
public sealed class MistralEncoderSmokeTests
{
    private static string MistralWeightsPath => TestPaths.TextEncoders.Mistral3SmallFp8;

    private readonly ITestOutputHelper _output;

    public MistralEncoderSmokeTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Loads the BFL Mistral-Small-3 distill (FP8 mixed, 30 layers) and runs a forward pass on a
    /// short synthetic prompt. Validates: weights load (FP8 scale companions correctly folded),
    /// forward returns finite hidden states of shape <c>[B, S, 5120]</c>. Doesn't validate
    /// against a Python reference — that's a separate validation task.
    /// <para>Skipped when the Mistral checkpoint is not on disk. The Tekken tokenizer embedded
    /// in the safetensors is not yet wired up, so this test feeds dummy token IDs (0..seqLen).</para>
    /// </summary>
    [Fact]
    public void Mistral_LoadAndForward_Gpu()
    {
        if (!File.Exists(MistralWeightsPath))
        {
            _output.WriteLine($"SKIPPED: Mistral checkpoint not found: {MistralWeightsPath}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(MistralEncoderSmokeTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: no CUDA driver available");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/4] Loading {Path.GetFileName(MistralWeightsPath)}...");
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(MistralWeightsPath);
        Dictionary<string, Tensor> raw = loader.GetAllTensors();
        sw.Stop();
        _output.WriteLine($"  {raw.Count} tensors loaded in {sw.ElapsedMilliseconds}ms");

        // Drop the embedded Tekken tokenizer (we don't need it for the smoke test) so it doesn't
        // confuse the FP8 scale pass or look like an unhandled tensor downstream.
        if (raw.ContainsKey("tekken_model"))
        {
            raw.Remove("tekken_model");
            _output.WriteLine("  Dropped embedded tekken_model tensor (not needed for forward pass)");
        }

        _output.WriteLine("[2/4] Folding FP8 scale companions into Fp8ScaleFactor...");
        sw.Restart();
        Dictionary<string, Tensor> processed = CheckpointConvertUtils.ApplyFp8ScaledDequant(raw);
        sw.Stop();
        int fp8Count = 0;
        int scaledCount = 0;
        foreach (KeyValuePair<string, Tensor> kvp in processed)
        {
            if (kvp.Value.DType.IsFp8)
            {
                fp8Count++;
                if (kvp.Value.Fp8ScaleFactor != 1.0f) scaledCount++;
            }
        }
        _output.WriteLine($"  Processed in {sw.ElapsedMilliseconds}ms ({processed.Count} keys; {fp8Count} FP8 tensors, {scaledCount} with non-unity Fp8ScaleFactor)");
        Assert.True(fp8Count > 0, "Expected FP8 tensors in Mistral checkpoint");
        Assert.Equal(fp8Count, scaledCount); // every FP8 weight must have a non-trivial scale folded in

        _output.WriteLine("[3/4] Building encoder + loading weights (Mistral preset, 30 layers)...");
        sw.Restart();
        LlamaStyleEncoder encoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.MistralSmall3);
        encoder.LoadWeights(processed);
        sw.Stop();
        _output.WriteLine($"  Encoder loaded in {sw.ElapsedMilliseconds}ms");

        // Synthetic input: small token sequence. We use ids in [10..10+seqLen) so they're all
        // ordinary BPE merges (not special tokens), avoiding any chat-template assumptions.
        const int SeqLen = 32;
        int[] tokens = new int[SeqLen];
        for (int i = 0; i < SeqLen; i++) tokens[i] = 10 + i;
        int[][] batch = [tokens];

        _output.WriteLine($"[4/4] Running GPU forward on a {SeqLen}-token synthetic prompt...");
        sw.Restart();
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        Tensor hidden = encoder.Encode(backend, batch);
        backend.Sync();
        sw.Stop();
        _output.WriteLine($"  Forward done in {sw.ElapsedMilliseconds}ms, output shape={hidden.Shape}, dtype={hidden.DType}");

        Assert.Equal(1, hidden.Shape[0]);
        Assert.Equal(SeqLen, hidden.Shape[1]);
        Assert.Equal(5120, hidden.Shape[2]);
        Assert.Equal(DType.F32, hidden.DType);

        ReadOnlySpan<float> data = hidden.AsReadOnlySpan<float>();
        int nan = 0, inf = 0;
        float min = float.MaxValue, max = float.MinValue;
        double sumAbs = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v)) nan++;
            else if (float.IsInfinity(v)) inf++;
            else { if (v < min) min = v; if (v > max) max = v; sumAbs += Math.Abs(v); }
        }
        _output.WriteLine($"  Stats: nan={nan}, inf={inf}, min={min:F4}, max={max:F4}, abs_mean={sumAbs / data.Length:F4}");

        Assert.Equal(0, nan);
        Assert.Equal(0, inf);
        Assert.True(sumAbs > 0, "Hidden states are all zero — forward pass produced no signal");

        hidden.Dispose();
        encoder.Dispose();
    }
}

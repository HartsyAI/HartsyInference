using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The released MovieRoll DiT's forward on CUDA, block-streamed: the shared weights are loaded once, each block's weights are converted and
/// loaded just before its forward (the DiT's <c>BeforeBlockForward</c> seam, chained ahead of the streaming scope's own hook), and the scope moves the
/// blocks through the device. The output velocity is compared with the upstream velocity by relative RMS.
/// Environment: <c>MOVIEROLL_DIT_PATH</c> the released file, <c>MOVIEROLL_INPUTS</c> the saved inputs (seed 7), <c>MOVIEROLL_VELOCITY_REF</c> the upstream
/// velocity (safetensors, key <c>velocity</c>). Skipped (reported, not passed) when any is unset or no CUDA device opens.
/// No other environment overrides: the forward runs the engine defaults (exact GEMMs via <c>numerics.movieRollExactGemm</c>, F32 block attention).
/// Bound: relative RMS at most 1e-3.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MovieRollCudaBlockForwardTests(ITestOutputHelper output)
{
    private const double RelativeRmsBound = 1e-3;

    [CudaRealWeightsFact("MOVIEROLL_DIT_PATH", "MOVIEROLL_INPUTS", "MOVIEROLL_VELOCITY_REF")]
    public void CudaBlockStreamedForward_MatchesTheUpstreamVelocity()
    {
        try
        {
            Forward();
        }
        finally
        {
            KnobStore.ResetOverrides();
        }
    }

    private void Forward()
    {
        string dit = Environment.GetEnvironmentVariable("MOVIEROLL_DIT_PATH")!;
        string inputsPath = Environment.GetEnvironmentVariable("MOVIEROLL_INPUTS")!;
        string refPath = Environment.GetEnvironmentVariable("MOVIEROLL_VELOCITY_REF")!;

        HunyuanVideo15Config config = HunyuanVideo15Config.Full;
        using SafeTensorsLoader weights = new();
        weights.Load(dit);
        using SafeTensorsLoader inputs = new();
        inputs.Load(inputsPath);
        using SafeTensorsLoader reference = new();
        reference.Load(refPath);

        using IBackend backend = GpuDevices.OpenCuda();
        output.WriteLine($"MOVIEROLL_EXACT_GEMM {EngineKnobs.MovieRollExactGemm.Value}");

        // Shared weights first: the top level and block 0 go through the converter once; block 0's outputs are released.
        Dictionary<string, Tensor> source = weights.GetAllTensors();
        Dictionary<string, Tensor> sharedOut = HunyuanVideo15CheckpointConverter.Convert(
            source.Where(p => !p.Key.StartsWith("double_blocks.", StringComparison.Ordinal) || p.Key.StartsWith("double_blocks.0.", StringComparison.Ordinal))
                .ToDictionary(p => p.Key, p => p.Value), numBlocks: 1, hidden: config.HiddenSize);
        using HunyuanVideo15Dit dit15 = new(config);
        output.WriteLine($"ATTENTION_F16 {dit15.AllowF16Attention}");
        dit15.LoadSharedWeights(sharedOut.Where(p => !p.Key.StartsWith("transformer_blocks.", StringComparison.Ordinal))
            .ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal));

        List<Tensor> previous = [];
        Action<int> loadBlock = index =>
        {
            Dictionary<string, Tensor> one = new(StringComparer.Ordinal);
            string prefix = $"double_blocks.{index}.";
            foreach (KeyValuePair<string, Tensor> pair in source)
            {
                if (pair.Key.StartsWith(prefix, StringComparison.Ordinal)) one["double_blocks.0." + pair.Key[prefix.Length..]] = pair.Value;
                else if (!pair.Key.StartsWith("double_blocks.", StringComparison.Ordinal)) one[pair.Key] = pair.Value;
            }
            Dictionary<string, Tensor> converted = HunyuanVideo15CheckpointConverter.Convert(one, numBlocks: 1, hidden: config.HiddenSize);
            Dictionary<string, Tensor> blockWeights = new(StringComparer.Ordinal);
            List<Tensor> owned = [];
            foreach (KeyValuePair<string, Tensor> pair in converted)
            {
                if (pair.Key.StartsWith("transformer_blocks.0.", StringComparison.Ordinal))
                {
                    blockWeights["transformer_blocks." + index + pair.Key["transformer_blocks.0".Length..]] = pair.Value;
                    if (!source.ContainsValue(pair.Value)) owned.Add(pair.Value);
                }
                else if (!source.ContainsValue(pair.Value)) pair.Value.Dispose();
            }
            dit15.LoadBlockWeights(index, blockWeights);
            foreach (Tensor old in previous) old.Dispose();
            previous = owned;
        };

        BlockStreamingScope stream = BlockStreamingScope.Open(new BlockStreamingOptions
        {
            Backend = backend,
            Denoiser = dit15,
            ModelName = "MovieRoll",
            HeadroomBytes = 6L << 30,
        });
        try
        {
            // The scope installed its hook; the weight load runs ahead of it for each block.
            Action<int>? scopeHook = dit15.BeforeBlockForward;
            dit15.BeforeBlockForward = index =>
            {
                loadBlock(index);
                scopeHook?.Invoke(index);
            };

            Tensor qwen = inputs.GetTensor("qwen");
            Tensor byt5 = inputs.GetTensor("byt5");
            Tensor noise = inputs.GetTensor("noise");
            float[] frameTimesteps = [1000f, 555.56f, 769.23f, 882.35f, 952.38f];
            System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
            using Tensor velocity = dit15.ForwardPerFrame(backend, noise, frameTimesteps, 1000f, qwen, byt5, conditioning: null);
            watch.Stop();
            output.WriteLine($"FORWARD_SECONDS {watch.Elapsed.TotalSeconds:F1}");
            stream.EndStep();

            Tensor expected = reference.GetTensor("velocity");
            ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
            ReadOnlySpan<float> a = velocity.AsReadOnlySpan<float>();
            Assert.Equal(e.Length, a.Length);
            double sumSquares = 0, sumReference = 0;
            for (int i = 0; i < e.Length; i++)
            {
                double d = a[i] - e[i];
                sumSquares += d * d;
                sumReference += (double)e[i] * e[i];
            }
            double relativeRms = Math.Sqrt(sumSquares / e.Length) / Math.Sqrt(sumReference / e.Length);
            output.WriteLine($"CUDA block-streamed forward: relative RMS {relativeRms:E3} against the upstream velocity (bound {RelativeRmsBound:E0})");
            Assert.True(relativeRms <= RelativeRmsBound, $"relative RMS {relativeRms:E3} exceeds {RelativeRmsBound:E0}.");
        }
        finally
        {
            stream.Dispose();
            foreach (Tensor old in previous) old.Dispose();
        }
    }
}

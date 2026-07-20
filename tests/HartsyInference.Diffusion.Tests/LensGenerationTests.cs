using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>End-to-end Microsoft Lens generation from the ComfyUI split files (DiT + GPT-OSS-20B
/// NVFP4 text encoder + Flux.2 VAE). Env-gated: set <c>LENS_DIT_CHECKPOINT</c>,
/// <c>LENS_TE_CHECKPOINT</c>, <c>LENS_VAE_CHECKPOINT</c>, and <c>GPT_OSS_VOCAB_DIR</c> (dir with
/// <c>gpt_oss_vocab.json</c>/<c>gpt_oss_merges.txt</c>). Saves a BMP under the Output dir for
/// visual inspection. Expect several minutes on an RTX 3060: the MoE text encode alone is ~1 min
/// and the DiT denoise loop still has host-glue overhead.</summary>
public sealed class LensGenerationTests
{
    private readonly ITestOutputHelper _output;
    public LensGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "Slow")]
    public void LensTurbo_GenerateImage_Gpu()
    {
        string? ditPath = Environment.GetEnvironmentVariable("LENS_DIT_CHECKPOINT");
        string? tePath = Environment.GetEnvironmentVariable("LENS_TE_CHECKPOINT");
        string? vaePath = Environment.GetEnvironmentVariable("LENS_VAE_CHECKPOINT");
        string? vocabDir = Environment.GetEnvironmentVariable("GPT_OSS_VOCAB_DIR");
        if (ditPath is null || tePath is null || vaePath is null || vocabDir is null ||
            !File.Exists(ditPath) || !File.Exists(tePath) || !File.Exists(vaePath))
        {
            _output.WriteLine("SKIPPED: set LENS_DIT_CHECKPOINT / LENS_TE_CHECKPOINT / LENS_VAE_CHECKPOINT / GPT_OSS_VOCAB_DIR.");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(LensGenerationTests).Assembly.Location)!;
        if (!Directory.Exists(Path.Combine(assemblyDir, "Ptx")) || !CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: PTX dir or CUDA driver unavailable.");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine("[1/3] Wiring Lens pipeline from ComfyUI files...");
        using CudaBackend backend = new(deviceOrdinal: 0, Path.Combine(assemblyDir, "Ptx"));
        using LensPipelineBundle bundle = LensPipelineFactory.LoadFromComfyFiles(
            backend, ditPath, tePath, vaePath, LensConfig.Turbo);
        using GptOssTokenizer tokenizer = new(
            Path.Combine(vocabDir, "gpt_oss_vocab.json"), Path.Combine(vocabDir, "gpt_oss_merges.txt"));
        _output.WriteLine($"  Wired in {sw.ElapsedMilliseconds}ms");

        const string prompt = "a photo of a corgi wearing a top hat, studio lighting, detailed fur";
        (int[] posTokens, _) = tokenizer.BuildChatInputs(prompt);
        _output.WriteLine($"[2/3] Prompt tokens: {posTokens.Length} (wrapper {GptOssTokenizer.DefaultTxtOffset})");

        // Step/size env overrides (parity probes run 1 step; perf baselines the default 8 @ 1024²).
        int steps = int.TryParse(Environment.GetEnvironmentVariable("LENS_T2I_STEPS"), out int st) ? st : 8;
        int size = int.TryParse(Environment.GetEnvironmentVariable("LENS_T2I_SIZE"), out int sz) ? sz : 1024;
        TextToImageRequest request = new()
        {
            Prompt = prompt,
            Width = size,
            Height = size,
            Steps = steps,
            CfgScale = 1.0f,
            Seed = 42,
        };

        sw.Restart();
        (byte[] rgb, int w, int h, int seed) = bundle.Pipeline.GenerateFromTokens(
            posTokens, negativeTokenIds: null, request,
            p => _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
        _output.WriteLine($"[3/3] Generated {w}x{h} in {sw.Elapsed.TotalSeconds:F1}s");

        Directory.CreateDirectory(TestPaths.OutputDir);
        string outputPath = Path.Combine(TestPaths.OutputDir, $"lens_turbo_gpu_{w}x{h}_s{request.Steps}_seed{seed}.bmp");
        ImagePostProcessor.SaveBmp(outputPath, rgb, w, h);
        _output.WriteLine($"Saved: {outputPath}");

        // Uniform-noise guard: a coherent image has strong per-row structure; noise does not.
        // Skipped for truncated (parity-probe) schedules — a 1-step partial denoise is legitimately noisy.
        if (steps >= 4)
            AssertNotUniformNoise(rgb, w, h);
    }

    /// <summary>Fails on the classic never-validated-port signature: per-pixel IID noise. Coherent images have neighboring-pixel correlation far above noise's ~0.</summary>
    private void AssertNotUniformNoise(byte[] rgb, int width, int height)
    {
        double sumA = 0, sumB = 0, sumAA = 0, sumBB = 0, sumAB = 0;
        long n = 0;
        for (int y = 0; y < height; y += 7)
        {
            for (int x = 0; x + 1 < width; x += 3)
            {
                int i = (y * width + x) * 3;
                double a = rgb[i], b = rgb[i + 3];
                sumA += a; sumB += b; sumAA += a * a; sumBB += b * b; sumAB += a * b;
                n++;
            }
        }
        double cov = sumAB - sumA * sumB / n;
        double varA = sumAA - sumA * sumA / n;
        double varB = sumBB - sumB * sumB / n;
        double corr = cov / Math.Sqrt(Math.Max(varA * varB, 1e-9));
        _output.WriteLine($"Neighbor-pixel correlation: {corr:F4} (noise ≈ 0, coherent ≳ 0.7)");
        Assert.True(corr > 0.5, $"output looks like uniform noise (neighbor corr {corr:F4})");
    }
}

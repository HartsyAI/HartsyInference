using System.Diagnostics;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>SSIM acceptance gate for SDXL: generates images via the C# pipeline and compares against reference PNGs produced by `tests/python-reference/dump_sdxl_reference_image.py` running diffusers with matching seed + steps + cfg.
///
/// <para><b>Strict gate (0.90) when reference noise is available</b> — the Python reference dump writes <c>init_noise_seed42.bin</c> alongside the PNGs; the test loads it as a F32 tensor and passes it via <see cref="TextToImageRequest.InitialNoise"/>, eliminating the RNG mismatch between PyTorch's <c>torch.Generator</c> and HartsyInference's <c>SeedGenerator</c>. With matched noise + matched weights, identical pipelines should produce SSIM ≥ 0.95; the 0.90 gate leaves headroom for F16 GEMM rounding through cuBLAS and small scheduler-step floating-point drift.</para>
///
/// <para><b>Loose gate (0.30) fallback</b> if the noise binary is absent — catches catastrophic regressions (NaN propagation, broken VAE, wrong scheduler) but lets RNG-driven differences pass.</para></summary>
[Trait("Category", "Integration")]
public sealed class SdxlSsimTests
{
    private const double StrictSsimThreshold = 0.90;
    private const double LooseSsimThreshold = 0.30;

    private static string ReferenceDir =>
        Environment.GetEnvironmentVariable("SDXL_REFERENCE_IMAGE_DIR")
        ?? Path.Combine(RepoRoot.Path, "tests", "python-reference", "sdxl_reference_images");

    private readonly ITestOutputHelper _output;
    public SdxlSsimTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Sdxl_GeneratesImage_WithinSsimThreshold()
    {
        string metaPath = Path.Combine(ReferenceDir, "meta.json");
        if (!File.Exists(metaPath))
        {
            _output.WriteLine($"SKIPPED: SDXL reference dir not found at {ReferenceDir}.");
            _output.WriteLine("Generate references first via: python tests/python-reference/dump_sdxl_reference_image.py --checkpoint <path> --output <ReferenceDir>");
            return;
        }
        if (!File.Exists(TestPaths.Sdxl.SingleFile))
        {
            _output.WriteLine($"SKIPPED: SDXL checkpoint not found at {TestPaths.Sdxl.SingleFile}.");
            return;
        }
        if (!File.Exists(TestPaths.Tokenizers.ClipVocab) || !File.Exists(TestPaths.Tokenizers.ClipMerges))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found.");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA not available.");
            return;
        }

        ReferenceMeta meta = JsonSerializer.Deserialize<ReferenceMeta>(File.ReadAllText(metaPath))
            ?? throw new InvalidDataException("Reference meta.json is empty or malformed.");

        if (meta.Prompts is null || meta.Prompts.Count == 0)
        {
            _output.WriteLine("SKIPPED: reference meta.json contains no prompts.");
            return;
        }

        string ptxDir = Path.Combine(RepoRoot.Path, "native", "cuda", "build");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(TestPaths.Sdxl.SingleFile);
        sw.Stop();
        _output.WriteLine($"Loaded checkpoint in {sw.ElapsedMilliseconds}ms.");

        using (loader)
        using (CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir))
        {
            UNet unet = new UNet(UNetConfig.SdxlBase);
            unet.LoadWeights(converted.UNet);

            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(converted.ClipL, "text_model");
            ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(converted.ClipG, "text_model");

            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sdxl);
            vaeDecoder.LoadWeights(converted.Vae);

            using ClipTokenizer tokenizer = new ClipTokenizer(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
            SdxlPipeline pipeline = new SdxlPipeline(backend, clipL, clipG, unet, vaeDecoder);

            string firstPrompt = meta.Prompts[0].Prompt;
            string rgbName = meta.Prompts[0].Rgb;
            if (string.IsNullOrEmpty(rgbName))
            {
                _output.WriteLine("SKIPPED: meta.json does not contain a 'rgb' entry — re-run dump_sdxl_reference_image.py to generate raw RGB binaries.");
                return;
            }
            string rgbPath = Path.Combine(ReferenceDir, rgbName);
            if (!File.Exists(rgbPath))
            {
                _output.WriteLine($"SKIPPED: reference RGB binary not found at {rgbPath}.");
                return;
            }
            byte[] referenceRgb = File.ReadAllBytes(rgbPath);
            int expectedLen = meta.Width * meta.Height * 3;
            if (referenceRgb.Length != expectedLen)
            {
                _output.WriteLine($"SKIPPED: reference RGB length {referenceRgb.Length} != {expectedLen} (W×H×3).");
                return;
            }

            int[] tokensL = tokenizer.Encode(firstPrompt);
            int[] tokensG = tokensL;
            string negativePrompt = meta.NegativePrompt ?? string.Empty;
            int[] negL = tokenizer.Encode(negativePrompt);
            int[] negG = negL;

            string noisePath = Path.Combine(ReferenceDir, $"init_noise_seed{meta.Seed}.bin");
            Tensor? injectedNoise = TryLoadNoise(noisePath, meta);
            double threshold = injectedNoise is not null ? StrictSsimThreshold : LooseSsimThreshold;
            _output.WriteLine(injectedNoise is not null
                ? $"Loaded reference noise from {Path.GetFileName(noisePath)}; using strict SSIM gate {StrictSsimThreshold}."
                : $"No reference noise binary at {noisePath}; falling back to loose SSIM gate {LooseSsimThreshold}.");

            TextToImageRequest req = new TextToImageRequest
            {
                Prompt = firstPrompt,
                NegativePrompt = negativePrompt,
                Width = meta.Width,
                Height = meta.Height,
                Steps = meta.Steps,
                CfgScale = meta.Cfg,
                Seed = meta.Seed,
                InitialNoise = injectedNoise,
            };

            sw.Restart();
            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                tokensL, negL, tokensG, negG,
                ClipTokenizer.FindEosPosition(tokensG),
                ClipTokenizer.FindEosPosition(negG),
                req,
                onProgress: p => _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
            sw.Stop();

            _output.WriteLine($"Generated in {sw.ElapsedMilliseconds}ms ({width}x{height}, seed={seed}).");

            Assert.Equal(meta.Width, width);
            Assert.Equal(meta.Height, height);

            double ssim = Ssim.Compute(referenceRgb, rgbData, width, height);
            _output.WriteLine($"SSIM vs reference: {ssim:F4} (threshold: {threshold})");
            Assert.True(ssim > threshold, $"SDXL SSIM {ssim:F4} below threshold {threshold} — possible regression in pipeline output.");

            pipeline.Dispose();
        }
    }

    private static unsafe Tensor? TryLoadNoise(string path, ReferenceMeta meta)
    {
        if (!File.Exists(path)) return null;
        byte[] raw = File.ReadAllBytes(path);
        int latentH = meta.Height / 8;
        int latentW = meta.Width / 8;
        long expectedFloats = (long)1 * 4 * latentH * latentW;
        if (raw.Length != expectedFloats * 4) return null;
        TensorShape shape = new TensorShape(1, 4, latentH, latentW);
        Tensor noise = new Tensor(shape, DType.F32);
        Span<byte> dst = new Span<byte>((void*)noise.DataPointer, raw.Length);
        raw.AsSpan().CopyTo(dst);
        return noise;
    }

    private sealed class ReferenceMeta
    {
        public string? Checkpoint { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public int Steps { get; set; }
        public float Cfg { get; set; }
        public int Seed { get; set; }
        public string? NegativePrompt { get; set; }
        public List<ReferencePrompt>? Prompts { get; set; }
    }

    private sealed class ReferencePrompt
    {
        public int Index { get; set; }
        public string Prompt { get; set; } = string.Empty;
        public string Image { get; set; } = string.Empty;
        public string Rgb { get; set; } = string.Empty;
    }
}

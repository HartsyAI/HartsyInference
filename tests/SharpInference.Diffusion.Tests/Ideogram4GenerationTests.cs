using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Cuda;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Schedulers;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tests.Common;
using SharpInference.Tokenizers;

namespace SharpInference.Diffusion.Tests;

/// <summary>End-to-end Ideogram 4 image generation. Skips cleanly when the checkpoint folder, Qwen3-VL tokenizer, or PTX dir are missing — none are bundled. Set <c>IDEOGRAM4_DIR</c> to a root folder with diffusers (<c>transformer/</c>, <c>unconditional_transformer/</c>, <c>text_encoder/</c>, <c>vae/</c>) or Comfy-Org (<c>diffusion_models/</c>, <c>text_encoders/</c>, <c>vae/</c>) layout.
///
/// NOTE: Ideogram 4 runs TWO 9.3B transformers concurrently (asymmetric CFG), so this needs a very-high-VRAM host. The VRAM probe skips below a conservative threshold rather than OOM-failing. The implementation is end-to-end wired; this test runs when the weights + VRAM are present.</summary>
public class Ideogram4GenerationTests
{
    private static string OutputDir => TestPaths.OutputDir;
    private readonly ITestOutputHelper _output;
    public Ideogram4GenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Ideogram4_Default20_Gpu_1024() =>
        RunGenerationTest("ideogram4_default20_1024", width: 1024, height: 1024, Ideogram4SamplerPreset.Default20);

    [Fact]
    public void Ideogram4_Turbo12_Gpu_1024() =>
        RunGenerationTest("ideogram4_turbo12_1024", width: 1024, height: 1024, Ideogram4SamplerPreset.Turbo12);

    private void RunGenerationTest(string outputName, int width, int height, Ideogram4SamplerPreset preset)
    {
        string dir = TestPaths.Ideogram4.Dir;
        if (!Directory.Exists(dir))
        {
            _output.WriteLine($"SKIPPED: Ideogram 4 folder not found: {dir}");
            _output.WriteLine("  Download from https://huggingface.co/Comfy-Org/Ideogram-4 (diffusion_models/, text_encoders/, vae/) and set IDEOGRAM4_DIR.");
            return;
        }
        if (!Directory.Exists(Path.Combine(dir, "vae")))
        {
            _output.WriteLine($"SKIPPED: vae/ subfolder missing under {dir}");
            return;
        }

        string vocab = TestPaths.Tokenizers.Qwen3Vocab;
        string merges = TestPaths.Tokenizers.Qwen3Merges;
        if (!File.Exists(vocab) || !File.Exists(merges))
        {
            _output.WriteLine($"SKIPPED: Qwen3 tokenizer not found ({vocab} / {merges}). Set QWEN3_TOKENIZER_DIR.");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(Ideogram4GenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        List<SafeTensorsLoader> allLoaders = [];
        try
        {
            _output.WriteLine("[1/5] Loading conditional + unconditional transformers...");
            (Dictionary<string, Tensor> condW, IReadOnlyList<SafeTensorsLoader> condL) =
                Ideogram4CheckpointConverter.LoadTransformer(dir, unconditional: false);
            allLoaders.AddRange(condL);
            (Dictionary<string, Tensor> uncondW, IReadOnlyList<SafeTensorsLoader> uncondL) =
                Ideogram4CheckpointConverter.LoadTransformer(dir, unconditional: true);
            allLoaders.AddRange(uncondL);

            _output.WriteLine("[2/5] Loading Qwen3-VL text encoder + VAE...");
            (Dictionary<string, Tensor> teW, IReadOnlyList<SafeTensorsLoader> teL) =
                Ideogram4CheckpointConverter.LoadTextEncoder(dir);
            allLoaders.AddRange(teL);
            (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vaeL) =
                Ideogram4CheckpointConverter.LoadVae(dir);
            allLoaders.AddRange(vaeL);

            _output.WriteLine("[3/5] Building models...");
            Ideogram4Config config = Ideogram4Config.V4;
            using Ideogram4Transformer conditional = new(config);
            conditional.LoadWeights(condW);
            using Ideogram4Transformer unconditional = new(config);
            unconditional.LoadWeights(uncondW);

            using LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen3_VL_8B);
            textEncoder.LoadWeights(teW);

            VaeDecoder vae = new(VaeConfig.Flux2);
            vae.LoadWeights(vaeW);

            _output.WriteLine("[4/5] Initializing CUDA backend (VRAM probe)...");
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            (nuint freeBytes, nuint totalBytes) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            // Two 9.3B DiTs (fp8 ≈ 9.5 GB each) + Qwen (~8 GB, freed before the loop) + VAE.
            const double MinRequiredGb = 22.0;
            if (freeGb < MinRequiredGb)
            {
                _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM (total {totalBytes / (1024.0 * 1024.0 * 1024.0):F1} GB); Ideogram 4 needs ≥{MinRequiredGb} GB for both transformers. The implementation is ready; run on a higher-VRAM host.");
                return;
            }

            using Qwen3Tokenizer tokenizer = new(vocab, merges, maxLength: config.MaxTextTokens);
            int[] promptTokens = tokenizer.EncodeChat("A photograph of an astronaut riding a horse on the moon");

            using Ideogram4Pipeline pipeline = new(backend, textEncoder, conditional, unconditional, vae, config);
            TextToImageRequest request = new()
            {
                Prompt = "A photograph of an astronaut riding a horse on the moon",
                NegativePrompt = "",
                Width = width,
                Height = height,
                Steps = preset.NumSteps,
                CfgScale = 7.0f,
                Seed = 42,
            };

            _output.WriteLine($"[5/5] Generating {width}x{height}, preset={preset.Name}, seed=42...");
            (byte[] rgb, int outW, int outH, int seed) = pipeline.GenerateFromTokens(
                promptTokens, request, preset,
                p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));

            Assert.Equal(width, outW);
            Assert.Equal(height, outH);
            Assert.Equal(width * height * 3, rgb.Length);

            Directory.CreateDirectory(OutputDir);
            string outputPath = Path.Combine(OutputDir, $"{outputName}_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgb, outW, outH);
            _output.WriteLine($"Saved: {outputPath} (total {sw.Elapsed.TotalSeconds:F1}s, seed={seed})");
        }
        finally
        {
            foreach (SafeTensorsLoader l in allLoaders) l.Dispose();
        }
    }
}

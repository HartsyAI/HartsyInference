using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>End-to-end ERNIE-Image (Baidu, Apache-2.0) image generation against the diffusers folder layout. Skips cleanly when the checkpoint folder, text encoder, or VAE are missing — these aren't bundled. Default paths are documented in <see cref="TestPaths.ErnieImage"/>.
///
/// The text encoder is Mistral3 ("ministral3" per <c>baidu/ERNIE-Image/text_encoder/config.json</c>), run via <see cref="ErnieImageLlamaTextEncoder"/> over <see cref="LlamaStyleEncoderConfig.Ministral3B"/>. Real prompts are tokenized with <see cref="ErnieTokenizer"/> when a <c>tokenizer.json</c> is found (set <c>ERNIE_TOKENIZER_JSON</c>, or drop it at <c>{modelDir}/tokenizer.json</c> / <c>{modelDir}/tokenizer/tokenizer.json</c>); otherwise a hardcoded token fallback keeps the wiring testable without the file.</summary>
public class ErnieImageGenerationTests
{
    private static string OutputDir => TestPaths.OutputDir;
    private readonly ITestOutputHelper _output;
    public ErnieImageGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ErnieImage_V1_Gpu_512_NoCfg() =>
        RunGenerationTest(TestPaths.ErnieImage.V1Dir, "ernie_image_v1_512_nocfg",
            width: 512, height: 512, steps: 25, cfgScale: 1.0f);

    [Fact]
    public void ErnieImage_V1_Gpu_512_Cfg() =>
        RunGenerationTest(TestPaths.ErnieImage.V1Dir, "ernie_image_v1_512_cfg",
            width: 512, height: 512, steps: 28, cfgScale: 4.0f);

    [Fact]
    public void ErnieImage_V1Turbo_Gpu_512_8Steps() =>
        RunGenerationTest(TestPaths.ErnieImage.V1TurboDir, "ernie_image_v1_turbo_512_8steps",
            width: 512, height: 512, steps: 8, cfgScale: 1.0f);

    private void RunGenerationTest(string modelDir, string outputName, int width, int height, int steps, float cfgScale)
    {
        if (!Directory.Exists(modelDir))
        {
            _output.WriteLine($"SKIPPED: ERNIE-Image folder not found: {modelDir}");
            _output.WriteLine($"  Download `diffusion_models/`, `text_encoders/`, `vae/` shards from https://huggingface.co/Comfy-Org/ERNIE-Image (or `transformer/`, `text_encoder/`, `vae/` from https://huggingface.co/baidu/ERNIE-Image)");
            _output.WriteLine($"  or set ERNIE_IMAGE_V1_DIR / ERNIE_IMAGE_V1_TURBO_DIR to override.");
            return;
        }

        // Accept either diffusers (`transformer/`, `text_encoder/`, `vae/`) or Comfy-Org
        // (`diffusion_models/`, `text_encoders/`, `vae/`) folder layouts. The converter handles both.
        string transformerSubdir = Directory.Exists(Path.Combine(modelDir, "transformer"))
            ? Path.Combine(modelDir, "transformer")
            : Path.Combine(modelDir, "diffusion_models");
        string textEncoderSubdir = Directory.Exists(Path.Combine(modelDir, "text_encoder"))
            ? Path.Combine(modelDir, "text_encoder")
            : Path.Combine(modelDir, "text_encoders");
        string vaeSubdir = Path.Combine(modelDir, "vae");
        if (!Directory.Exists(transformerSubdir))
        {
            _output.WriteLine($"SKIPPED: neither transformer/ nor diffusion_models/ subfolder found in: {modelDir}");
            return;
        }
        if (!Directory.Exists(vaeSubdir))
        {
            _output.WriteLine($"SKIPPED: vae/ subfolder missing: {vaeSubdir}");
            return;
        }
        if (!Directory.Exists(textEncoderSubdir))
        {
            _output.WriteLine($"SKIPPED: neither text_encoder/ nor text_encoders/ subfolder found in: {modelDir}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(ErnieImageGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Load + convert transformer shards ──
        _output.WriteLine($"[1/6] Loading transformer shards from {transformerSubdir}...");
        (ErnieImageCheckpointConverter.ConvertedWeights converted, IReadOnlyList<SafeTensorsLoader> transformerLoaders) =
            ErnieImageCheckpointConverter.LoadAndConvert(modelDir);
        _output.WriteLine($"  Loaded {converted.Transformer.Count} keys in {sw.ElapsedMilliseconds}ms");

        try
        {
            // ── 2. Load VAE shards ──
            _output.WriteLine($"[2/6] Loading VAE shards from {vaeSubdir}...");
            sw.Restart();
            (Dictionary<string, Tensor> vaeWeights, IReadOnlyList<SafeTensorsLoader> vaeLoaders) =
                ErnieImageCheckpointConverter.LoadVae(modelDir);
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms ({vaeWeights.Count} keys)");

            try
            {
                // ── 3. Load text encoder shards ──
                _output.WriteLine($"[3/6] Loading text encoder shards from {textEncoderSubdir}...");
                sw.Restart();
                (Dictionary<string, Tensor> teWeights, IReadOnlyList<SafeTensorsLoader> teLoaders) =
                    ErnieImageCheckpointConverter.LoadTextEncoder(modelDir);
                _output.WriteLine($"  Text encoder loaded in {sw.ElapsedMilliseconds}ms ({teWeights.Count} keys)");

                try
                {
                    // ── 4. Build models ──
                    _output.WriteLine($"[4/6] Building transformer + VAE + text encoder...");
                    sw.Restart();

                    ErnieImageConfig config = ErnieImageConfig.V1;
                    using ErnieImageTransformer transformer = new(config);
                    transformer.LoadWeights(converted.Transformer);

                    VaeDecoder vae = new(VaeConfig.Flux2);
                    vae.LoadWeights(vaeWeights);

                    // ERNIE-Image text encoder is Mistral3 (a 3072-hidden Ministral 3B variant) per
                    // baidu/ERNIE-Image/text_encoder/config.json (model_type "ministral3"). Wrap in
                    // ErnieImageLlamaTextEncoder which exposes hidden_states[-2] to match diffusers'
                    // `output.hidden_states[-2]` convention from pipeline_ernie_image.py.
                    LlamaStyleEncoder llama = new(LlamaStyleEncoderConfig.Ministral3B);
                    llama.LoadWeights(teWeights);
                    IErnieTextEncoder textEncoder = new ErnieImageLlamaTextEncoder(llama)
                        .WithHiddenSize(LlamaStyleEncoderConfig.Ministral3B.HiddenSize);
                    _output.WriteLine($"  Models ready in {sw.ElapsedMilliseconds}ms (using LlamaStyleEncoder Ministral3B)");

                    // ── 5. Initialize backend + preload weights ──
                    // Only preload transformer; VAE uploads lazily during decode (PHASE_3_DEVIATIONS #18).
                    // ERNIE (13.8 GB transformer + 334 MB VAE) exceeds 16 GB at full preload when SwarmUI holds 2.4 GB.
                    _output.WriteLine($"[5/6] Initializing CUDA backend + preloading transformer...");
                    sw.Restart();
                    using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
                    (nuint freeBytes, nuint totalBytes) = backend.Context.GetMemoryInfo();
                    double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
                    const double MinRequiredGb = 14.0;
                    if (freeGb < MinRequiredGb)
                    {
                        _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM (total {totalBytes / (1024.0 * 1024.0 * 1024.0):F1} GB); need ≥{MinRequiredGb} GB to fit ERNIE-Image FP16 transformer (~13.8 GB) + VAE + text encoder. Free up GPU memory or use a larger card. The implementation is end-to-end ready; this test will run when sufficient VRAM is available.");
                        return;
                    }
                    backend.PreloadWeights(transformer.EnumerateWeights());
                    _output.WriteLine($"  Backend ready in {sw.ElapsedMilliseconds}ms (device: {backend.Capabilities.Name})");

                    using ErnieImagePipeline pipeline = new(backend, textEncoder, transformer, vae, config);

                    TextToImageRequest request = new()
                    {
                        Prompt = "A photograph of an astronaut riding a horse",
                        NegativePrompt = "",
                        Width = width,
                        Height = height,
                        Steps = steps,
                        CfgScale = cfgScale,
                        Seed = 42,
                    };

                    // ── 6. Generate ──
                    _output.WriteLine($"\n[6/6] Generating {width}x{height}, {steps} steps, cfg={cfgScale}, seed=42...");

                    // Real tokenization when a tokenizer.json is available; hardcoded fallback keeps the
                    // wiring exercisable without it (token-id parity is validation-gated on the real file).
                    int[] promptTokens;
                    int[] negTokens;
                    string? tokenizerJson = FindTokenizerJson(modelDir);
                    if (tokenizerJson is not null)
                    {
                        _output.WriteLine($"  Tokenizing with {tokenizerJson}");
                        using ErnieTokenizer ernieTokenizer = new(tokenizerJson);
                        promptTokens = ernieTokenizer.Encode(request.Prompt);
                        negTokens = ernieTokenizer.Encode(request.NegativePrompt ?? "");
                    }
                    else
                    {
                        _output.WriteLine("  tokenizer.json not found — using hardcoded fallback tokens (set ERNIE_TOKENIZER_JSON for real prompts)");
                        promptTokens = [1, 2, 3, 4, 5, 6, 7, 8];
                        negTokens = [1, 2, 3, 4, 5, 6, 7, 8];
                    }

                    Stopwatch genSw = Stopwatch.StartNew();
                    (byte[] rgb, int outW, int outH, int seed) = pipeline.GenerateFromTokens(
                        promptTokens, negTokens, promptTokens.Length, negTokens.Length, request,
                        progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));
                    genSw.Stop();
                    _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalSeconds:F1}s (seed={seed})");

                    Assert.Equal(width, outW);
                    Assert.Equal(height, outH);
                    Assert.Equal(width * height * 3, rgb.Length);

                    Directory.CreateDirectory(OutputDir);
                    string outputPath = Path.Combine(OutputDir, $"{outputName}_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
                    ImagePostProcessor.SaveBmp(outputPath, rgb, outW, outH);
                    _output.WriteLine($"  Saved: {outputPath}");

                    totalSw.Stop();
                    _output.WriteLine($"\nTotal: {totalSw.Elapsed.TotalSeconds:F1}s");
                }
                finally
                {
                    foreach (SafeTensorsLoader l in teLoaders) l.Dispose();
                }
            }
            finally
            {
                foreach (SafeTensorsLoader l in vaeLoaders) l.Dispose();
            }
        }
        finally
        {
            foreach (SafeTensorsLoader l in transformerLoaders) l.Dispose();
        }
    }

    /// <summary>Resolves the ERNIE tokenizer.json: <c>ERNIE_TOKENIZER_JSON</c> env var first, then <c>{modelDir}/tokenizer.json</c> and <c>{modelDir}/tokenizer/tokenizer.json</c> (the diffusers folder layout).</summary>
    private static string? FindTokenizerJson(string modelDir)
    {
        string? env = Environment.GetEnvironmentVariable("ERNIE_TOKENIZER_JSON");
        if (!string.IsNullOrEmpty(env) && File.Exists(env))
            return env;
        string direct = Path.Combine(modelDir, "tokenizer.json");
        if (File.Exists(direct))
            return direct;
        string nested = Path.Combine(modelDir, "tokenizer", "tokenizer.json");
        return File.Exists(nested) ? nested : null;
    }
}

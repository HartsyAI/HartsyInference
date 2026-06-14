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

/// <summary>End-to-end Chroma image generation against the <c>lodestones/Chroma</c> single-file safetensors
/// (BFL-style native format) plus a separate T5-XXL text encoder + Flux VAE.
///
/// Skips cleanly when any of the required artifacts is missing. Mirrors the AuraFlow tests' "look up via
/// <see cref="TestPaths"/>, skip with a hint" pattern.</summary>
public class ChromaGenerationTests
{
    private static string OutputDir => TestPaths.OutputDir;
    private readonly ITestOutputHelper _output;
    public ChromaGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Chroma_V1_Gpu_512_Cfg5() =>
        RunGenerationTest("chroma_v1_512_cfg5", width: 512, height: 512, steps: 35, cfgScale: 5.0f);

    [Fact]
    public void Chroma_V1_Gpu_512_NoCfg() =>
        RunGenerationTest("chroma_v1_512_nocfg", width: 512, height: 512, steps: 25, cfgScale: 1.0f);

    private void RunGenerationTest(string outputName, int width, int height, int steps, float cfgScale)
    {
        string ckpt = TestPaths.Chroma.V1;
        if (!File.Exists(ckpt))
        {
            _output.WriteLine($"SKIPPED: Chroma checkpoint not found: {ckpt}");
            _output.WriteLine($"  Download from https://huggingface.co/lodestones/Chroma1-HD or set CHROMA_V1_PATH.");
            return;
        }

        if (!File.Exists(TestPaths.Chroma.T5XxlSpiece))
        {
            _output.WriteLine($"SKIPPED: T5-XXL SentencePiece tokenizer not found: {TestPaths.Chroma.T5XxlSpiece}");
            return;
        }

        // Chroma needs a T5-XXL text encoder. The Chroma single-file does NOT bundle T5, so we
        // extract it from another bundled safetensors — by default the SD3.5 Medium FP8 file we already
        // downloaded for the Sd35 tests bundles T5-XXL under `text_encoders.t5xxl.transformer.*`.
        string t5SourcePath = TestPaths.Chroma.T5XxlSource;
        if (!File.Exists(t5SourcePath))
        {
            _output.WriteLine($"SKIPPED: T5-XXL source not found: {t5SourcePath}");
            _output.WriteLine($"  Default expects an SD3.5 single-file at {TestPaths.Sd35.Medium}; set CHROMA_T5XXL_SOURCE to override.");
            return;
        }

        // Chroma uses the 16-channel Flux VAE. We pull it from the Flux Dev / Krea checkpoint by default.
        string vaePath = TestPaths.Vae.FluxVaeSource;
        if (!File.Exists(vaePath))
        {
            _output.WriteLine($"SKIPPED: Flux VAE source not found: {vaePath}");
            _output.WriteLine($"  Set FLUX_VAE_SOURCE_PATH or place a Flux Dev checkpoint at the default path.");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(ChromaGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        _output.WriteLine($"[1/7] Loading + converting Chroma checkpoint: {Path.GetFileName(ckpt)}");
        (ChromaCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader transformerLoader) =
            ChromaCheckpointConverter.LoadAndConvert(ckpt);
        _output.WriteLine($"  Converted in {sw.ElapsedMilliseconds}ms ({converted.Transformer.Count} keys)");

        using (transformerLoader)
        {
            Dictionary<string, Tensor> transformerWeights = converted.Transformer;

            // ── Load T5-XXL by extracting from the SD3.5 single-file (it bundles T5 under text_encoders.t5xxl.transformer.*) ──
            _output.WriteLine($"[2/7] Extracting T5-XXL from {Path.GetFileName(t5SourcePath)}...");
            sw.Restart();
            (Sd3CheckpointConverter.ConvertedWeights sd3Bundle, SafeTensorsLoader t5Loader) =
                Sd3CheckpointConverter.LoadAndConvert(t5SourcePath);
            try
            {
                Dictionary<string, Tensor> t5Weights = sd3Bundle.T5;
                if (t5Weights.Count == 0)
                {
                    _output.WriteLine($"SKIPPED: SD3 source had no T5 weights — does {Path.GetFileName(t5SourcePath)} include the bundled T5-XXL?");
                    return;
                }
                _output.WriteLine($"  {t5Weights.Count} T5 tensors extracted in {sw.ElapsedMilliseconds}ms");

                // ── Load Flux VAE from a Flux Dev / Krea checkpoint ──
                _output.WriteLine($"[3/7] Loading Flux VAE: {Path.GetFileName(vaePath)}");
                sw.Restart();
                using SafeTensorsLoader vaeLoader = new();
                vaeLoader.Load(vaePath);
                FluxCheckpointConverter.ConvertedWeights vaeConverted =
                    FluxCheckpointConverter.Convert(vaeLoader.GetAllTensors());
                Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(vaeConverted.Vae);
                _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms ({vaeF32.Count} keys)");

                // ── Tokenize ──
                _output.WriteLine($"[4/7] Tokenizing prompt...");
                using T5Tokenizer tokenizer = new(TestPaths.Chroma.T5XxlSpiece, maxLength: 512);
                string prompt = "A photograph of an astronaut riding a horse";
                string negPrompt = "low quality, blurry, deformed";
                int[] promptTokens = tokenizer.Encode(prompt);
                int[] negTokens = tokenizer.Encode(negPrompt);
                int[] promptMask = T5Tokenizer.CreateAttentionMask(promptTokens);
                int[] negMask = T5Tokenizer.CreateAttentionMask(negTokens);

                // ── Build T5-XXL encoder ──
                _output.WriteLine($"[5/7] Building T5-XXL text encoder...");
                sw.Restart();
                T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl);
                t5.LoadWeights(t5Weights);
                _output.WriteLine($"  T5 ready in {sw.ElapsedMilliseconds}ms");

                // ── Build Chroma transformer ──
                _output.WriteLine($"[6/7] Building Chroma transformer...");
                sw.Restart();
                ChromaConfig config = ChromaConfig.V1;
                using ChromaTransformer transformer = new(config);
                transformer.LoadWeights(transformerWeights);
                _output.WriteLine($"  Transformer ready in {sw.ElapsedMilliseconds}ms");

                // ── Build Flux VAE decoder ──
                VaeDecoder vae = new(VaeConfig.Chroma);
                vae.LoadWeights(vaeF32);

                // ── Initialize CUDA backend ──
                // Don't preload everything at once — T5 (5GB) + Chroma transformer (9GB) + VAE (334MB)
                // exceeds 12GB. The pipeline frees the transformer + T5 before VAE decode (PHASE_3 #18 / #33),
                // and T5 auto-uploads on first use; preloading just the transformer at start is the right
                // staging for a 12 GB card.
                _output.WriteLine($"[7/7] Initializing CUDA backend (pipeline handles weight staging)...");
                sw.Restart();
                using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
                // NB: the pipeline itself does T5-preload → encode → free → transformer-preload →
                // denoise → free → vae-preload → decode. Pre-uploading the transformer here would
                // collide with the pipeline's T5 upload on a 12 GB card.
                _output.WriteLine($"  Backend ready in {sw.ElapsedMilliseconds}ms (device: {backend.Capabilities.Name})");

                using ChromaPipeline pipeline = new(backend, t5, transformer, vae, config);

                TextToImageRequest request = new()
                {
                    Prompt = prompt,
                    NegativePrompt = negPrompt,
                    Width = width,
                    Height = height,
                    Steps = steps,
                    CfgScale = cfgScale,
                    Seed = 42,
                };

                _output.WriteLine($"\nGenerating {width}x{height}, {steps} steps, cfg={cfgScale}, seed=42...");
                Stopwatch genSw = Stopwatch.StartNew();
                (byte[] rgb, int outW, int outH, int seed) = pipeline.GenerateFromTokens(
                    promptTokens, negTokens, promptMask, negMask, request,
                    progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));
                genSw.Stop();
                _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalSeconds:F1}s (seed={seed})");

                Assert.Equal(width, outW);
                Assert.Equal(height, outH);
                Assert.Equal(width * height * 3, rgb.Length);
                ValidateImageNotDegenerate(rgb);

                Directory.CreateDirectory(OutputDir);
                string outputPath = Path.Combine(OutputDir, $"{outputName}_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
                ImagePostProcessor.SaveBmp(outputPath, rgb, outW, outH);
                _output.WriteLine($"  Saved: {outputPath}");

                totalSw.Stop();
                _output.WriteLine($"\nTotal: {totalSw.Elapsed.TotalSeconds:F1}s");
            }
            finally
            {
                t5Loader.Dispose();
            }
        }
    }

    private void ValidateImageNotDegenerate(byte[] rgbData)
    {
        int nonZero = 0, nonFF = 0;
        foreach (byte b in rgbData)
        {
            if (b != 0) nonZero++;
            if (b != 255) nonFF++;
        }
        float nzPct = nonZero / (float)rgbData.Length * 100;
        float nffPct = nonFF / (float)rgbData.Length * 100;
        _output.WriteLine($"  Non-zero: {nzPct:F1}%, Non-255: {nffPct:F1}%");
        Assert.True(nzPct > 10, "Image appears all-black");
        Assert.True(nffPct > 10, "Image appears all-white");
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            DType dt = kvp.Value.DType;
            f32[kvp.Key] = (dt == DType.F16 || dt == DType.BF16) ? kvp.Value.CastTo(DType.F32) : kvp.Value;
        }
        return f32;
    }
}

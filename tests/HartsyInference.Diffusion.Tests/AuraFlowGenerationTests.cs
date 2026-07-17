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

/// <summary>End-to-end AuraFlow image generation against the <c>fal/AuraFlow-v0.3</c> single-file
/// transformer + a separate Pile-T5-XL text encoder + the SDXL VAE.
///
/// Skips cleanly when any of the required artifacts is missing (the user must download them and
/// either place them at the default paths or set the corresponding env vars).
/// Default paths are documented in <see cref="TestPaths.AuraFlow"/>.</summary>
[Trait("Category", "Integration")]
public class AuraFlowGenerationTests
{
    private static string OutputDir => TestPaths.OutputDir;
    private readonly ITestOutputHelper _output;
    public AuraFlowGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void AuraFlow_V03_Gpu_512_NoCfg() =>
        RunGenerationTest("auraflow_v03_512_nocfg", width: 512, height: 512, steps: 25, cfgScale: 1.0f);

    [Fact]
    public void AuraFlow_V03_Gpu_512_Cfg() =>
        RunGenerationTest("auraflow_v03_1024_cfg", width: 1024, height: 1024, steps: 28, cfgScale: 3.5f);

    private void RunGenerationTest(string outputName, int width, int height, int steps, float cfgScale)
    {
        // Prefer the FP8-bundled file from `calcuis/aura` (transformer + Pile-T5-XL + VAE in one
        // safetensors). Fall back to the BF16 transformer-only file from `fal/AuraFlow-v0.3`
        // (which then needs separate text_encoder/ shards + a standalone VAE).
        bool bundled;
        string ckpt;
        if (File.Exists(TestPaths.AuraFlow.V03Fp8))
        {
            ckpt = TestPaths.AuraFlow.V03Fp8;
            bundled = true;
        }
        else if (File.Exists(TestPaths.AuraFlow.V03))
        {
            ckpt = TestPaths.AuraFlow.V03;
            bundled = false;
        }
        else
        {
            _output.WriteLine($"SKIPPED: no AuraFlow checkpoint found at {TestPaths.AuraFlow.V03Fp8} or {TestPaths.AuraFlow.V03}");
            _output.WriteLine($"  FP8 bundled: https://huggingface.co/calcuis/aura/blob/main/aura_flow_0.3_fp8_scaled.safetensors (9.66 GB)");
            _output.WriteLine($"  BF16 transformer-only: https://huggingface.co/fal/AuraFlow-v0.3/blob/main/aura_flow_0.3.safetensors (16.5 GB)");
            return;
        }

        if (!File.Exists(TestPaths.AuraFlow.PileT5XlSpiece))
        {
            _output.WriteLine($"SKIPPED: T5 SentencePiece tokenizer not found: {TestPaths.AuraFlow.PileT5XlSpiece}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(AuraFlowGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        // ── Convert AuraFlow transformer single-file ──
        _output.WriteLine($"[1/7] Loading + converting AuraFlow checkpoint: {Path.GetFileName(ckpt)}");
        (AuraFlowCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader transformerLoader) =
            AuraFlowCheckpointConverter.LoadAndConvert(ckpt);
        _output.WriteLine($"  Converted in {sw.ElapsedMilliseconds}ms ({converted.Transformer.Count} keys)");

        using (transformerLoader)
        {
            Dictionary<string, Tensor> transformerWeights = converted.Transformer;
            Dictionary<string, Tensor> t5Weights;
            Dictionary<string, Tensor> vaeWeights;

            if (bundled)
            {
                // FP8 single-file: T5 + VAE buckets are populated by the converter.
                _output.WriteLine($"[2/7] Bundled file — extracted T5 ({converted.T5.Count} keys) + VAE ({converted.Vae.Count} keys) from same safetensors.");
                t5Weights = converted.T5;
                vaeWeights = converted.Vae;
                if (t5Weights.Count == 0)
                {
                    _output.WriteLine($"SKIPPED: bundled file but T5 bucket is empty — checkpoint may not bundle Pile-T5-XL.");
                    return;
                }
                if (vaeWeights.Count == 0)
                {
                    // Fall back to the standalone aura_vae.safetensors if the bundled file lacks VAE keys.
                    string vaePath = TestPaths.AuraFlow.Vae;
                    if (!File.Exists(vaePath))
                    {
                        _output.WriteLine($"SKIPPED: bundled file lacks VAE and no fallback at {vaePath}");
                        return;
                    }
                    _output.WriteLine($"  Falling back to standalone VAE: {Path.GetFileName(vaePath)}");
                    using SafeTensorsLoader vLoader = new();
                    vLoader.Load(vaePath);
                    vaeWeights = vLoader.GetAllTensors();
                }
            }
            else
            {
                // BF16 transformer-only file: load Pile-T5-XL shards from a separate directory + standalone VAE.
                string pileT5Dir = TestPaths.AuraFlow.PileT5XlDir;
                if (!Directory.Exists(pileT5Dir))
                {
                    _output.WriteLine($"SKIPPED: BF16 path requires Pile-T5-XL shards at {pileT5Dir}");
                    return;
                }
                _output.WriteLine($"[2/7] Loading Pile-T5-XL from {pileT5Dir}...");
                sw.Restart();
                t5Weights = new Dictionary<string, Tensor>();
                List<SafeTensorsLoader> tLoaders = new();
                foreach (string shard in Directory.GetFiles(pileT5Dir, "*.safetensors").OrderBy(p => p))
                {
                    SafeTensorsLoader shardLoader = new();
                    shardLoader.Load(shard);
                    foreach (KeyValuePair<string, Tensor> kvp in shardLoader.GetAllTensors())
                        t5Weights[kvp.Key] = kvp.Value;
                    tLoaders.Add(shardLoader);
                }
                _output.WriteLine($"  {t5Weights.Count} T5 tensors loaded across {tLoaders.Count} shard(s) in {sw.ElapsedMilliseconds}ms");

                string vaePath = TestPaths.AuraFlow.Vae;
                if (!File.Exists(vaePath))
                {
                    _output.WriteLine($"SKIPPED: standalone VAE not found at {vaePath}");
                    foreach (SafeTensorsLoader l in tLoaders) l.Dispose();
                    return;
                }
                _output.WriteLine($"[3/7] Loading standalone VAE: {Path.GetFileName(vaePath)}");
                using SafeTensorsLoader vaeLoader2 = new();
                vaeLoader2.Load(vaePath);
                vaeWeights = vaeLoader2.GetAllTensors();
            }

            try
            {
                _output.WriteLine($"  T5 keys: {t5Weights.Count}, VAE keys: {vaeWeights.Count}");

                // ── Tokenize ──
                _output.WriteLine($"[4/7] Tokenizing prompt...");
                using T5Tokenizer tokenizer = new(TestPaths.AuraFlow.PileT5XlSpiece);
                string prompt = "A photograph of an astronaut riding a horse";
                string negPrompt = "";
                int[] promptTokens = tokenizer.Encode(prompt);
                int[] negTokens = tokenizer.Encode(negPrompt);
                int[] promptMask = CreateAttentionMask(promptTokens);
                int[] negMask = CreateAttentionMask(negTokens);

                // ── Build Pile-T5-XL encoder ──
                _output.WriteLine($"[5/7] Building Pile-T5-XL text encoder...");
                sw.Restart();
                T5TextEncoder t5 = new(T5TextEncoderConfig.PileT5Xl);
                t5.LoadWeights(t5Weights);
                _output.WriteLine($"  T5 ready in {sw.ElapsedMilliseconds}ms");

                // ── Build AuraFlow transformer ──
                _output.WriteLine($"[6/7] Building AuraFlow transformer...");
                sw.Restart();
                AuraFlowConfig config = AuraFlowConfig.V03;
                using AuraFlowTransformer transformer = new(config);
                transformer.LoadWeights(transformerWeights);
                _output.WriteLine($"  Transformer ready in {sw.ElapsedMilliseconds}ms");

                // ── Build VAE decoder (AuraFlow's bundled VAE OR standalone aura_vae.safetensors) ──
                VaeDecoder vae = new(VaeConfig.AuraFlow);
                vae.LoadWeights(CastWeightsToF32(vaeWeights));

                // ── Initialize CUDA backend + preload only the transformer ──
                // T5 (~3.7 GB FP16, ~1.9 GB FP8) auto-uploads on first use; VAE preloads after pipeline
                // frees the transformer (PHASE_3 #18 / #33). Preloading everything would OOM 12 GB cards
                // sharing VRAM with SwarmUI etc.
                _output.WriteLine($"[7/7] Initializing CUDA backend + preloading transformer...");
                sw.Restart();
                using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
                // AuraFlow 6.8B fp8 + Pile-T5-XL exceeds 24 GB if every weight keeps a resident F16 cast.
                // Low-VRAM mode: transient per-GEMM weight cast (one at a time, freed) keeps it ~9 GB.
                backend.CacheWeightCasts = false;
                // (Preload would re-introduce the resident cast, so it stays off here.)
                // backend.PreloadWeights(transformer.EnumerateWeights());
                _output.WriteLine($"  Backend ready in {sw.ElapsedMilliseconds}ms (device: {backend.Capabilities.Name})");

                using AuraFlowPipeline pipeline = new(backend, t5, transformer, vae, config);

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
                // tLoaders are scoped inside the BF16 branch and disposed there if used.
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

    private static int[] CreateAttentionMask(int[] tokenIds)
    {
        int[] mask = new int[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++)
            mask[i] = tokenIds[i] != T5Tokenizer.PadTokenId ? 1 : 0;
        return mask;
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

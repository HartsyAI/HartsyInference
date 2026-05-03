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
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tests.Common;
using SharpInference.Tokenizers;

namespace SharpInference.Diffusion.Tests;

/// <summary>End-to-end AuraFlow image generation against the <c>fal/AuraFlow-v0.3</c> single-file
/// transformer + a separate Pile-T5-XL text encoder + the SDXL VAE.
///
/// Skips cleanly when any of the required artifacts is missing (the user must download them and
/// either place them at the default paths or set the corresponding env vars).
/// Default paths are documented in <see cref="TestPaths.AuraFlow"/>.</summary>
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
        RunGenerationTest("auraflow_v03_512_cfg", width: 512, height: 512, steps: 28, cfgScale: 3.5f);

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
            // Cast non-F32 to F32 for the diff-friendly first-run baseline. (Production runs would skip
            // this and let cuBLAS handle F16/FP8 directly via backend.Linear.)
            Dictionary<string, Tensor> transformerWeights = converted.Transformer;

            // ── Load Pile-T5-XL shards (diffusers folder layout). The fal/AuraFlow-v0.3 repo ships
            //    text_encoder/ as multiple safetensors shards; we load each and merge into one dict. ──
            _output.WriteLine($"[2/7] Loading Pile-T5-XL from {pileT5Dir}...");
            sw.Restart();
            Dictionary<string, Tensor> t5Weights = new();
            List<SafeTensorsLoader> t5Loaders = new();
            try
            {
                foreach (string shard in Directory.GetFiles(pileT5Dir, "*.safetensors").OrderBy(p => p))
                {
                    SafeTensorsLoader shardLoader = new();
                    shardLoader.Load(shard);
                    foreach (KeyValuePair<string, Tensor> kvp in shardLoader.GetAllTensors())
                        t5Weights[kvp.Key] = kvp.Value;
                    t5Loaders.Add(shardLoader);
                }
                if (t5Weights.Count == 0)
                {
                    _output.WriteLine($"SKIPPED: no .safetensors shards found in {pileT5Dir}");
                    return;
                }
                _output.WriteLine($"  {t5Weights.Count} tensors loaded across {t5Loaders.Count} shard(s) in {sw.ElapsedMilliseconds}ms");

                // ── Load SDXL VAE ──
                _output.WriteLine($"[3/7] Loading SDXL VAE: {Path.GetFileName(vaePath)}");
                sw.Restart();
                using SafeTensorsLoader vaeLoader = new();
                vaeLoader.Load(vaePath);
                Dictionary<string, Tensor> vaeWeights = vaeLoader.GetAllTensors();
                _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms ({vaeWeights.Count} keys)");

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

                // ── Build SDXL VAE decoder ──
                VaeDecoder vae = new(VaeConfig.Sdxl);
                vae.LoadWeights(CastWeightsToF32(vaeWeights));

                // ── Initialize CUDA backend + preload ──
                _output.WriteLine($"[7/7] Initializing CUDA backend + preloading weights...");
                sw.Restart();
                using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
                backend.PreloadWeights(t5.EnumerateWeights());
                backend.PreloadWeights(transformer.EnumerateWeights());
                backend.PreloadWeights(vae.EnumerateWeights());
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
                foreach (SafeTensorsLoader l in t5Loaders) l.Dispose();
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

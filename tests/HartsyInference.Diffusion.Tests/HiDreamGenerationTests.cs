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

/// <summary>End-to-end HiDream i1 generation. Skips cleanly when any of the four text encoders, the
/// transformer, the VAE, the tokenizer assets, or the PTX directory is missing. Also skips when free
/// VRAM is below the threshold required for the FP16 transformer.</summary>
public sealed class HiDreamGenerationTests
{
    private static string OutputDir => TestPaths.OutputDir;
    private readonly ITestOutputHelper _output;

    public HiDreamGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void HiDream_I1_Full_Gpu_1024_Cfg() =>
        RunGenerationTest("hidream_i1_1024_cfg", width: 1024, height: 1024, steps: 50, cfgScale: 5.0f);

    [Fact]
    public void HiDream_I1_Dev_Gpu_1024_NoCfg() =>
        RunGenerationTest("hidream_i1_dev_1024_nocfg", width: 1024, height: 1024, steps: 25, cfgScale: 1.0f);

    private void RunGenerationTest(string outputName, int width, int height, int steps, float cfgScale)
    {
        if (!File.Exists(TestPaths.HiDream.Transformer) ||
            !File.Exists(TestPaths.HiDream.Vae) ||
            !File.Exists(TestPaths.HiDream.ClipL) ||
            !File.Exists(TestPaths.HiDream.ClipG) ||
            !File.Exists(TestPaths.HiDream.T5) ||
            !File.Exists(TestPaths.HiDream.Llama))
        {
            _output.WriteLine($"SKIPPED: HiDream artifacts missing — needs transformer, vae, clip_l, clip_g, t5xxl, llama_3_1.");
            _output.WriteLine($"  Download from https://huggingface.co/HiDream-ai/HiDream-I1-Full or HiDream-I1-Dev");
            return;
        }
        if (!File.Exists(TestPaths.Tokenizers.ClipVocab) || !File.Exists(TestPaths.Tokenizers.ClipMerges) ||
            !File.Exists(TestPaths.Tokenizers.T5XxlSpiece))
        {
            _output.WriteLine($"SKIPPED: missing tokenizer assets (CLIP / T5).");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(HiDreamGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        _output.WriteLine($"[1/7] Loading + converting transformer: {Path.GetFileName(TestPaths.HiDream.Transformer)}");
        (HiDreamCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            HiDreamCheckpointConverter.LoadAndConvert(TestPaths.HiDream.Transformer);
        _output.WriteLine($"  Loaded {converted.Transformer.Count} transformer keys in {sw.ElapsedMilliseconds}ms (fp8={converted.IsFp8Mix})");

        try
        {
            using (loader)
            {
                Dictionary<string, Tensor> transformerWeights = CastWeightsToF32(converted.Transformer);

                sw.Restart();
                HiDreamConfig config = HiDreamConfig.Full;
                using HiDreamTransformer transformer = new(config);
                transformer.LoadWeights(transformerWeights);
                _output.WriteLine($"[2/7] Transformer loaded in {sw.ElapsedMilliseconds}ms");

                sw.Restart();
                VaeDecoder vae = new(VaeConfig.Flux);
                vae.LoadWeights(CastWeightsToF32(LoadStandalone(TestPaths.HiDream.Vae)));
                _output.WriteLine($"[3/7] VAE loaded in {sw.ElapsedMilliseconds}ms");

                sw.Restart();
                ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
                clipL.LoadWeights(LoadStandalone(TestPaths.HiDream.ClipL), "text_model");
                ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
                clipG.LoadWeights(LoadStandalone(TestPaths.HiDream.ClipG), "text_model");
                T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl);
                t5.LoadWeights(LoadStandalone(TestPaths.HiDream.T5));
                // HiDream's fourth text encoder is Llama-3.1-8B-Instruct, run as a feature extractor.
                // Uses the dedicated Llama31_8B preset: 32 layers (Qwen3-8B's 36 would over-read the
                // layer harvest), no per-head QK-norm (Qwen3 has it; Llama doesn't), theta=500k.
                LlamaStyleEncoder llama = new(LlamaStyleEncoderConfig.Llama31_8B);
                llama.LoadWeights(LoadStandalone(TestPaths.HiDream.Llama));
                _output.WriteLine($"[4/7] Quad text encoders loaded in {sw.ElapsedMilliseconds}ms");

                sw.Restart();
                using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
                (nuint freeBytes, nuint totalBytes) = backend.Context.GetMemoryInfo();
                double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
                const double MinRequiredGb = 30.0;
                if (freeGb < MinRequiredGb)
                {
                    _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM (total {totalBytes / (1024.0 * 1024.0 * 1024.0):F1} GB); need ≥{MinRequiredGb} GB to fit HiDream i1 transformer + four text encoders.");
                    transformer.Dispose();
                    return;
                }
                backend.PreloadWeights(transformer.EnumerateWeights());
                _output.WriteLine($"[5/7] Backend + preload ready in {sw.ElapsedMilliseconds}ms");

                using ClipTokenizer clipTokenizer = new(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
                using T5Tokenizer t5Tokenizer = new(TestPaths.Tokenizers.T5XxlSpiece, maxLength: 256);

                string prompt = "A photograph of an astronaut riding a horse";
                string negPrompt = "";

                int[] cl = clipTokenizer.Encode(prompt);
                int[] negCl = clipTokenizer.Encode(negPrompt);
                int[] cg = clipTokenizer.Encode(prompt);
                int[] negCg = clipTokenizer.Encode(negPrompt);
                int[] t5Tok = t5Tokenizer.Encode(prompt);
                int[] negT5 = t5Tokenizer.Encode(negPrompt);
                int[] t5Mask = T5Tokenizer.CreateAttentionMask(t5Tok);
                int[] negT5Mask = T5Tokenizer.CreateAttentionMask(negT5);

                int[] ll, negLl;
                if (EmbeddedTokenizerResources.HasLlama3Assets)
                {
                    // Preferred path: the embedded Llama-3.1 tokenizer (same as the SwarmUI extension uses).
                    using LlamaTokenizer llamaTokenizer = new(maxLength: 256);
                    ll = llamaTokenizer.Encode(prompt);
                    negLl = llamaTokenizer.Encode(negPrompt);
                }
                else if (File.Exists(TestPaths.Tokenizers.LlamaVocab) && File.Exists(TestPaths.Tokenizers.LlamaMerges))
                {
                    using LlamaTokenizer llamaTokenizer = new(TestPaths.Tokenizers.LlamaVocab, TestPaths.Tokenizers.LlamaMerges, maxLength: 256);
                    ll = llamaTokenizer.Encode(prompt);
                    negLl = llamaTokenizer.Encode(negPrompt);
                }
                else
                {
                    _output.WriteLine("  Note: Llama-3.1 tokenizer not embedded (no llama3_vocab.json/merges.txt in HartsyInference.Tokenizers/Resources) and no vocab.json+merges.txt on disk — using PLACEHOLDER tokens. Output will be wrong on the Llama branch. Embed the assets for a real result.");
                    ll = [LlamaTokenizer.BosTokenId, 1, 2, 3, 4, 5, 6, 7];
                    negLl = [LlamaTokenizer.BosTokenId, 1, 2, 3, 4, 5, 6, 7];
                }
                int eosL = ClipTokenizer.FindEosPosition(cl);
                int negEosL = ClipTokenizer.FindEosPosition(negCl);
                int eosG = ClipTokenizer.FindEosPosition(cg);
                int negEosG = ClipTokenizer.FindEosPosition(negCg);

                using HiDreamPipeline pipeline = new(backend, clipL, clipG, t5, llama, transformer, vae, config);

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

                _output.WriteLine($"\n[6/7] Generating {width}x{height}, {steps} steps, cfg={cfgScale}, seed=42...");
                Stopwatch genSw = Stopwatch.StartNew();
                (byte[] rgb, int outW, int outH, int seed) = pipeline.GenerateFromTokens(
                    cl, negCl, cg, negCg,
                    eosL, negEosL, eosG, negEosG,
                    t5Tok, negT5, t5Mask, negT5Mask,
                    ll, negLl, request,
                    progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));
                genSw.Stop();
                _output.WriteLine($"\n[7/7] Generation complete in {genSw.Elapsed.TotalSeconds:F1}s (seed={seed})");

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

                t5.Dispose();
                llama.Dispose();
            }
        }
        catch (NotImplementedException nie)
        {
            _output.WriteLine($"SKIPPED: HiDreamPipeline body has unfinished sections — {nie.Message}");
        }
    }

    private static Dictionary<string, Tensor> LoadStandalone(string path)
    {
        SafeTensorsLoader l = new();
        l.Load(path);
        return l.GetAllTensors();
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

    private void ValidateImageNotDegenerate(byte[] rgb)
    {
        int nonZero = 0, nonFF = 0;
        foreach (byte b in rgb)
        {
            if (b != 0) nonZero++;
            if (b != 255) nonFF++;
        }
        float nzPct = nonZero / (float)rgb.Length * 100;
        float nffPct = nonFF / (float)rgb.Length * 100;
        _output.WriteLine($"  Non-zero: {nzPct:F1}%, Non-255: {nffPct:F1}%");
        Assert.True(nzPct > 10, "Image appears all-black");
        Assert.True(nffPct > 10, "Image appears all-white");
    }
}

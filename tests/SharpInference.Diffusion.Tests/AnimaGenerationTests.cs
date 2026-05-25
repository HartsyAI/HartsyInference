using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Cuda;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Models.Vae.QwenImage;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tests.Common;

namespace SharpInference.Diffusion.Tests;

/// <summary>End-to-end Anima (Cosmos-Predict2 family) text-to-image generation. Anima's text encoder
/// is not ported as a first-class component; the test consumes pre-computed F32 embeddings (the same
/// format diffusers' pipeline accepts via <c>prompt_embeds</c>). Skips cleanly when artifacts or VRAM
/// are insufficient.</summary>
public sealed class AnimaGenerationTests
{
    private static string OutputDir => TestPaths.OutputDir;
    private readonly ITestOutputHelper _output;

    public AnimaGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Anima_T2I_Gpu_1024_Cfg() =>
        RunGenerationTest("anima_t2i_1024_cfg", width: 1024, height: 1024, steps: 30, cfgScale: 5.0f);

    [Fact]
    public void Anima_T2I_Gpu_512_NoCfg() =>
        RunGenerationTest("anima_t2i_512_nocfg", width: 512, height: 512, steps: 25, cfgScale: 1.0f);

    private void RunGenerationTest(string outputName, int width, int height, int steps, float cfgScale)
    {
        if (!File.Exists(TestPaths.Anima.Transformer))
        {
            _output.WriteLine($"SKIPPED: Anima transformer not found: {TestPaths.Anima.Transformer}");
            _output.WriteLine($"  Download from https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image (or the Anima fork).");
            return;
        }
        if (!File.Exists(TestPaths.Anima.Vae))
        {
            _output.WriteLine($"SKIPPED: VAE not found: {TestPaths.Anima.Vae}");
            return;
        }
        if (!File.Exists(TestPaths.Anima.PromptEmbeds))
        {
            _output.WriteLine($"SKIPPED: pre-computed prompt embeddings not found: {TestPaths.Anima.PromptEmbeds}");
            _output.WriteLine($"  Generate via Cosmos reference pipeline and dump as raw F32 [seq_len, dim].");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(AnimaGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        _output.WriteLine($"[1/5] Loading + converting transformer: {Path.GetFileName(TestPaths.Anima.Transformer)}");
        (AnimaCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            AnimaCheckpointConverter.LoadAndConvert(TestPaths.Anima.Transformer);
        _output.WriteLine($"  Loaded {converted.Transformer.Count} keys in {sw.ElapsedMilliseconds}ms (fp8={converted.IsFp8Mix})");

        try
        {
            using (loader)
            {
                Dictionary<string, Tensor> transformerWeights = CastWeightsToF32(converted.Transformer);
                Dictionary<string, Tensor> llmAdapterWeights = CastWeightsToF32(converted.LlmAdapter);

                sw.Restart();
                AnimaConfig config = AnimaConfig.AnimaPreview3;
                using AnimaTransformer transformer = new(config);
                transformer.LoadWeights(transformerWeights);
                using AnimaLlmAdapter llmAdapter = new(config.LlmAdapter);
                llmAdapter.LoadWeights(llmAdapterWeights);
                _output.WriteLine($"[2/5] Transformer + LlmAdapter ready in {sw.ElapsedMilliseconds}ms (depth={config.NumLayers}, hidden={config.HiddenSize})");

                sw.Restart();
                using SafeTensorsLoader vaeLoader = new();
                vaeLoader.Load(TestPaths.Anima.Vae);
                Dictionary<string, Tensor> vaeWeights = CastWeightsToF32(vaeLoader.GetAllTensors());
                // Anima uses the Qwen-Image VAE (3D causal autoencoder collapsed to 2D for image mode);
                // the standard VaeDecoder can't load this checkpoint because the key layout differs.
                using QwenImageVaeDecoder vae = new(VaeConfig.QwenImage);
                vae.LoadWeights(vaeWeights);
                _output.WriteLine($"[3/5] VAE ready in {sw.ElapsedMilliseconds}ms");

                sw.Restart();
                using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
                (nuint freeBytes, nuint totalBytes) = backend.Context.GetMemoryInfo();
                double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
                const double MinRequiredGb = 8.0;
                if (freeGb < MinRequiredGb)
                {
                    _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM (total {totalBytes / (1024.0 * 1024.0 * 1024.0):F1} GB); need ≥{MinRequiredGb} GB for Anima 2B FP16.");
                    return;
                }
                backend.PreloadWeights(transformer.EnumerateWeights());
                _output.WriteLine($"[4/5] Backend + preload ready in {sw.ElapsedMilliseconds}ms");

                // Text embedding dim = LlmAdapter input dim = Qwen-3 0.6B hidden (1024).
                using Tensor promptEmbeds = LoadF32Tensor(TestPaths.Anima.PromptEmbeds, config.LlmAdapter.HiddenSize);
                // Negative embeds — produced alongside the positive prompt by encode_anima_prompt.py.
                // Required when cfgScale > 1.0; the pipeline throws if missing in that case.
                string negPromptPath = Path.Combine(Path.GetDirectoryName(TestPaths.Anima.PromptEmbeds)!, "neg_prompt.bin");
                Tensor? negPromptEmbeds = (cfgScale > 1.0f && File.Exists(negPromptPath))
                    ? LoadF32Tensor(negPromptPath, config.LlmAdapter.HiddenSize)
                    : null;
                _output.WriteLine($"  Loaded prompt embeds: {promptEmbeds.Shape}, neg: {(negPromptEmbeds?.Shape.ToString() ?? "(none)")}");

                using AnimaPipeline pipeline = new(backend, transformer, llmAdapter, vae, config);

                TextToImageRequest request = new()
                {
                    Prompt = "(supplied via pre-computed embeddings)",
                    NegativePrompt = "",
                    Width = width,
                    Height = height,
                    Steps = steps,
                    CfgScale = cfgScale,
                    Seed = 42,
                };

                // T5 token IDs (the LlmAdapter main stream lookup keys) are produced by encode_anima_prompt.py
                // alongside the Qwen3 embeddings. We load the raw int64 binaries.
                string testEmbedsDir = Path.GetDirectoryName(TestPaths.Anima.PromptEmbeds)!;
                int[] t5Ids = LoadInt64TensorAsInt32(Path.Combine(testEmbedsDir, "prompt_t5_ids.bin"));
                int[]? negT5Ids = (cfgScale > 1.0f)
                    ? LoadInt64TensorAsInt32(Path.Combine(testEmbedsDir, "neg_prompt_t5_ids.bin"))
                    : null;
                _output.WriteLine($"  T5 token IDs: pos len={t5Ids.Length}, neg len={negT5Ids?.Length ?? 0}");

                _output.WriteLine($"\n[5/5] Generating {width}x{height}, {steps} steps, cfg={cfgScale}, seed=42...");
                Stopwatch genSw = Stopwatch.StartNew();
                (byte[] rgb, int outW, int outH, int seed) = pipeline.GenerateFromEmbeddings(
                    promptEmbeds, t5Ids, request, cfgScale, negPromptEmbeds, negT5Ids,
                    progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));
                negPromptEmbeds?.Dispose();
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
        }
        catch (NotImplementedException nie)
        {
            _output.WriteLine($"SKIPPED: pipeline body has unfinished sections — {nie.Message}");
        }
    }

    private static unsafe int[] LoadInt64TensorAsInt32(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"T5 token IDs file not found at {path}. Re-run encode_anima_prompt.py to produce it alongside prompt.bin.", path);
        byte[] raw = File.ReadAllBytes(path);
        long count = raw.LongLength / sizeof(long);
        int[] result = new int[count];
        fixed (byte* src = raw)
        {
            long* lp = (long*)src;
            for (long i = 0; i < count; i++) result[i] = checked((int)lp[i]);
        }
        return result;
    }

    private static unsafe Tensor LoadF32Tensor(string path, int embedDim)
    {
        byte[] data = File.ReadAllBytes(path);
        long totalFloats = data.Length / sizeof(float);
        if (totalFloats % embedDim != 0)
            throw new InvalidOperationException($"Embedding file {path} has {totalFloats} floats which is not a multiple of {embedDim}");

        int seqLen = (int)(totalFloats / embedDim);
        Tensor result = new(new TensorShape(1, seqLen, embedDim), DType.F32);
        fixed (byte* src = data)
        {
            Buffer.MemoryCopy(src, (void*)result.DataPointer, data.Length, data.Length);
        }
        return result;
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

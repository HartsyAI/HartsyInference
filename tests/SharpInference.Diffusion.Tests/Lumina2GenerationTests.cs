using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Cuda;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tests.Common;

namespace SharpInference.Diffusion.Tests;

/// <summary>End-to-end Lumina-Image-2.0 generation. The Gemma 2 2B text encoder is not yet implemented;
/// the test consumes pre-computed F32 caption embeddings (the same format the diffusers pipeline accepts
/// via <c>prompt_embeds</c>). Skips cleanly when checkpoint, VAE, embeddings, or PTX are missing, or when
/// the GPU has insufficient free VRAM for the FP16 transformer.</summary>
public sealed class Lumina2GenerationTests
{
    private static string OutputDir => TestPaths.OutputDir;
    private readonly ITestOutputHelper _output;

    public Lumina2GenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Lumina2_V2_0_Gpu_1024_Cfg() =>
        RunGenerationTest("lumina2_v2_0_1024_cfg", width: 1024, height: 1024, steps: 30, cfgScale: 4.0f);

    [Fact]
    public void Lumina2_V2_0_Gpu_512_NoCfg() =>
        RunGenerationTest("lumina2_v2_0_512_nocfg", width: 512, height: 512, steps: 25, cfgScale: 1.0f);

    private void RunGenerationTest(string outputName, int width, int height, int steps, float cfgScale)
    {
        if (!File.Exists(TestPaths.Lumina2.Transformer))
        {
            _output.WriteLine($"SKIPPED: Lumina2 transformer not found: {TestPaths.Lumina2.Transformer}");
            _output.WriteLine($"  Download from https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0");
            return;
        }
        if (!File.Exists(TestPaths.Lumina2.Vae))
        {
            _output.WriteLine($"SKIPPED: Lumina2 VAE not found: {TestPaths.Lumina2.Vae}");
            return;
        }
        if (!File.Exists(TestPaths.Lumina2.PromptEmbeds))
        {
            _output.WriteLine($"SKIPPED: pre-computed Gemma 2 caption embeddings not found at {TestPaths.Lumina2.PromptEmbeds}.");
            _output.WriteLine($"  Generate via diffusers: pipe.encode_prompt(prompt) and dump the F32 numpy array (no header).");
            return;
        }
        bool useCfg = cfgScale > 1.0f;
        if (useCfg && !File.Exists(TestPaths.Lumina2.NegPromptEmbeds))
        {
            _output.WriteLine($"SKIPPED: cfgScale={cfgScale} requires negative-prompt embeddings at {TestPaths.Lumina2.NegPromptEmbeds}.");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(Lumina2GenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        _output.WriteLine($"[1/5] Loading + converting transformer: {Path.GetFileName(TestPaths.Lumina2.Transformer)}");
        (Lumina2CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader transformerLoader) =
            Lumina2CheckpointConverter.LoadAndConvert(TestPaths.Lumina2.Transformer);
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms ({converted.Transformer.Count} keys, fp8={converted.IsFp8Mix})");

        try
        {
            using (transformerLoader)
            {
                Dictionary<string, Tensor> transformerWeights = CastWeightsToF32(converted.Transformer);

                sw.Restart();
                Lumina2Config config = Lumina2Config.FromWeights(transformerWeights);
                using Lumina2Transformer transformer = new(config);
                transformer.LoadWeights(transformerWeights);
                _output.WriteLine($"[2/5] Transformer ready in {sw.ElapsedMilliseconds}ms (depth={config.NumLayers}, hidden={config.HiddenSize})");

                sw.Restart();
                using SafeTensorsLoader vaeLoader = new();
                vaeLoader.Load(TestPaths.Lumina2.Vae);
                Dictionary<string, Tensor> vaeWeights = CastWeightsToF32(vaeLoader.GetAllTensors());
                VaeDecoder vae = new(VaeConfig.Flux);
                vae.LoadWeights(vaeWeights);
                _output.WriteLine($"[3/5] VAE ready in {sw.ElapsedMilliseconds}ms");

                sw.Restart();
                using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
                (nuint freeBytes, nuint totalBytes) = backend.Context.GetMemoryInfo();
                double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
                const double MinRequiredGb = 8.0;
                if (freeGb < MinRequiredGb)
                {
                    _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM (total {totalBytes / (1024.0 * 1024.0 * 1024.0):F1} GB); need ≥{MinRequiredGb} GB for Lumina-Image-2.0 (2B FP16, ~4 GB transformer + VAE + activations).");
                    return;
                }
                backend.PreloadWeights(transformer.EnumerateWeights());
                _output.WriteLine($"[4/5] Backend + preload ready in {sw.ElapsedMilliseconds}ms (device: {backend.Capabilities.Name})");

                using Tensor promptEmbeds = LoadF32Tensor(TestPaths.Lumina2.PromptEmbeds, config.CapFeatDim);
                Tensor? negPromptEmbeds = useCfg ? LoadF32Tensor(TestPaths.Lumina2.NegPromptEmbeds, config.CapFeatDim) : null;
                _output.WriteLine($"  Prompt embeds: {promptEmbeds.Shape}");

                using Lumina2Pipeline pipeline = new(backend, transformer, vae, config);

                TextToImageRequest request = new()
                {
                    Prompt = "(supplied via pre-computed embeddings)",
                    NegativePrompt = useCfg ? "(supplied via pre-computed embeddings)" : "",
                    Width = width,
                    Height = height,
                    Steps = steps,
                    CfgScale = cfgScale,
                    Seed = 42,
                };

                _output.WriteLine($"\n[5/5] Generating {width}x{height}, {steps} steps, cfg={cfgScale}, seed=42...");
                Stopwatch genSw = Stopwatch.StartNew();
                (byte[] rgb, int outW, int outH, int seed) = pipeline.GenerateFromEmbeddings(
                    promptEmbeds, request, cfgScale, negPromptEmbeds,
                    progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));
                genSw.Stop();
                _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalSeconds:F1}s (seed={seed})");

                negPromptEmbeds?.Dispose();

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

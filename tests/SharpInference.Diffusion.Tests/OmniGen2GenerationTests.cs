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

/// <summary>End-to-end OmniGen 2 (Qwen2.5-VL-conditioned MMDiT) text-to-image. The Qwen2.5-VL forward
/// is not implemented as a first-class C# component yet; the test consumes pre-computed F32 caption
/// embeddings (the format diffusers' pipeline accepts via <c>prompt_embeds</c>). Skips cleanly when any
/// artifact or the GPU's free VRAM is insufficient. The transformer's <c>Forward</c> is currently a
/// scaffold that throws <see cref="NotImplementedException"/> with a clear marker — the test catches
/// that and skips with a "first-run wiring needed" message.</summary>
public sealed class OmniGen2GenerationTests
{
    private static string OutputDir => TestPaths.OutputDir;
    private readonly ITestOutputHelper _output;

    public OmniGen2GenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void OmniGen2_V1_Gpu_1024_Cfg() =>
        RunGenerationTest("omnigen2_v1_1024_cfg", width: 1024, height: 1024, steps: 30, cfgScale: 5.0f);

    [Fact]
    public void OmniGen2_V1_Gpu_512_NoCfg() =>
        RunGenerationTest("omnigen2_v1_512_nocfg", width: 512, height: 512, steps: 25, cfgScale: 1.0f);

    private void RunGenerationTest(string outputName, int width, int height, int steps, float cfgScale)
    {
        if (!File.Exists(TestPaths.OmniGen2.Transformer))
        {
            _output.WriteLine($"SKIPPED: OmniGen 2 transformer not found: {TestPaths.OmniGen2.Transformer}");
            _output.WriteLine($"  Download from https://huggingface.co/OmniGen2/OmniGen2");
            return;
        }
        if (!File.Exists(TestPaths.OmniGen2.Vae))
        {
            _output.WriteLine($"SKIPPED: VAE not found: {TestPaths.OmniGen2.Vae}");
            return;
        }
        if (!File.Exists(TestPaths.OmniGen2.PromptEmbeds))
        {
            _output.WriteLine($"SKIPPED: pre-computed Qwen2.5-VL embeddings not found at {TestPaths.OmniGen2.PromptEmbeds}");
            _output.WriteLine($"  Generate via diffusers reference pipeline and dump as raw F32 [seq_len, 2048].");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(OmniGen2GenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        _output.WriteLine($"[1/5] Loading + converting transformer: {Path.GetFileName(TestPaths.OmniGen2.Transformer)}");
        (OmniGen2CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            OmniGen2CheckpointConverter.LoadAndConvert(TestPaths.OmniGen2.Transformer);
        _output.WriteLine($"  Loaded {converted.Transformer.Count} keys in {sw.ElapsedMilliseconds}ms (fp8={converted.IsFp8Mix})");

        try
        {
            using (loader)
            {
                Dictionary<string, Tensor> transformerWeights = CastWeightsToF32(converted.Transformer);

                sw.Restart();
                OmniGen2Config config = OmniGen2Config.V1;
                using OmniGen2Transformer transformer = new(config);
                transformer.LoadWeights(transformerWeights);
                _output.WriteLine($"[2/5] Transformer ready in {sw.ElapsedMilliseconds}ms (depth={config.NumLayers}, hidden={config.HiddenSize})");

                sw.Restart();
                using SafeTensorsLoader vaeLoader = new();
                vaeLoader.Load(TestPaths.OmniGen2.Vae);
                Dictionary<string, Tensor> vaeWeights = CastWeightsToF32(vaeLoader.GetAllTensors());
                VaeDecoder vae = new(VaeConfig.Flux);
                vae.LoadWeights(vaeWeights);
                _output.WriteLine($"[3/5] VAE ready in {sw.ElapsedMilliseconds}ms");

                sw.Restart();
                using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
                (nuint freeBytes, nuint totalBytes) = backend.Context.GetMemoryInfo();
                double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
                const double MinRequiredGb = 12.0;
                if (freeGb < MinRequiredGb)
                {
                    _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM (total {totalBytes / (1024.0 * 1024.0 * 1024.0):F1} GB); need ≥{MinRequiredGb} GB for OmniGen 2 FP16 transformer + VAE + Qwen2.5-VL.");
                    return;
                }
                backend.PreloadWeights(transformer.EnumerateWeights());
                _output.WriteLine($"[4/5] Backend + preload ready in {sw.ElapsedMilliseconds}ms");

                using Tensor promptEmbeds = LoadF32Tensor(TestPaths.OmniGen2.PromptEmbeds, config.TextFeatDim);

                using OmniGen2Pipeline pipeline = new(backend, transformer, vae, config);

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

                _output.WriteLine($"\n[5/5] Generating {width}x{height}, {steps} steps, cfg={cfgScale}, seed=42...");
                Stopwatch genSw = Stopwatch.StartNew();
                (byte[] rgb, int outW, int outH, int seed) = pipeline.GenerateFromEmbeddings(
                    promptEmbeds, request, cfgScale, null,
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
        }
        catch (NotImplementedException nie)
        {
            _output.WriteLine($"SKIPPED: OmniGen2Transformer.Forward needs first-run wiring — {nie.Message}");
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

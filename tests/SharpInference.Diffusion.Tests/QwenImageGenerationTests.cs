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

/// <summary>End-to-end Qwen-Image generation smoke test. Expects three artifacts: a Qwen-Image transformer checkpoint, a Qwen2.5-VL-7B text encoder checkpoint, and a 16-channel Qwen-Image VAE. Skips cleanly when any of those paths or the Qwen3 BPE tokenizer assets are missing — Qwen-Image isn't bundled with the repo, so this test is intended to run on a developer GPU box, not CI.</summary>
public class QwenImageGenerationTests
{
    private readonly ITestOutputHelper _output;

    public QwenImageGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void QwenImage_V1_Gpu_512_NoCfg_Smoke() =>
        RunGenerationTest("qwen_image_v1_512_smoke", width: 512, height: 512, steps: 8, cfgScale: 1.0f);

    private void RunGenerationTest(string outputName, int width, int height, int steps, float cfgScale)
    {
        string transformerPath = TestPaths.QwenImage.V1;
        string textEncoderPath = TestPaths.QwenImage.TextEncoder;
        string vaePath = TestPaths.QwenImage.Vae;

        if (!File.Exists(transformerPath))
        {
            _output.WriteLine($"SKIPPED: Qwen-Image transformer not found: {transformerPath}");
            return;
        }
        if (!File.Exists(textEncoderPath))
        {
            _output.WriteLine($"SKIPPED: Qwen2.5-VL text encoder not found: {textEncoderPath}");
            return;
        }
        if (!File.Exists(vaePath))
        {
            _output.WriteLine($"SKIPPED: Qwen-Image VAE not found: {vaePath}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(QwenImageGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        _output.WriteLine($"[1/7] Loading + converting transformer: {Path.GetFileName(transformerPath)}");
        (QwenImageCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader transformerLoader) =
            QwenImageCheckpointConverter.LoadAndConvert(transformerPath);
        sw.Stop();
        _output.WriteLine($"  Converted in {sw.ElapsedMilliseconds}ms (transformer={converted.Transformer.Count}, " +
            $"text_encoder={converted.TextEncoder.Count}, vae={converted.Vae.Count})");

        Dictionary<string, Tensor> textEncoderWeights = converted.TextEncoder.Count > 0
            ? converted.TextEncoder
            : LoadStandalone(textEncoderPath);
        Dictionary<string, Tensor> vaeWeights = converted.Vae.Count > 0
            ? converted.Vae
            : LoadStandalone(vaePath);

        using (transformerLoader)
        {
            _output.WriteLine("[2/7] Tokenizing prompt with Qwen3 BPE tokenizer...");
            using Qwen3Tokenizer tokenizer = new(maxLength: 512);

            string prompt = "A photograph of an astronaut riding a horse";
            string negPrompt = "";

            int[] promptTokens = tokenizer.Encode(prompt);
            int[] negTokens = tokenizer.Encode(negPrompt);

            _output.WriteLine("[3/7] Loading Qwen2.5-VL-7B text encoder...");
            sw.Restart();
            LlamaStyleEncoder textEncoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
            textEncoder.LoadWeights(textEncoderWeights);
            sw.Stop();
            _output.WriteLine($"  Text encoder loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[4/7] Loading Qwen-Image transformer...");
            sw.Restart();
            QwenImageConfig config = QwenImageConfig.V1;
            QwenImageTransformer transformer = new QwenImageTransformer(config);
            transformer.LoadWeights(converted.Transformer);
            sw.Stop();
            _output.WriteLine($"  Transformer loaded in {sw.ElapsedMilliseconds}ms (depth={config.Depth}, " +
                $"hidden={config.HiddenSize}, heads={config.NumHeads})");

            _output.WriteLine("[5/7] Loading Qwen-Image VAE...");
            sw.Restart();
            VaeDecoder vae = new VaeDecoder(VaeConfig.QwenImage);
            vae.LoadWeights(CastWeightsToF32(vaeWeights));
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[6/7] Initializing CUDA backend...");
            sw.Restart();
            using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
            sw.Stop();
            _output.WriteLine($"  Backend ready in {sw.ElapsedMilliseconds}ms (device: {backend.Capabilities.Name})");

            (nuint freeBytes, nuint totalBytes) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            const double MinRequiredGb = 22.0;
            if (freeGb < MinRequiredGb)
            {
                _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM (total {totalBytes / (1024.0 * 1024.0 * 1024.0):F1} GB); need ≥{MinRequiredGb} GB to fit Qwen-Image at FP8 (transformer ~20 GB) + Qwen2.5-VL-7B + 16-channel VAE. The implementation is end-to-end ready; this test will run on a larger card or once a GGUF Q4_K Qwen-Image checkpoint is wired in (transformer ~6 GB → fits 12 GB).");
                transformer.Dispose();
                textEncoder.Dispose();
                return;
            }

            using QwenImagePipeline pipeline = new QwenImagePipeline(backend, textEncoder, transformer, vae, config);

            TextToImageRequest request = new TextToImageRequest
            {
                Prompt = prompt,
                NegativePrompt = negPrompt,
                Width = width,
                Height = height,
                Steps = steps,
                CfgScale = cfgScale,
                Seed = 42,
            };

            _output.WriteLine($"\n[7/7] Generating {width}x{height}, {steps} steps, cfg={cfgScale}, seed=42...");
            Stopwatch genSw = Stopwatch.StartNew();

            (byte[] rgbData, int outW, int outH, int seed) = pipeline.GenerateFromTokens(
                promptTokens, negTokens, request,
                progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));

            genSw.Stop();
            _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalSeconds:F1}s (seed={seed})");

            Assert.Equal(width, outW);
            Assert.Equal(height, outH);
            Assert.Equal(width * height * 3, rgbData.Length);
            ValidateImageNotDegenerate(rgbData);

            string outputDir = TestPaths.OutputDir;
            Directory.CreateDirectory(outputDir);
            string outputPath = Path.Combine(outputDir, $"{outputName}_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outputPath, rgbData, outW, outH);
            _output.WriteLine($"  Saved: {outputPath}");

            totalSw.Stop();
            _output.WriteLine($"\nTotal: {totalSw.Elapsed.TotalSeconds:F1}s");

            transformer.Dispose();
            textEncoder.Dispose();
        }
    }

    private static Dictionary<string, Tensor> LoadStandalone(string path)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        Dictionary<string, Tensor> raw = loader.GetAllTensors();
        return raw;
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            f32[kvp.Key] = (kvp.Value.DType == DType.F16 || kvp.Value.DType == DType.BF16)
                ? kvp.Value.CastTo(DType.F32)
                : kvp.Value;
        }
        return f32;
    }

    private void ValidateImageNotDegenerate(byte[] rgbData)
    {
        int nonZero = 0, nonFF = 0;
        foreach (byte b in rgbData)
        {
            if (b != 0) nonZero++;
            if (b != 255) nonFF++;
        }
        float nonZeroPct = nonZero / (float)rgbData.Length * 100;
        float nonFFPct = nonFF / (float)rgbData.Length * 100;
        _output.WriteLine($"  Non-zero: {nonZeroPct:F1}%, Non-255: {nonFFPct:F1}%");
        Assert.True(nonZeroPct > 10, "Image appears to be all black");
        Assert.True(nonFFPct > 10, "Image appears to be all white");
    }
}

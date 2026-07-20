using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>End-to-end Qwen-Image generation smoke test. Expects three artifacts: a Qwen-Image transformer checkpoint, a Qwen2.5-VL-7B text encoder checkpoint, and a 16-channel Qwen-Image VAE. Skips cleanly when any of those paths or the Qwen3 BPE tokenizer assets are missing — Qwen-Image isn't bundled with the repo, so this test is intended to run on a developer GPU box, not CI.</summary>
[Trait("Category", "Integration")]
public class QwenImageGenerationTests
{
    private readonly ITestOutputHelper _output;

    public QwenImageGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void QwenImage_V1_Gpu_512_NoCfg_Smoke() =>
        RunGenerationTest("qwen_image_v1_512_smoke", width: 512, height: 512, steps: 8, cfgScale: 1.0f);

    /// <summary>Real e2e via the Q4_K_M GGUF (~13 GB) — fits the 4090 where the 20 GB fp8 single-file does not.
    /// 1024-native, true-CFG. This is the runnable Qwen-Image path on 24 GB hardware.</summary>
    [Fact]
    public void QwenImage_V1_Gpu_1024_Cfg_Gguf() =>
        RunGenerationTest("qwen_image_v1_1024_gguf", width: 1024, height: 1024, steps: 28, cfgScale: 4.0f, useGguf: true);

    /// <summary>Fast diagnostic: 6 steps @1024 GGUF to read the [DIAG] conditioning/velocity/latent stats without a full run.</summary>
    [Fact]
    public void QwenImage_Diag_6step_Gguf() =>
        RunGenerationTest("qwen_image_diag_6step", width: 1024, height: 1024, steps: 6, cfgScale: 4.0f, useGguf: true);

    private void RunGenerationTest(string outputName, int width, int height, int steps, float cfgScale, bool useGguf = false)
    {
        string transformerPath = useGguf ? TestPaths.QwenImage.V1Gguf : TestPaths.QwenImage.V1;
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

        _output.WriteLine($"[1/7] Loading + converting transformer ({(useGguf ? "GGUF Q4_K, lazy" : "safetensors")}): {Path.GetFileName(transformerPath)}");
        QwenImageCheckpointConverter.ConvertedWeights converted;
        IDisposable transformerLoader;
        if (useGguf)
        {
            // Lazy GGUF load (Flux pattern): keeps Q4_K/Q5_K/Q6_K tensors quantized; CudaBackend.Linear
            // dequants per-GEMM on the GPU. Identity key-map → bare diffusers keys → converter. The GGUF is
            // transformer-only, so converted.TextEncoder/.Vae come back empty and fall through to the standalone
            // safetensors loads below. Handle must stay alive (mmap views) until LoadWeights copies to GPU.
            GgufModelLoader.LoadedGgufModel ggufHandle = GgufModelLoader.Load(transformerPath);
            // GGUF stores matrix dims [in, out] (data row-major [out, in], same as HF); relabel rank-2 shapes to
            // [out, in] before the converter splits/maps keys, exactly as the LLM GGUF path does. Without this every
            // Linear is transposed and the first matmul (timestep embedder) derives M=0 → a degenerate kernel launch.
            Dictionary<string, Tensor> ggufRawDict = GgufModelLoader.RelabelRank2ToPyTorchOrder(ggufHandle.Weights);
            converted = QwenImageCheckpointConverter.Convert(ggufRawDict);
            transformerLoader = ggufHandle;
        }
        else
        {
            (QwenImageCheckpointConverter.ConvertedWeights c, SafeTensorsLoader loader) =
                QwenImageCheckpointConverter.LoadAndConvert(transformerPath);
            converted = c;
            transformerLoader = loader;
        }
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
            // Qwen-Image conditioning MUST follow diffusers' _get_qwen_prompt_embeds: wrap the prompt in the fixed
            // encode template (ChatML with the "Describe the image..." system message), tokenize at REAL length (NOT
            // padded to 512 — padding pollutes the cross-attention), then drop the 34-token template prefix from the
            // hidden states (promptDropIndex below). The Qwen2.5-VL tokenizer shares Qwen2's base BPE. Feeding the raw
            // prompt padded to 512 with no template produced incoherent (grid) output.
            _output.WriteLine("[2/7] Tokenizing prompt with Qwen-Image template (Qwen2.5-VL ChatML)...");
            using Qwen2Tokenizer tokenizer = new();
            const string qwenImageSystem =
                "Describe the image by detailing the color, shape, size, texture, quantity, text, " +
                "spatial relationships of the objects and background:";
            const int templatePrefixDrop = 34;

            string prompt = "A photograph of an astronaut riding a horse";
            string negPrompt = "";

            int[] promptTokens = tokenizer.EncodeChat(prompt, systemPrompt: qwenImageSystem, addGenerationPrompt: true);
            int[] negTokens = tokenizer.EncodeChat(negPrompt, systemPrompt: qwenImageSystem, addGenerationPrompt: true);
            _output.WriteLine($"  prompt tokens={promptTokens.Length} (drop {templatePrefixDrop} → {promptTokens.Length - templatePrefixDrop} cond), neg tokens={negTokens.Length}");

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
            QwenImageVaeDecoder vae = new QwenImageVaeDecoder(VaeConfig.QwenImage);
            vae.LoadWeights(CastWeightsToF32(vaeWeights));
            sw.Stop();
            _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[6/7] Initializing CUDA backend...");
            sw.Restart();
            using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
            // GGUF Q4_K is a ~20B model: caching an F16 dequant of every weight (CacheWeightCasts default) would need
            // ~40 GB. Force the transient per-GEMM dequant so only the quantized weights stay resident (~13 GB) plus
            // one weight's F16 cast at a time — fits the 4090. This is the low-VRAM lever the MinRequiredGb=14 gate assumes.
            if (useGguf)
                backend.CacheWeightCasts = false;
            sw.Stop();
            _output.WriteLine($"  Backend ready in {sw.ElapsedMilliseconds}ms (device: {backend.Capabilities.Name})");

            (nuint freeBytes, nuint totalBytes) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            // GGUF Q4_K transformer stays ~7 GB resident (GPU dequant per-GEMM); the encoder loads+frees before
            // denoise. FP8 single-file needs ~20 GB resident for the transformer alone.
            double MinRequiredGb = useGguf ? 14.0 : 22.0;
            if (freeGb < MinRequiredGb)
            {
                _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM (total {totalBytes / (1024.0 * 1024.0 * 1024.0):F1} GB); need ≥{MinRequiredGb} GB. GGUF Q4_K path fits ~14 GB; FP8 single-file needs ~22 GB (transformer ~20 GB) + Qwen2.5-VL-7B + 16-channel VAE.");
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
                progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"),
                promptDropIndex: templatePrefixDrop, negativeDropIndex: templatePrefixDrop);

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

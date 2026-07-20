using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>End-to-end Kandinsky 5.0 image generation. Because the dual text encoder stack
/// (Qwen2.5-VL-7B + CLIP-L) is not yet implemented in HartsyInference, the test relies on
/// pre-computed embeddings shipped as raw F32 binaries — the same format Diffusers' pipeline
/// accepts via <c>prompt_embeds_qwen</c> / <c>prompt_embeds_clip</c>. Tests skip cleanly when any
/// artifact is missing.</summary>
[Trait("Category", "Integration")]
public class Kandinsky5GenerationTests
{
    private static string OutputDir => TestPaths.OutputDir;
    private readonly ITestOutputHelper _output;
    public Kandinsky5GenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Kandinsky5_Lite_Gpu_1024_Cfg() =>
        RunGenerationTest("kandinsky5_lite_1024_cfg", width: 1024, height: 1024, steps: 50, cfgScale: 3.5f);

    [Fact]
    public void Kandinsky5_Lite_Gpu_512_NoCfg() =>
        RunGenerationTest("kandinsky5_lite_512_nocfg", width: 512, height: 512, steps: 25, cfgScale: 1.0f);

    private void RunGenerationTest(string outputName, int width, int height, int steps, float cfgScale)
    {
        bool useSingle = File.Exists(TestPaths.Kandinsky5.TransformerSingle);
        bool useFolder = Directory.Exists(TestPaths.Kandinsky5.TransformerDir);
        if (!useSingle && !useFolder)
        {
            _output.WriteLine($"SKIPPED: no Kandinsky 5 transformer found at:");
            _output.WriteLine($"  - single file: {TestPaths.Kandinsky5.TransformerSingle}");
            _output.WriteLine($"  - diffusers dir: {TestPaths.Kandinsky5.TransformerDir}");
            _output.WriteLine($"  Download from: https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers");
            return;
        }

        if (!File.Exists(TestPaths.Kandinsky5.Vae))
        {
            _output.WriteLine($"SKIPPED: VAE not found: {TestPaths.Kandinsky5.Vae}");
            return;
        }

        if (!File.Exists(TestPaths.Kandinsky5.PromptQwenEmbeds) ||
            !File.Exists(TestPaths.Kandinsky5.PromptClipPooled))
        {
            _output.WriteLine($"SKIPPED: pre-computed prompt embeddings not found:");
            _output.WriteLine($"  Qwen: {TestPaths.Kandinsky5.PromptQwenEmbeds}");
            _output.WriteLine($"  CLIP: {TestPaths.Kandinsky5.PromptClipPooled}");
            _output.WriteLine($"  Generate via the diffusers reference pipeline using `pipe.encode_prompt(...)`");
            _output.WriteLine($"  and dump the F32 numpy arrays to .bin files (no header).");
            return;
        }

        bool useCfg = cfgScale > 1.0f;
        if (useCfg && (!File.Exists(TestPaths.Kandinsky5.NegPromptQwenEmbeds) ||
                       !File.Exists(TestPaths.Kandinsky5.NegPromptClipPooled)))
        {
            _output.WriteLine($"SKIPPED: cfgScale={cfgScale} requires negative-prompt embeddings:");
            _output.WriteLine($"  Qwen: {TestPaths.Kandinsky5.NegPromptQwenEmbeds}");
            _output.WriteLine($"  CLIP: {TestPaths.Kandinsky5.NegPromptClipPooled}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(Kandinsky5GenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Load + convert transformer ──
        Kandinsky5CheckpointConverter.ConvertedWeights converted;
        List<IDisposable> tensorOwners = new();
        if (useSingle)
        {
            _output.WriteLine($"[1/6] Loading transformer single file: {Path.GetFileName(TestPaths.Kandinsky5.TransformerSingle)}");
            (Kandinsky5CheckpointConverter.ConvertedWeights c, SafeTensorsLoader loader) =
                Kandinsky5CheckpointConverter.LoadAndConvert(TestPaths.Kandinsky5.TransformerSingle);
            converted = c;
            tensorOwners.Add(loader);
        }
        else
        {
            _output.WriteLine($"[1/6] Loading transformer diffusers dir: {TestPaths.Kandinsky5.TransformerDir}");
            (Kandinsky5CheckpointConverter.ConvertedWeights c, List<SafeTensorsLoader> loaders) =
                Kandinsky5CheckpointConverter.LoadDiffusersFolder(TestPaths.Kandinsky5.TransformerDir);
            converted = c;
            foreach (SafeTensorsLoader l in loaders) tensorOwners.Add(l);
        }
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms ({converted.Transformer.Count} keys)");

        try
        {
            // Cast the BF16 transformer to F16 (12 GB, fits the 24 GB card directly, native F16 cuBLAS GEMM).
            // F32 would be 24 GB (OOM); the BF16+CacheWeightCasts=false transient path produced a BLANK image
            // (the transient dequant path is validated for fp8/GGUF, not BF16) — F16 sidesteps it.
            Dictionary<string, Tensor> transformerWeights = new(converted.Transformer.Count);
            foreach (KeyValuePair<string, Tensor> kvp in converted.Transformer)
                transformerWeights[kvp.Key] = kvp.Value.DType == DType.BF16 ? kvp.Value.CastTo(DType.F16) : kvp.Value;

            // ── 2. Build transformer ──
            sw.Restart();
            Kandinsky5Config config = Kandinsky5Config.Lite;
            using Kandinsky5Transformer transformer = new(config);
            transformer.LoadWeights(transformerWeights);
            _output.WriteLine($"[2/6] Transformer ready in {sw.ElapsedMilliseconds}ms");

            // ── 3. Build VAE ──
            sw.Restart();
            using SafeTensorsLoader vaeLoader = new();
            vaeLoader.Load(TestPaths.Kandinsky5.Vae);
            Dictionary<string, Tensor> vaeWeights = CastWeightsToF32(vaeLoader.GetAllTensors());
            VaeDecoder vae = new(VaeConfig.Flux);
            vae.LoadWeights(vaeWeights);
            _output.WriteLine($"[3/6] VAE ready in {sw.ElapsedMilliseconds}ms");

            // ── 4. Initialize CUDA backend + preload transformer (probe VRAM before/after) ──
            sw.Restart();
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            // BF16 transformer (12 GB): transient per-GEMM dequant instead of a resident F32 weight cache
            // (which would need 24 GB). Mirrors the fp8 model tests (Boogu/Krea2/Qwen-GGUF).
            backend.CacheWeightCasts = false;
            (nuint freeBefore, nuint totalVram) = backend.Context.GetMemoryInfo();
            _output.WriteLine($"[4/6] Backend ready (device: {backend.Capabilities.Name})");
            _output.WriteLine($"  VRAM before preload: {freeBefore / (1024 * 1024)} MiB free / {totalVram / (1024 * 1024)} MiB total");

            backend.PreloadWeights(transformer.EnumerateWeights());
            (nuint freeAfter, _) = backend.Context.GetMemoryInfo();
            long preloadCostMib = ((long)freeBefore - (long)freeAfter) / (1024 * 1024);
            _output.WriteLine($"  VRAM after preload: {freeAfter / (1024 * 1024)} MiB free (Δ {preloadCostMib} MiB)");
            _output.WriteLine($"  Preload took {sw.ElapsedMilliseconds}ms");

            // ── 5. Load pre-computed text embeddings ──
            _output.WriteLine($"[5/6] Loading pre-computed text embeddings...");
            using Tensor qwenEmbeds = LoadF32Tensor(TestPaths.Kandinsky5.PromptQwenEmbeds, config.InTextDim);
            using Tensor clipPooled = LoadPooled(TestPaths.Kandinsky5.PromptClipPooled, config.InTextDim2);
            Tensor? negQwen = useCfg ? LoadF32Tensor(TestPaths.Kandinsky5.NegPromptQwenEmbeds, config.InTextDim) : null;
            Tensor? negClip = useCfg ? LoadPooled(TestPaths.Kandinsky5.NegPromptClipPooled, config.InTextDim2) : null;
            _output.WriteLine($"  Qwen seq: {qwenEmbeds.Shape}, CLIP pooled: {clipPooled.Shape}");

            using Kandinsky5Pipeline pipeline = new(backend, transformer, vae, config);

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

            _output.WriteLine($"\n[6/6] Generating {width}x{height}, {steps} steps, cfg={cfgScale}, seed=42...");
            Stopwatch genSw = Stopwatch.StartNew();
            (byte[] rgb, int outW, int outH, int seed) = pipeline.GenerateFromEmbeddings(
                qwenEmbeds, clipPooled, negQwen, negClip, request,
                progress => _output.WriteLine($"  Step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"));
            genSw.Stop();
            _output.WriteLine($"\nGeneration complete in {genSw.Elapsed.TotalSeconds:F1}s (seed={seed})");

            negQwen?.Dispose();
            negClip?.Dispose();

            (nuint freeFinal, _) = backend.Context.GetMemoryInfo();
            _output.WriteLine($"  VRAM final: {freeFinal / (1024 * 1024)} MiB free");

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
            foreach (IDisposable d in tensorOwners) d.Dispose();
        }
    }

    /// <summary>Loads a raw F32 tensor stored as <c>[seq_len, embed_dim]</c> with no header. Wraps in a
    /// batch-of-1 tensor <c>[1, seq_len, embed_dim]</c>.</summary>
    private static unsafe Tensor LoadF32Tensor(string path, int embedDim)
    {
        byte[] data = File.ReadAllBytes(path);
        long totalFloats = data.Length / sizeof(float);
        if (totalFloats % embedDim != 0)
            throw new InvalidOperationException(
                $"Embedding file {path} has {totalFloats} floats which is not a multiple of {embedDim}");

        int seqLen = (int)(totalFloats / embedDim);
        Tensor result = new Tensor(new TensorShape(1, seqLen, embedDim), DType.F32);
        fixed (byte* src = data)
        {
            Buffer.MemoryCopy(src, (void*)result.DataPointer, data.Length, data.Length);
        }
        return result;
    }

    /// <summary>Loads a raw F32 pooled embedding stored as <c>[embed_dim]</c>. Wraps in a batch-of-1
    /// tensor <c>[1, embed_dim]</c>.</summary>
    private static unsafe Tensor LoadPooled(string path, int embedDim)
    {
        byte[] data = File.ReadAllBytes(path);
        long totalFloats = data.Length / sizeof(float);
        if (totalFloats != embedDim)
            throw new InvalidOperationException(
                $"Pooled embedding file {path} has {totalFloats} floats; expected {embedDim}");

        Tensor result = new Tensor(new TensorShape(1, embedDim), DType.F32);
        fixed (byte* src = data)
        {
            Buffer.MemoryCopy(src, (void*)result.DataPointer, data.Length, data.Length);
        }
        return result;
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

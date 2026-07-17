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

/// <summary>
/// End-to-end Flux.2 Klein 4B image generation. Loads the BFL-format Klein 4B transformer,
/// the Qwen3-4B text encoder, and the Flux.2 VAE, runs the full pipeline (Qwen3 multi-layer
/// concat → Flux2Transformer denoising → BN-style un-normalize + 2×2 unpatchify → VAE decode),
/// and saves a BMP for visual inspection.
/// </summary>
[Trait("Category", "Integration")]
public sealed class Flux2GenerationTests
{
    private static string Klein4BPath => TestPaths.Flux2.Klein;
    private static string Qwen3Path => TestPaths.TextEncoders.Qwen3_4B;
    private static string Flux2VaePath => TestPaths.Vae.Flux2;
    private static string Qwen3VocabPath => TestPaths.Tokenizers.Qwen3Vocab;
    private static string Qwen3MergesPath => TestPaths.Tokenizers.Qwen3Merges;
    private static string OutputDir => TestPaths.OutputDir;

    private readonly ITestOutputHelper _output;

    public Flux2GenerationTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Klein 4B end-to-end generation. Long-running test — expects ~1–3 minutes on RTX 3060
    /// (12GB) at 512×512 with 10 steps. Uses BF16 native weights for the transformer (no F32
    /// cast — CudaBackend handles the cast at GEMM time) and F32 cast for the Qwen3 encoder
    /// (matches the existing Qwen3 smoke test's verified path).
    /// </summary>
    [Fact]
    public void Klein4B_GenerateImage_Gpu()
    {
        if (!File.Exists(Klein4BPath))
        {
            _output.WriteLine($"SKIPPED: Klein 4B checkpoint not found: {Klein4BPath}");
            return;
        }
        if (!File.Exists(Qwen3Path))
        {
            _output.WriteLine($"SKIPPED: Qwen3-4B weights not found: {Qwen3Path}");
            return;
        }
        if (!File.Exists(Flux2VaePath))
        {
            _output.WriteLine($"SKIPPED: Flux.2 VAE not found: {Flux2VaePath}");
            return;
        }
        if (!File.Exists(Qwen3VocabPath) || !File.Exists(Qwen3MergesPath))
        {
            _output.WriteLine("SKIPPED: Qwen3 tokenizer files not found");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(Flux2GenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: no CUDA driver available");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();
        Flux2Config config = Flux2Config.Klein4B;
        int mlpInner = (int)(config.HiddenSize * config.MlpRatio);

        // ── 1. Load + cast + convert Klein 4B transformer ─────────────
        // ──    CudaBackend doesn't yet support BF16↔F32 GPU casts, so we pre-cast
        // ──    BF16 weights to F16 on the CPU before conversion. Same byte size
        // ──    (~7.7GB) and F16↔F32 GPU casts already exist, so the GEMM path
        // ──    works (F16 weight cast to F32 per-call to match F32 activations).
        _output.WriteLine($"[1/8] Loading Klein 4B: {Path.GetFileName(Klein4BPath)}");
        Stopwatch sw = Stopwatch.StartNew();
        using SafeTensorsLoader kleinLoader = new SafeTensorsLoader();
        kleinLoader.Load(Klein4BPath);
        Dictionary<string, Tensor> kleinRaw = kleinLoader.GetAllTensors();
        Dictionary<string, Tensor> kleinF16 = new(kleinRaw.Count);
        foreach (KeyValuePair<string, Tensor> kvp in kleinRaw)
        {
            DType d = kvp.Value.DType;
            kleinF16[kvp.Key] = (d == DType.F32 || d == DType.F16) ? kvp.Value : kvp.Value.CastTo(DType.F16);
        }
        kleinRaw.Clear();
        sw.Stop();
        _output.WriteLine($"  Klein loaded + cast to F16 ({kleinF16.Count} tensors) in {sw.ElapsedMilliseconds}ms");

        _output.WriteLine("[2/8] Converting Klein BFL → canonical layout (split QKV / MLP / swap_scale_shift)...");
        sw.Restart();
        Flux2CheckpointConverter converter = new Flux2CheckpointConverter(config.HiddenSize, mlpInner);
        Dictionary<string, Tensor> kleinConverted = converter.ConvertTransformer(kleinF16);
        kleinF16.Clear();
        sw.Stop();
        _output.WriteLine($"  Converted to {kleinConverted.Count} keys in {sw.ElapsedMilliseconds}ms");

        // ── 3. Build Flux2Transformer ─────────────────────────────────
        _output.WriteLine("[3/8] Building Flux2Transformer (F16 weights)...");
        sw.Restart();
        Flux2Transformer transformer = new Flux2Transformer(config);
        transformer.LoadWeights(kleinConverted);
        kleinConverted.Clear();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        sw.Stop();
        _output.WriteLine($"  Transformer loaded in {sw.ElapsedMilliseconds}ms");

        // ── 4. Load Qwen3-4B encoder. CUDA backend doesn't yet implement BF16→F32 GPU
        // ──    casts, so we have to pre-cast the BF16 weights on the CPU. Casting all the
        // ──    way to F32 would peak host memory at ~16GB (above what fits alongside Klein
        // ──    4B's converted weights on a 32GB system). Cast to F16 instead — same 8GB as
        // ──    the original BF16, and F16→F32 is supported on the GPU so the encoder can
        // ──    promote per-GEMM. The encoder's <c>LoadWeights</c> still does small CPU
        // ──    F32 casts on norms / embed table (a few MB) which is fine.
        _output.WriteLine($"[4/8] Loading Qwen3-4B: {Path.GetFileName(Qwen3Path)}");
        sw.Restart();
        using SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
        qwenLoader.Load(Qwen3Path);
        Dictionary<string, Tensor> qwenRaw = qwenLoader.GetAllTensors();
        Dictionary<string, Tensor> qwenCast = new(qwenRaw.Count);
        foreach (KeyValuePair<string, Tensor> kvp in qwenRaw)
        {
            DType d = kvp.Value.DType;
            qwenCast[kvp.Key] = (d == DType.F32 || d == DType.F16) ? kvp.Value : kvp.Value.CastTo(DType.F16);
        }
        qwenRaw.Clear();
        sw.Stop();
        _output.WriteLine($"  Qwen3 loaded + cast to F16 in {sw.ElapsedMilliseconds}ms");

        LlamaStyleEncoder encoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_4B);
        encoder.LoadWeights(qwenCast);
        qwenCast.Clear();
        GC.Collect();
        GC.WaitForPendingFinalizers();

        // ── 5. Load Flux.2 VAE (already F32 in checkpoint) ────────────
        _output.WriteLine($"[5/8] Loading Flux.2 VAE: {Path.GetFileName(Flux2VaePath)}");
        sw.Restart();
        using SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
        vaeLoader.Load(Flux2VaePath);
        Dictionary<string, Tensor> vaeWeights = vaeLoader.GetAllTensors();

        // Extract BN stats — they live at top level alongside the VAE module weights.
        if (!vaeWeights.TryGetValue("bn.running_mean", out Tensor? bnMean))
            throw new InvalidOperationException("Flux.2 VAE checkpoint missing 'bn.running_mean'");
        if (!vaeWeights.TryGetValue("bn.running_var", out Tensor? bnVar))
            throw new InvalidOperationException("Flux.2 VAE checkpoint missing 'bn.running_var'");

        VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Flux2);
        vaeDecoder.LoadWeights(vaeWeights);
        sw.Stop();
        _output.WriteLine($"  VAE loaded in {sw.ElapsedMilliseconds}ms (bn_mean shape={bnMean.Shape}, bn_var shape={bnVar.Shape})");

        // ── 6. Tokenize prompt with chat template ─────────────────────
        _output.WriteLine("[6/8] Tokenizing prompt with Qwen3 chat template...");
        const int MaxLen = 512;
        using Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(Qwen3VocabPath, Qwen3MergesPath, maxLength: MaxLen);
        string prompt = "A photograph of an astronaut riding a horse";
        int[] tokenIds = tokenizer.EncodeChat(prompt);
        _output.WriteLine($"  Prompt: \"{prompt}\" → {tokenIds.Length} tokens (first 16: [{string.Join(", ", tokenIds[..Math.Min(16, tokenIds.Length)])}])");

        // ── 7. Build pipeline + generate ──────────────────────────────
        _output.WriteLine("[7/8] Setting up pipeline + CudaBackend...");
        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);

        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = prompt,
            Width = 512,
            Height = 512,
            Steps = 10,
            Seed = 42,
        };

        Flux2Pipeline pipeline = new Flux2Pipeline(
            backend, encoder, transformer, vaeDecoder,
            bnMean, bnVar, config,
            hiddenLayers: [9, 18, 27],
            bnEps: 1e-5f);

        _output.WriteLine($"[8/8] Generating {request.Width}x{request.Height} image, {request.Steps} steps, seed={request.Seed}...");
        sw.Restart();
        (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
            tokenIds, request,
            guidanceScale: 0f,    // Klein has no guidance embed; ignored
            onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));

        backend.Sync();
        sw.Stop();
        _output.WriteLine($"Generation done in {sw.ElapsedMilliseconds}ms (seed={seed})");

        Directory.CreateDirectory(OutputDir);
        string outputPath = Path.Combine(OutputDir, $"flux2_klein4b_gpu_{width}x{height}_s{request.Steps}_seed{seed}.bmp");
        ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
        _output.WriteLine($"Saved: {outputPath}");

        Assert.Equal(512, width);
        Assert.Equal(512, height);
        Assert.Equal(512 * 512 * 3, rgbData.Length);

        ValidateImageNotDegenerate(rgbData);

        totalSw.Stop();
        _output.WriteLine($"\nTotal test time: {totalSw.ElapsedMilliseconds}ms");

        pipeline.Dispose();
        transformer.Dispose();
        encoder.Dispose();
    }

    /// <summary>Lightweight degeneracy check — confirms the image isn't all black or all white.</summary>
    private static void ValidateImageNotDegenerate(byte[] rgbData)
    {
        int nonZero = 0;
        int nonFf = 0;
        for (int i = 0; i < rgbData.Length; i++)
        {
            if (rgbData[i] != 0) nonZero++;
            if (rgbData[i] != 0xFF) nonFf++;
        }
        float nonZeroPct = 100f * nonZero / rgbData.Length;
        float nonFfPct = 100f * nonFf / rgbData.Length;
        Assert.True(nonZeroPct > 10, $"Image appears to be all black ({nonZeroPct:F1}% non-zero)");
        Assert.True(nonFfPct > 10, $"Image appears to be all white ({nonFfPct:F1}% non-FF)");
    }
}

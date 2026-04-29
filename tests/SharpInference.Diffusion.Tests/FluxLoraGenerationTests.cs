using System.Diagnostics;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.Lora;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace SharpInference.Diffusion.Tests;

/// <summary>End-to-end Flux+LoRA generation tests. The primary path validates AI Toolkit (ostris/ai-toolkit) format LoRAs since that is the dominant Flux trainer; secondary tests cover Kohya and diffusers PEFT formats. All tests skip cleanly when the appropriate FLUX_*_LORA_PATH env var is not set.</summary>
public sealed class FluxLoraGenerationTests
{
    private static string PickPath(string envVar, string winPath, string linuxPath)
    {
        string? v = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrEmpty(v)) return v;
        return OperatingSystem.IsWindows() ? winPath : linuxPath;
    }

    private static readonly string LinuxRepoRoot =
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

    private static readonly string FluxSingleFilePath = PickPath(
        "FLUX_SINGLE_FILE_PATH",
        @"C:\Users\kaleb\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\Flux\flux1-schnell-fp8.safetensors",
        Path.Combine(LinuxRepoRoot, "Models", "flux1-schnell-fp8.safetensors"));

    private static readonly string TokenizerVocabPath = PickPath(
        "CLIP_VOCAB_PATH",
        @"C:\Users\kaleb\Desktop\projects\SharpInference\tests\test-models\clip_vocab.json",
        Path.Combine(LinuxRepoRoot, "tests", "test-models", "clip_vocab.json"));

    private static readonly string TokenizerMergesPath = PickPath(
        "CLIP_MERGES_PATH",
        @"C:\Users\kaleb\Desktop\projects\SharpInference\tests\test-models\clip_merges.txt",
        Path.Combine(LinuxRepoRoot, "tests", "test-models", "clip_merges.txt"));

    private static readonly string T5SpieceModelPath = PickPath(
        "T5_SPIECE_MODEL_PATH",
        @"C:\Users\kaleb\Desktop\projects\SharpInference\tests\test-models\t5_spiece.model",
        Path.Combine(LinuxRepoRoot, "tests", "test-models", "t5_spiece.model"));

    private static readonly string OutputDir = PickPath(
        "FLUX_OUTPUT_DIR",
        @"C:\Users\kaleb\Desktop\projects\SharpInference\Output",
        Path.Combine(LinuxRepoRoot, "Output"));

    private static readonly string? FluxAiToolkitLoraPath =
        Environment.GetEnvironmentVariable("FLUX_AITOOLKIT_LORA_PATH");

    private static readonly string? FluxKohyaLoraPath =
        Environment.GetEnvironmentVariable("FLUX_KOHYA_LORA_PATH");

    private static readonly string? FluxDiffusersLoraPath =
        Environment.GetEnvironmentVariable("FLUX_DIFFUSERS_LORA_PATH");

    private readonly ITestOutputHelper _output;

    public FluxLoraGenerationTests(ITestOutputHelper output) => _output = output;

    /// <summary>Primary AI Toolkit Flux LoRA test. Skips when FLUX_AITOOLKIT_LORA_PATH is not set. Validates the F4 format path (lora_transformer_ prefix + .lora_A.weight / .lora_B.weight suffixes).</summary>
    [Fact]
    public void AiToolkit_Flux_WithLora_GenerateImage_Cpu_Small()
    {
        RunFluxLoraTest(FluxAiToolkitLoraPath, "FLUX_AITOOLKIT_LORA_PATH", LoraFormat.AiToolkitFlux, "aitk_lora");
    }

    /// <summary>GPU end-to-end visual diff. Generates the same prompt+seed twice — once baseline (no LoRA), once with the yearbook LoRA merged into the FP8 transformer weights via LoraStack. Saves both BMPs side-by-side so the LoRA effect can be visually confirmed. Skips when FLUX_DIFFUSERS_LORA_PATH (or FLUX_AITOOLKIT_LORA_PATH) is not set or the PTX directory is missing.</summary>
    [Fact]
    public void Yearbook_Flux_BaselineVsLora_GenerateImage_Gpu()
    {
        string? loraPath = FluxDiffusersLoraPath ?? FluxAiToolkitLoraPath;
        if (string.IsNullOrEmpty(loraPath) || !File.Exists(loraPath))
        {
            _output.WriteLine("SKIPPED: set FLUX_DIFFUSERS_LORA_PATH (or FLUX_AITOOLKIT_LORA_PATH).");
            return;
        }
        if (!File.Exists(FluxSingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Flux checkpoint not found: {FluxSingleFilePath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found.");
            return;
        }
        if (!File.Exists(T5SpieceModelPath))
        {
            _output.WriteLine("SKIPPED: T5 SentencePiece model not found.");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(FluxLoraGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/9] Loading Flux checkpoint (mmap, FP8 native): {Path.GetFileName(FluxSingleFilePath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(FluxSingleFilePath);
        sw.Stop();
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms (transformer={converted.Transformer.Count} keys)");

        using (loader)
        {
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);
            Directory.CreateDirectory(OutputDir);

            (int dB, int sB, bool hasG) = FluxCheckpointConverter.DetectArchitecture(converted.Transformer);
            FluxConfig config = hasG ? FluxConfig.Dev : FluxConfig.Schnell;
            _output.WriteLine($"  Architecture: {dB} double + {sB} single, guidance={hasG}");

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);

            using ClipTokenizer clipTokenizer = new(TokenizerVocabPath, TokenizerMergesPath);
            using T5Tokenizer t5Tokenizer = new(T5SpieceModelPath, maxLength: 256);
            string prompt = "woman playing the guitar, on stage, singing a song, laser lights, punk rocker";
            int[] clipTokens = clipTokenizer.Encode(prompt);
            int eosPos = ClipTokenizer.FindEosPosition(clipTokens);
            int[] t5Tokens = t5Tokenizer.Encode(prompt);
            int[] t5Mask = T5Tokenizer.CreateAttentionMask(t5Tokens);

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                Width = 512,
                Height = 512,
                Steps = 4,
                Seed = 42,
            };

            // ─── Phase 1: baseline ────────────────────────────────────────────
            _output.WriteLine("[2/9] Loading models for baseline pass...");
            sw.Restart();
            FluxTransformer baselineTransformer = new(config);
            baselineTransformer.LoadWeights(converted.Transformer);
            ClipTextEncoder baselineClipL = new(ClipTextEncoderConfig.SdxlClipL);
            baselineClipL.LoadWeights(converted.ClipL, "text_model");
            T5TextEncoder baselineT5 = new(T5TextEncoderConfig.Xxl);
            baselineT5.LoadWeights(converted.T5);
            VaeDecoder baselineVae = new(VaeConfig.Flux);
            baselineVae.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  Models built in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine($"[3/9] Baseline generation (no LoRA, {request.Width}x{request.Height}, {request.Steps} steps, seed={request.Seed})...");
            FluxPipeline baselinePipe = new(backend, baselineClipL, baselineT5, baselineTransformer, baselineVae, config);
            sw.Restart();
            (byte[] baselineRgb, int bw, int bh, int _) = baselinePipe.GenerateFromTokens(
                clipTokens, eosPos, t5Tokens, t5Mask,
                request, guidanceScale: 0.0f,
                onProgress: p => _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
            backend.Sync();
            sw.Stop();
            _output.WriteLine($"  Baseline generated in {sw.ElapsedMilliseconds}ms");

            string baselinePath = Path.Combine(OutputDir, $"yearbook_baseline_{bw}x{bh}_seed{request.Seed}.bmp");
            ImagePostProcessor.SaveBmp(baselinePath, baselineRgb, bw, bh);
            _output.WriteLine($"  Saved baseline: {baselinePath}");
            ValidateImageNotDegenerate(baselineRgb);

            baselinePipe.Dispose();
            baselineTransformer.Dispose();
            baselineT5.Dispose();

            // ─── Phase 2: merge LoRA, regenerate ──────────────────────────────
            _output.WriteLine($"[4/9] Loading LoRA: {Path.GetFileName(loraPath)}");
            using LoraFile lora = LoraFile.Load(loraPath);
            _output.WriteLine($"  format={lora.Format}, layers={lora.Layers.Count}");

            _output.WriteLine("[5/9] Merging LoRA via LoraStack (CpuBackend, FP8→F32→FP8)...");
            using CpuBackend cpuForMerge = new();
            using LoraStack stack = new();
            stack.Add(lora, strength: 1.0f);
            sw.Restart();
            int merged = stack.ApplyTo(converted.Transformer, LoraTarget.Transformer, cpuForMerge);
            sw.Stop();
            _output.WriteLine($"  Merged {merged} transformer weights in {sw.Elapsed.TotalSeconds:F1}s");
            Assert.True(merged > 0, "expected the LoRA to merge into at least one transformer weight");

            _output.WriteLine("[6/9] Reloading transformer with merged dict...");
            sw.Restart();
            FluxTransformer loraTransformer = new(config);
            loraTransformer.LoadWeights(converted.Transformer);
            ClipTextEncoder loraClipL = new(ClipTextEncoderConfig.SdxlClipL);
            loraClipL.LoadWeights(converted.ClipL, "text_model");
            T5TextEncoder loraT5 = new(T5TextEncoderConfig.Xxl);
            loraT5.LoadWeights(converted.T5);
            VaeDecoder loraVae = new(VaeConfig.Flux);
            loraVae.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  Models rebuilt in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine($"[7/9] LoRA generation (same prompt + seed)...");
            FluxPipeline loraPipe = new(backend, loraClipL, loraT5, loraTransformer, loraVae, config);
            sw.Restart();
            (byte[] loraRgb, int lw, int lh, int _) = loraPipe.GenerateFromTokens(
                clipTokens, eosPos, t5Tokens, t5Mask,
                request, guidanceScale: 0.0f,
                onProgress: p => _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
            backend.Sync();
            sw.Stop();
            _output.WriteLine($"  LoRA generated in {sw.ElapsedMilliseconds}ms");

            string loraPathBmp = Path.Combine(OutputDir, $"yearbook_with_lora_{lw}x{lh}_seed{request.Seed}.bmp");
            ImagePostProcessor.SaveBmp(loraPathBmp, loraRgb, lw, lh);
            _output.WriteLine($"  Saved LoRA: {loraPathBmp}");
            ValidateImageNotDegenerate(loraRgb);

            // ─── Phase 3: confirm the LoRA actually changed something ─────────
            _output.WriteLine("[8/9] Comparing baseline vs LoRA outputs...");
            Assert.Equal(baselineRgb.Length, loraRgb.Length);
            long sumDiff = 0;
            int diffPixels = 0;
            for (int i = 0; i < baselineRgb.Length; i++)
            {
                int d = Math.Abs(baselineRgb[i] - loraRgb[i]);
                sumDiff += d;
                if (d > 5) diffPixels++;
            }
            float meanAbsDiff = sumDiff / (float)baselineRgb.Length;
            float diffPct = diffPixels / (float)baselineRgb.Length * 100;
            _output.WriteLine($"  Mean abs byte-diff: {meanAbsDiff:F2}");
            _output.WriteLine($"  Pixels with diff >5: {diffPct:F1}%");
            Assert.True(meanAbsDiff > 2.0f, $"LoRA had no visible effect (mean abs diff = {meanAbsDiff:F2})");

            _output.WriteLine($"[9/9] Compare:");
            _output.WriteLine($"  baseline:  {baselinePath}");
            _output.WriteLine($"  with LoRA: {loraPathBmp}");

            loraPipe.Dispose();
            loraTransformer.Dispose();
            loraT5.Dispose();
            totalSw.Stop();
            _output.WriteLine($"\nTotal: {totalSw.Elapsed.TotalMinutes:F1} min");
        }
    }

    /// <summary>Lightweight key-coverage validation: load a Flux LoRA and the Flux checkpoint header (no weight cast), assert every LoRA target key exists in the converted transformer weight dictionary. Validates the mapper end-to-end on real-world LoRA files without running inference. Skips when FLUX_DIFFUSERS_LORA_PATH (or FLUX_AITOOLKIT_LORA_PATH as a fallback) is not set.</summary>
    [Fact]
    public void Flux_Lora_KeyCoverage_AgainstRealCheckpoint()
    {
        string? loraPath = FluxDiffusersLoraPath ?? FluxAiToolkitLoraPath ?? FluxKohyaLoraPath;
        if (string.IsNullOrEmpty(loraPath) || !File.Exists(loraPath))
        {
            _output.WriteLine("SKIPPED: set FLUX_DIFFUSERS_LORA_PATH (or FLUX_AITOOLKIT_LORA_PATH / FLUX_KOHYA_LORA_PATH).");
            return;
        }
        if (!File.Exists(FluxSingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Flux checkpoint not found: {FluxSingleFilePath}");
            return;
        }

        _output.WriteLine($"Loading LoRA: {Path.GetFileName(loraPath)}");
        using LoraFile lora = LoraFile.Load(loraPath);
        _output.WriteLine($"  format={lora.Format}, layers={lora.Layers.Count}");

        _output.WriteLine($"Loading Flux checkpoint header (mmap): {Path.GetFileName(FluxSingleFilePath)}");
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(FluxSingleFilePath);
        using (loader)
        {
            HashSet<string> transformerKeys = [.. converted.Transformer.Keys];
            HashSet<string> clipLKeys = [.. converted.ClipL.Keys];
            _output.WriteLine($"  transformer keys: {transformerKeys.Count}, clip-L keys: {clipLKeys.Count}");

            int xfmHits = 0, xfmMiss = 0, clipHits = 0, clipMiss = 0;
            List<string> firstMisses = [];
            foreach (LoraLayer layer in lora.Layers)
            {
                bool hit = layer.Target switch
                {
                    LoraTarget.Transformer => transformerKeys.Contains(layer.TargetKey),
                    LoraTarget.ClipL => clipLKeys.Contains(layer.TargetKey),
                    _ => false,
                };
                if (layer.Target == LoraTarget.Transformer) { if (hit) xfmHits++; else xfmMiss++; }
                else if (layer.Target == LoraTarget.ClipL) { if (hit) clipHits++; else clipMiss++; }
                if (!hit && firstMisses.Count < 10)
                {
                    firstMisses.Add($"{layer.Target}: {layer.TargetKey}");
                }
            }

            _output.WriteLine($"Coverage:");
            _output.WriteLine($"  Transformer: {xfmHits} hits / {xfmMiss} misses");
            _output.WriteLine($"  CLIP-L:      {clipHits} hits / {clipMiss} misses");
            if (firstMisses.Count > 0)
            {
                _output.WriteLine("First misses (up to 10):");
                foreach (string m in firstMisses) _output.WriteLine($"  {m}");

                _output.WriteLine("Sample transformer keys (for reference):");
                foreach (string k in transformerKeys.OrderBy(x => x).Take(8)) _output.WriteLine($"  {k}");
            }

            // Hard assertion: every layer must hit, otherwise the mapper is wrong for this real-world file.
            Assert.Equal(0, xfmMiss);
            Assert.Equal(0, clipMiss);
            Assert.True(xfmHits + clipHits > 0, "no LoRA layers matched any model key");
        }
    }

    /// <summary>Secondary Kohya Flux LoRA test. Skips when FLUX_KOHYA_LORA_PATH is not set. Validates the F3 format path (lora_unet_double_blocks_*_qkv with 3-way fused-QKV split).</summary>
    [Fact]
    public void Kohya_Flux_WithLora_GenerateImage_Cpu_Small()
    {
        RunFluxLoraTest(FluxKohyaLoraPath, "FLUX_KOHYA_LORA_PATH", LoraFormat.KohyaFlux, "kohya_lora");
    }

    /// <summary>HuggingFace PEFT diffusers Flux LoRA test. Skips when FLUX_DIFFUSERS_LORA_PATH is not set. Validates the F5 format path (transformer.*.lora_A.weight identity-style mapping).</summary>
    [Fact]
    public void Diffusers_Flux_WithLora_GenerateImage_Cpu_Small()
    {
        RunFluxLoraTest(FluxDiffusersLoraPath, "FLUX_DIFFUSERS_LORA_PATH", LoraFormat.DiffusersFlux, "diffusers_lora");
    }

    private void RunFluxLoraTest(string? loraPath, string envVarName, LoraFormat expectedFormat, string outputTag)
    {
        if (string.IsNullOrEmpty(loraPath) || !File.Exists(loraPath))
        {
            _output.WriteLine($"SKIPPED: Set {envVarName} to a {expectedFormat} LoRA .safetensors file (got '{loraPath}').");
            return;
        }
        if (!File.Exists(FluxSingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Flux checkpoint not found: {FluxSingleFilePath}");
            return;
        }
        if (!File.Exists(TokenizerVocabPath) || !File.Exists(TokenizerMergesPath))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found.");
            return;
        }
        if (!File.Exists(T5SpieceModelPath))
        {
            _output.WriteLine("SKIPPED: T5 SentencePiece model not found.");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/8] Loading Flux checkpoint: {Path.GetFileName(FluxSingleFilePath)}");
        Stopwatch sw = Stopwatch.StartNew();
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(FluxSingleFilePath);
        sw.Stop();
        _output.WriteLine($"  Loaded in {sw.ElapsedMilliseconds}ms (transformer={converted.Transformer.Count} keys)");

        using (loader)
        {
            // Cast all weights to F32. LoRA merging requires writable weights and identical dtype on both
            // operands; FP8 base is rejected by LoraStack (see LORA_KEY_MAPPING.md "FP8 base weights" row).
            // Note: this can use a lot of RAM for Flux Dev (~48 GB). Set FLUX_SINGLE_FILE_PATH to a smaller
            // checkpoint or run on a machine with enough RAM.
            Dictionary<string, Tensor> transformerF32 = CastWeightsToF32(converted.Transformer);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> t5F32 = CastWeightsToF32(converted.T5);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            using CpuBackend backend = new();

            _output.WriteLine($"[2/8] Loading LoRA: {Path.GetFileName(loraPath!)}");
            sw.Restart();
            using LoraFile lora = LoraFile.Load(loraPath!);
            sw.Stop();
            _output.WriteLine($"  LoRA loaded in {sw.ElapsedMilliseconds}ms (format={lora.Format}, layers={lora.Layers.Count})");
            Assert.Equal(expectedFormat, lora.Format);

            using LoraStack stack = new();
            stack.Add(lora, strength: 1.0f);

            _output.WriteLine("[3/8] Merging LoRA into Flux transformer + CLIP-L...");
            sw.Restart();
            int xfmMerged = stack.ApplyTo(transformerF32, LoraTarget.Transformer, backend);
            int clipMerged = stack.ApplyTo(clipLF32, LoraTarget.ClipL, backend);
            sw.Stop();
            _output.WriteLine($"  Merged {xfmMerged} transformer + {clipMerged} CLIP-L weights in {sw.ElapsedMilliseconds}ms");

            (int doubleBlocks, int singleBlocks, bool hasGuidance) =
                FluxCheckpointConverter.DetectArchitecture(converted.Transformer);
            FluxConfig config = hasGuidance ? FluxConfig.Dev : FluxConfig.Schnell;
            _output.WriteLine($"[4/8] Architecture: {doubleBlocks} double + {singleBlocks} single, guidance={hasGuidance}");

            _output.WriteLine("[5/8] Building model components...");
            sw.Restart();
            FluxTransformer transformer = new(config);
            transformer.LoadWeights(transformerF32);
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");
            T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(t5F32);
            VaeDecoder vae = new(VaeConfig.Flux);
            vae.LoadWeights(vaeF32);
            sw.Stop();
            _output.WriteLine($"  Models built in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine("[6/8] Tokenizing prompt...");
            using ClipTokenizer clipTok = new(TokenizerVocabPath, TokenizerMergesPath);
            string prompt = "A photograph of an astronaut riding a horse";
            int[] clipTokens = clipTok.Encode(prompt);
            int eosPos = ClipTokenizer.FindEosPosition(clipTokens);
            using T5Tokenizer t5Tok = new(T5SpieceModelPath, maxLength: 256);
            int[] t5Tokens = t5Tok.Encode(prompt);
            int[] t5Mask = T5Tokenizer.CreateAttentionMask(t5Tokens);

            TextToImageRequest request = new()
            {
                Prompt = prompt,
                Width = 256,
                Height = 256,
                Steps = hasGuidance ? 8 : 4,
                Seed = 42,
            };

            _output.WriteLine($"[7/8] Generating {request.Width}x{request.Height}, {request.Steps} steps...");
            FluxPipeline pipeline = new(backend, clipL, t5, transformer, vae, config);
            Stopwatch genSw = Stopwatch.StartNew();
            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                clipTokens, eosPos, t5Tokens, t5Mask,
                request, guidanceScale: hasGuidance ? 3.5f : 0.0f,
                onProgress: p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
            genSw.Stop();
            _output.WriteLine($"  Generation: {genSw.Elapsed.TotalMinutes:F1} min");

            Assert.Equal(256, width);
            Assert.Equal(256, height);
            Assert.Equal(256 * 256 * 3, rgbData.Length);
            ValidateImageNotDegenerate(rgbData);

            _output.WriteLine("[8/8] Saving output...");
            Directory.CreateDirectory(OutputDir);
            string outPath = Path.Combine(OutputDir, $"flux_{outputTag}_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outPath, rgbData, width, height);
            _output.WriteLine($"  Saved: {outPath}");

            pipeline.Dispose();
            transformer.Dispose();
            t5.Dispose();
            totalSw.Stop();
            _output.WriteLine($"\nTotal: {totalSw.Elapsed.TotalMinutes:F1} min");
        }
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> result = new(weights.Count);
        foreach ((string k, Tensor t) in weights)
        {
            result[k] = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        }
        return result;
    }

    private static void ValidateImageNotDegenerate(byte[] rgb)
    {
        int nonZero = 0, nonFF = 0;
        foreach (byte b in rgb) { if (b != 0) nonZero++; if (b != 255) nonFF++; }
        float nzPct = nonZero / (float)rgb.Length * 100;
        float nfPct = nonFF / (float)rgb.Length * 100;
        Assert.True(nzPct > 10, $"image is all black ({nzPct:F1}% non-zero)");
        Assert.True(nfPct > 10, $"image is all white ({nfPct:F1}% non-255)");
    }
}

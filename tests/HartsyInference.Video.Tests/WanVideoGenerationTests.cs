using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.Lora;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>End-to-end Wan2.2 TI2V-5B T2V generation against a real checkpoint: umT5-XXL encode (embedded
/// umT5 spiece) → <see cref="WanVideoPipeline"/> → BMP frame sequence. Skips cleanly when the DiT single-file /
/// Wan2.2 VAE / umT5 weights / PTX dir are missing, or VRAM is insufficient. The manual first-run
/// validation entry — numerics are validation-pending (structure is verified by <see cref="WanVideoPipelineTests"/>).</summary>
public class WanVideoGenerationTests
{
    private readonly ITestOutputHelper _output;
    public WanVideoGenerationTests(ITestOutputHelper output) => _output = output;

    private static double EnvD(string name, double fallback) =>
        double.TryParse(Environment.GetEnvironmentVariable(name), out double v) ? v : fallback;

    [Fact]
    public async Task Wan22Ti2V5B_Gpu_T2V_480p_ShortClip()
    {
        string ckpt = TestPaths.WanVideo.Ti2V5B, vaePath = TestPaths.WanVideo.VaePath;
        string umt5Path = TestPaths.WanVideo.Umt5Xxl;
        if (!File.Exists(ckpt)) { _output.WriteLine($"SKIPPED: Wan2.2 TI2V-5B checkpoint not found: {ckpt} (set WAN22_TI2V_5B_PATH)."); return; }
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: Wan2.2 VAE safetensors not found: {vaePath} (set WAN22_VAE_PATH)."); return; }
        if (!File.Exists(umt5Path)) { _output.WriteLine($"SKIPPED: umT5-XXL weights not found: {umt5Path} (set UMT5_XXL_PATH)."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(WanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/5] Loading + converting Wan DiT: {Path.GetFileName(ckpt)}");
        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) =
            WanVideoCheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _dit = ditLoader;
        _output.WriteLine($"  {conv.Transformer.Count} keys in {sw.ElapsedMilliseconds}ms");

        List<SafeTensorsLoader> loaders = [];
        try
        {
            _output.WriteLine($"[2/5] Loading Wan2.2 VAE + umT5-XXL...");
            sw.Restart();
            (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vl);
            using SafeTensorsLoader umt5Loader = new();
            umt5Loader.Load(umt5Path);
            Dictionary<string, Tensor> umt5W = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());
            _output.WriteLine($"  VAE {vaeW.Count} keys, umT5 {umt5W.Count} keys in {sw.ElapsedMilliseconds}ms");

            // Optional LoRA merge (kohya/musubi, Comfy diffusion_model., or diffusers-PEFT — auto-detected).
            // Merged on CPU before LoadWeights; the stack owns the merged tensors for the run's lifetime.
            using LoraStack loraStack = new();
            if (File.Exists(TestPaths.WanVideo.LoraPath))
            {
                float strength = float.TryParse(Environment.GetEnvironmentVariable("WAN_LORA_STRENGTH"), out float s) ? s : 1.0f;
                loraStack.AddFromPath(TestPaths.WanVideo.LoraPath, strength);
                using HartsyInference.Cpu.CpuBackend mergeBackend = new();
                int mergedCount = loraStack.ApplyToWeights(mergeBackend, transformerWeights: conv.Transformer);
                _output.WriteLine($"  LoRA: {Path.GetFileName(TestPaths.WanVideo.LoraPath)} @ {strength} → {mergedCount} weights merged");
            }

            WanVideoConfig cfg = WanVideoConfig.Ti2V5B;
            using WanVideoTransformer transformer = new(cfg);
            transformer.LoadWeights(conv.Transformer);
            Wan22VaeDecoder vae = new();
            vae.LoadWeights(vaeW);
            using T5TextEncoder umt5 = new(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5W);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            // WAN_NO_CACHE_CASTS=1 → dequant fp16 weights transiently per GEMM instead of caching the ~9.4 GB of
            // BF16 casts (trades a re-cast per forward for headroom; needed if full-res peak sits near 24 GB).
            if (Environment.GetEnvironmentVariable("WAN_NO_CACHE_CASTS") == "1") backend.CacheWeightCasts = false;
            (nuint freeBytes, _) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            // Env overrides let a tiny debug clip run on the 3060 (WAN_MIN_GB=10 WAN_W=320 WAN_H=192 WAN_FRAMES=9 WAN_STEPS=4).
            double MinGb = EnvD("WAN_MIN_GB", 14.0);
            if (freeGb < MinGb) { _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM; Wan2.2 TI2V-5B needs ≥{MinGb} GB."); return; }

            _output.WriteLine($"[3/5] umT5 encode...");
            sw.Restart();
            using T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: 512);
            int[] promptTokens = tokenizer.Encode("a cinematic shot of a cat walking through a sunlit garden, shallow depth of field");
            int[] negTokens = tokenizer.Encode("blurry, low quality, distorted, watermark");
            Tensor batch = umt5.Encode(backend,
                [promptTokens, negTokens],
                [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
            int seqLen = promptTokens.Length;
            Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, seqLen, 4096);
            Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, seqLen, 4096);
            batch.Dispose();
            // Reclaim umT5 VRAM before the 10 GB DiT preload (the pipeline preloads the transformer itself).
            backend.Sync();
            backend.FreeWeights(umt5.EnumerateWeights());
            _output.WriteLine($"  encoded in {sw.ElapsedMilliseconds}ms");

            int genW = (int)EnvD("WAN_W", 832), genH = (int)EnvD("WAN_H", 480);
            int genFrames = (int)EnvD("WAN_FRAMES", 33), genSteps = (int)EnvD("WAN_STEPS", 30);
            _output.WriteLine($"[4/5] Generating {genFrames}-frame {genW}x{genH} clip, {genSteps} steps (NUMERIC OUTPUT VALIDATION-PENDING)...");
            WanVideoPipeline pipeline = new(backend, transformer, vae, cfg);
            TextToImageRequest req = new() { Prompt = "cat", Width = genW, Height = genH, Steps = genSteps, CfgScale = cfg.GuidanceScale, Seed = 42 };
            string outDir = Path.Combine(TestPaths.OutputDir, $"wan_video_{DateTime.Now:yyyyMMdd_HHmmss}");
            await new BmpSequenceEncoder().EncodeAsync(
                pipeline.GenerateFramesAsync(promptEmbeds, negEmbeds, req, numFrames: genFrames,
                    p => _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)")),
                outDir, fps: 24);
            promptEmbeds.Dispose();
            negEmbeds.Dispose();

            _output.WriteLine($"[5/5] Wrote frames → {outDir}");
            Assert.Equal(genFrames, Directory.GetFiles(outDir, "frame_*.bmp").Length);
        }
        finally
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
        }
    }

    /// <summary>Isolation diagnostic for the all-zero video output: decode a host-created NON-ZERO random latent
    /// through the real-weight <see cref="Wan22VaeDecoder"/> alone (no transformer). If the decoded RGB is all-zero,
    /// the decoder is the bug; if it is non-zero, the zero originates upstream in the transformer/denoise. Uses the
    /// exact host-read path the failing pipeline uses (<see cref="VideoRgbFrames.ExtractFrame"/>). GPU-only.</summary>
    [Fact]
    public void Wan22Vae_Decode_RandomLatent_NonZero_Gpu()
    {
        string vaePath = TestPaths.WanVideo.VaePath;
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: Wan2.2 VAE not found: {vaePath}"); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(WanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA"); return; }

        (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) = LanceCheckpointConverter.LoadVae(vaePath);
        try
        {
            Wan22VaeDecoder vae = new();
            vae.LoadWeights(vaeW);
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);

            // Host-built NON-ZERO latent [1,48,1,12,20] (single image frame → smallest full spatial decode).
            int T = 1, hLat = 12, wLat = 20;
            Tensor latent = new(new TensorShape([1L, 48, T, hLat, wLat]), DType.F32);
            Span<float> ls = latent.AsSpan<float>();
            Random rng = new(7);
            for (int i = 0; i < ls.Length; i++) ls[i] = (float)(rng.NextDouble() * 2 - 1);
            double inAbs = 0; for (int i = 0; i < ls.Length; i++) inAbs += Math.Abs(ls[i]);
            _output.WriteLine($"[in] latent [1,48,{T},{hLat},{wLat}] meanAbs={inAbs / ls.Length:F5}");

            Tensor rgb = vae.Decode(backend, latent);
            backend.Sync();
            _output.WriteLine($"[out] decoded rank={rgb.Shape.Rank} dims=[{string.Join(",", Enumerable.Range(0, rgb.Shape.Rank).Select(d => (long)rgb.Shape[d]))}]");

            byte[] frame0 = VideoRgbFrames.ExtractFrame(rgb, 0);
            int nz = 0, nff = 0; long sum = 0;
            foreach (byte b in frame0) { if (b != 0) nz++; if (b != 255) nff++; sum += b; }
            _output.WriteLine($"[out] frame0 {frame0.Length} bytes: nonzero={nz} ({100.0 * nz / frame0.Length:F1}%), non255={nff}, mean={(double)sum / frame0.Length:F2}");

            Assert.True(nz > 0, "DECODER BUG: real-weight decode of a NON-ZERO latent produced ALL-ZERO RGB (zero is decoder-side, not the transformer).");
            _output.WriteLine("VERDICT: decoder produces non-zero RGB from a non-zero latent → the all-zero video output is UPSTREAM (transformer/denoise), not the VAE decoder.");
        }
        finally { foreach (SafeTensorsLoader l in vl) l.Dispose(); }
    }
}

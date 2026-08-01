using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>End-to-end LTX-2 dual-stream (video+audio) T2V generation against the bundled 19B fp8 single file
/// (DiT + video VAE + audio VAE + vocoder) plus the external Gemma-3-12B fp8 text encoder → <see cref="LtxVideo2Pipeline"/>
/// → BMP frame sequence (+ optional waveform). Skips cleanly when artifacts are missing or VRAM is insufficient.
/// First-run validation entry — numerics validation-pending.</summary>
[Trait("Category", "Integration")]
public class LtxVideo2GenerationTests
{
    private readonly ITestOutputHelper _output;
    public LtxVideo2GenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task LtxVideo2_Gpu_T2VA_ShortClip()
    {
        string ckpt = TestPaths.LtxVideo2.SingleFile;
        string gemmaPath = TestPaths.LtxVideo2.GemmaEncoder, gemmaTok = TestPaths.LtxVideo2.GemmaTokenizer;
        if (!File.Exists(ckpt)) { _output.WriteLine($"SKIPPED: LTX-2 checkpoint not found: {ckpt} (set LTX2_PATH)."); return; }
        if (!File.Exists(gemmaPath)) { _output.WriteLine($"SKIPPED: Gemma-3-12B encoder not found: {gemmaPath} (set LTX2_GEMMA_PATH)."); return; }
        if (!File.Exists(gemmaTok)) { _output.WriteLine($"SKIPPED: Gemma tokenizer not found: {gemmaTok} (set LTX2_GEMMA_TOKENIZER)."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(LtxVideo2GenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/5] Loading + converting LTX-2 single file: {Path.GetFileName(ckpt)}");
        (LtxVideo2CheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ckptLoader) =
            LtxVideo2CheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _ckpt = ckptLoader;
        _output.WriteLine($"  DiT {conv.Transformer.Count} / connectors {conv.Connectors.Count} / vae {conv.Vae.Count} / " +
            $"audioVae {conv.AudioVae.Count} / vocoder {conv.Vocoder.Count} keys in {sw.ElapsedMilliseconds}ms");
        if (conv.Vae.Count == 0)
        {
            _output.WriteLine("SKIPPED: transformer-only split file (no bundled VAE) — this test needs a bundled checkpoint; the split path is covered by the ltx-2 recipe.");
            return;
        }

        LtxVideo2Config cfg = LtxVideo2Config.V23;
        using LtxVideo2Transformer transformer = new(cfg);
        transformer.LoadWeights(conv.Transformer);
        LtxVideo2TextConnectors connectors = new(cfg);
        connectors.LoadWeights(conv.Connectors);
        // Latent un-normalization stats (per-channel) — the diffusion model works in normalized latent space, so the
        // decoder must apply z = latent·std + mean. Without these the VAE gets a wrongly-scaled latent → garbage.
        float[]? latentsMean = ExtractStat(conv.Vae, "latents_mean");
        float[]? latentsStd = ExtractStat(conv.Vae, "latents_std");
        LtxVideo2VaeDecoder vae = new(latentsMean: latentsMean, latentsStd: latentsStd);
        vae.LoadWeights(CastWeightsToF32(conv.Vae));
        LtxAudioVaeDecoder? audioVae = null;
        LtxAudioVocoder? vocoder = null;
        float[]? audioLatentsMean = null, audioLatentsStd = null;
        if (conv.AudioVae.Count > 0 && conv.Vocoder.Count > 0)
        {
            audioVae = new LtxAudioVaeDecoder();
            audioVae.LoadWeights(CastWeightsToF32(conv.AudioVae));
            vocoder = new LtxAudioVocoder();
            vocoder.LoadWeights(CastWeightsToF32(conv.Vocoder));
            // Audio latent stats ([128] over the packed 8×16 latent) — the pipeline denormalizes before unpacking.
            audioLatentsMean = ExtractStat(conv.AudioVae, "latents_mean");
            audioLatentsStd = ExtractStat(conv.AudioVae, "latents_std");
        }

        _output.WriteLine($"[2/5] Loading Gemma-3-12B from {Path.GetFileName(gemmaPath)}...");
        sw.Restart();
        using SafeTensorsLoader gemmaLoader = new();
        gemmaLoader.Load(gemmaPath);
        using LlamaStyleEncoder gemma = new(LlamaStyleEncoderConfig.Gemma3_12B);
        gemma.LoadWeights(gemmaLoader.GetAllTensors());
        _output.WriteLine($"  Gemma loaded in {sw.ElapsedMilliseconds}ms");

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
        backend.CacheWeightCasts = false;   // fp8 stays fp8-resident; caching F16 casts would OOM.
        (nuint freeBytes, _) = backend.Context.GetMemoryInfo();
        double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
        // Block-swap streams the ~22 GB DiT, so peak VRAM is the Gemma encode (~13 GB) + a small streaming window.
        const double MinGb = 15.0;
        if (freeGb < MinGb) { _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM; LTX-2 needs ≥{MinGb} GB (block-swapped DiT)."); return; }

        using GemmaTokenizer tokenizer = new(gemmaTok, maxLength: 256);
        int[] promptTokens = tokenizer.Encode("a cinematic shot of a cat walking through a sunlit garden, birds chirping");
        int[] negTokens = tokenizer.Encode("blurry, low quality, distorted, watermark");

        int steps = int.TryParse(Environment.GetEnvironmentVariable("LTX2_STEPS"), out int st) ? st : 20;
        int numFrames = int.TryParse(Environment.GetEnvironmentVariable("LTX2_FRAMES"), out int nf) ? nf : 25;
        int width = int.TryParse(Environment.GetEnvironmentVariable("LTX2_W"), out int w) ? w : 512;
        int height = int.TryParse(Environment.GetEnvironmentVariable("LTX2_H"), out int h) ? h : 320;
        _output.WriteLine($"[3/5] Generating {numFrames}f {width}x{height}, {steps} steps (NUMERIC OUTPUT VALIDATION-PENDING)...");
        LtxVideo2Pipeline pipeline = new(backend, transformer, connectors, vae, gemma, cfg, audioVae, vocoder,
            audioLatentsMean, audioLatentsStd);
        TextToImageRequest req = new() { Prompt = "cat", Width = width, Height = height, Steps = steps, CfgScale = cfg.GuidanceScale, Seed = 42 };
        LtxVideo2Pipeline.Ltx2Result result = pipeline.GenerateFromTokens(promptTokens, negTokens, req, numFrames,
            frameRate: 24.0, p => _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));

        _output.WriteLine($"[4/5] Writing {result.Frames.Length} frames" + (result.Audio is not null ? " + audio" : "") + "...");
        string outDir = Path.Combine(TestPaths.OutputDir, $"ltx2_video_{DateTime.Now:yyyyMMdd_HHmmss}");
        Directory.CreateDirectory(outDir);
        VideoFrame[] frames = new VideoFrame[result.Frames.Length];
        for (int i = 0; i < frames.Length; i++) frames[i] = new VideoFrame(i, result.Width, result.Height, result.Frames[i]);
        await new BmpSequenceEncoder().EncodeAsync(ToAsync(frames), outDir, fps: 24);

        _output.WriteLine($"[5/5] Wrote frames → {outDir}");
        Assert.Equal(numFrames, Directory.GetFiles(outDir, "frame_*.bmp").Length);

        // Prompt-cache roundtrip (LTX2_CACHE_ROUNDTRIP=1): gen 2 same tokens must skip the Gemma phase
        // (cache HIT), gen 3 different tokens must re-encode (miss path + staging). Frames dumped for both.
        if (Environment.GetEnvironmentVariable("LTX2_CACHE_ROUNDTRIP") == "1")
        {
            sw.Restart();
            LtxVideo2Pipeline.Ltx2Result hit = pipeline.GenerateFromTokens(promptTokens, negTokens, req, numFrames,
                frameRate: 24.0);
            _output.WriteLine($"[cache] gen 2 (same tokens, expect HIT): {sw.ElapsedMilliseconds}ms");
            await DumpFrames(hit, Path.Combine(TestPaths.OutputDir, $"ltx2_video_cachehit_{DateTime.Now:yyyyMMdd_HHmmss}"));

            int[] promptTokens2 = tokenizer.Encode("a red vintage car driving down a coastal road at sunset, engine rumbling");
            sw.Restart();
            LtxVideo2Pipeline.Ltx2Result miss = pipeline.GenerateFromTokens(promptTokens2, negTokens, req, numFrames,
                frameRate: 24.0);
            _output.WriteLine($"[cache] gen 3 (different tokens, expect MISS): {sw.ElapsedMilliseconds}ms");
            await DumpFrames(miss, Path.Combine(TestPaths.OutputDir, $"ltx2_video_cachemiss_{DateTime.Now:yyyyMMdd_HHmmss}"));

            // Gen 4 exercises the prefix top-up: the gen-3 MISS evicted the persistent prefix for the Gemma
            // encode and re-uploaded a squeezed count; this HIT must restore the pinned max (log evidence).
            sw.Restart();
            LtxVideo2Pipeline.Ltx2Result topUp = pipeline.GenerateFromTokens(promptTokens2, negTokens, req, numFrames,
                frameRate: 24.0);
            _output.WriteLine($"[cache] gen 4 (same tokens as gen 3, expect HIT + prefix top-up): {sw.ElapsedMilliseconds}ms");
            Assert.Equal(miss.Frames.Length, topUp.Frames.Length);
            for (int i = 0; i < miss.Frames.Length; i += 12)
                Assert.True(miss.Frames[i].AsSpan().SequenceEqual(topUp.Frames[i]), $"gen 4 frame {i} differs from gen 3 (same tokens+seed must be reproducible)");
        }
    }

    private static async Task DumpFrames(LtxVideo2Pipeline.Ltx2Result result, string outDir)
    {
        Directory.CreateDirectory(outDir);
        VideoFrame[] frames = new VideoFrame[result.Frames.Length];
        for (int i = 0; i < frames.Length; i++) frames[i] = new VideoFrame(i, result.Width, result.Height, result.Frames[i]);
        await new BmpSequenceEncoder().EncodeAsync(ToAsync(frames), outDir, fps: 24);
    }

    private static async IAsyncEnumerable<VideoFrame> ToAsync(VideoFrame[] frames)
    {
        foreach (VideoFrame f in frames) { yield return f; await Task.Yield(); }
    }

    /// <summary>Isolates the video-VAE grid artifact: decode ONE fixed random latent through the LTX-2 video VAE on
    /// BOTH the CUDA and CPU backends and dump the middle frame of each. If CUDA shows a 32-px grid but CPU does not,
    /// the bug is the CUDA host-glue/activation-cache stale-read (GPU-residency fix); if both match, the VAE logic is
    /// the issue (or the grid lives upstream in the DiT). Fast — no DiT, tiny latent.</summary>
    [Fact]
    public async Task LtxVideo2_VaeDecode_CudaVsCpu_Diagnostic()
    {
        string ckpt = TestPaths.LtxVideo2.SingleFile;
        if (!File.Exists(ckpt)) { _output.WriteLine($"SKIPPED: {ckpt} not found."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(LtxVideo2GenerationTests).Assembly.Location)!, "Ptx");
        (LtxVideo2CheckpointConverter.ConvertedWeights conv, SafeTensorsLoader loader) = LtxVideo2CheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _l = loader;
        if (conv.Vae.Count == 0)
        {
            _output.WriteLine("SKIPPED: transformer-only split file (no bundled VAE) — this test needs a bundled checkpoint.");
            return;
        }
        float[]? lm = ExtractStat(conv.Vae, "latents_mean"), ls = ExtractStat(conv.Vae, "latents_std");

        // Tiny smooth-ish random latent [1,128,2,6,8] → output 9×192×256. Same data for both backends.
        int C = 128, T = 2, H = 6, W = 8;
        TensorShape shape = new([1L, C, T, H, W]);
        long n = shape.ElementCount;
        float[] data = new float[n];
        Random rng = new(1234);
        for (long i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        unsafe void Fill(Tensor t) { fixed (float* dp = data) Buffer.MemoryCopy(dp, (void*)t.DataPointer, n * 4, n * 4); }

        string outDir = Path.Combine(TestPaths.OutputDir, "ltx2_vae_diag");
        Directory.CreateDirectory(outDir);
        foreach ((string tag, IBackend backend, bool dispose) in Backends(ptxDir))
        {
            using IBackend b = dispose ? backend : null!;
            LtxVideo2VaeDecoder vae = new(latentsMean: lm, latentsStd: ls);
            vae.LoadWeights(CastWeightsToF32(conv.Vae));
            Tensor latent = new(shape, DType.F32);
            Fill(latent);
            Tensor rgb = vae.Decode(backend, latent);
            latent.Dispose();
            int mid = (int)rgb.Shape[2] / 2;
            byte[] frame = VideoRgbFrames.ExtractFrame(rgb, mid);
            int w = (int)rgb.Shape[4], h = (int)rgb.Shape[3];
            double sum = 0, sum2 = 0; foreach (byte px in frame) { sum += px; sum2 += (double)px * px; }
            double mean = sum / frame.Length, std = Math.Sqrt(sum2 / frame.Length - mean * mean);
            await new BmpSequenceEncoder().EncodeAsync(One(new VideoFrame(0, w, h, frame)), Path.Combine(outDir, tag), fps: 1);
            _output.WriteLine($"[VAE-DIAG {tag}] {w}x{h} mean={mean:F1} std={std:F1} → {Path.Combine(outDir, tag)}");
            rgb.Dispose();
        }
    }

    private static IEnumerable<(string, IBackend, bool)> Backends(string ptxDir)
    {
        yield return ("cpu", new HartsyInference.Cpu.CpuBackend(), true);
        if (Directory.Exists(ptxDir))
            yield return ("cuda", new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir), true);
    }

    private static async IAsyncEnumerable<VideoFrame> One(VideoFrame f) { yield return f; await Task.Yield(); }

    private static unsafe float[]? ExtractStat(Dictionary<string, Tensor> vae, string key)
    {
        if (!vae.TryGetValue(key, out Tensor? t)) return null;
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        int n = (int)f32.ElementCount;
        float[] outv = new float[n];
        float* p = (float*)f32.DataPointer;
        for (int i = 0; i < n; i++) outv[i] = p[i];
        return outv;
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

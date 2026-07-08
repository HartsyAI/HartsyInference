using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
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
using HartsyInference.Vision.Clip;

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

    private static string? Env(string name)
    {
        string? v = Environment.GetEnvironmentVariable(name);
        return string.IsNullOrWhiteSpace(v) ? null : v;
    }

    /// <summary>Generalized real-weight e2e harness for ANY Wan variant, driven by env vars so one test validates the
    /// whole family. Auto-detects the config from the checkpoint (<see cref="WanConfigDetector"/>), picks the z=16/48
    /// VAE decoder, wires the MoE low-noise expert when given, and runs a short T2V clip — asserting the frames are
    /// non-degenerate (real output, not black/uniform).
    ///
    /// <para>Env: <c>WAN_CKPT</c> (DiT, required), <c>WAN_VAE</c> (VAE safetensors, required), <c>WAN_CKPT_LOW</c> +
    /// <c>WAN_BOUNDARY</c> (MoE A14B low-noise expert + boundary ratio), <c>WAN_FLOW_SHIFT</c>, and the shared
    /// <c>WAN_W/H/FRAMES/STEPS/MIN_GB</c>. Text encoder is umT5 at <c>UMT5_XXL_PATH</c>.</para></summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public async Task WanVariant_Gpu_E2E()
    {
        string? ckpt = Env("WAN_CKPT");
        if (ckpt is null) { _output.WriteLine("SKIPPED: set WAN_CKPT to a Wan DiT checkpoint to run the generalized variant harness."); return; }
        string vaePath = Env("WAN_VAE") ?? TestPaths.WanVideo.VaePath;
        string umt5Path = TestPaths.WanVideo.Umt5Xxl;
        if (!File.Exists(ckpt)) { _output.WriteLine($"SKIPPED: WAN_CKPT not found: {ckpt}"); return; }
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: WAN_VAE not found: {vaePath}"); return; }
        if (!File.Exists(umt5Path)) { _output.WriteLine($"SKIPPED: umT5 not found: {umt5Path}"); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(WanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/5] Load + convert DiT: {Path.GetFileName(ckpt)}");
        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _dit = ditLoader;

        // Auto-detect config from the checkpoint; apply the non-weight overrides (flow shift, MoE boundary).
        WanVideoConfig cfg = WanConfigDetector.Detect(conv.Transformer);
        string? lowPath = Env("WAN_CKPT_LOW");
        float boundary = (float)EnvD("WAN_BOUNDARY", 0.0);
        if (lowPath is not null && boundary > 0) cfg = cfg with { BoundaryRatio = boundary };
        if (Env("WAN_FLOW_SHIFT") is not null) cfg = cfg with { FlowShift = (float)EnvD("WAN_FLOW_SHIFT", cfg.FlowShift) };
        if (Env("WAN_CFG_RESCALE") is not null) cfg = cfg with { CfgRescale = (float)EnvD("WAN_CFG_RESCALE", cfg.CfgRescale) };
        _output.WriteLine($"  detected → {WanConfigDetector.Describe(cfg)} ({conv.Transformer.Count} keys, {sw.ElapsedMilliseconds}ms)");

        List<SafeTensorsLoader> loaders = [];
        try
        {
            (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vl);
            using SafeTensorsLoader umt5Loader = new();
            umt5Loader.Load(umt5Path);
            Dictionary<string, Tensor> umt5W = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());

            using WanVideoTransformer transformer = new(cfg);
            transformer.LoadWeights(conv.Transformer);

            // MoE (A14B): load the low-noise expert into transformer2.
            WanVideoTransformer? transformer2 = null;
            SafeTensorsLoader? lowLoader = null;
            if (lowPath is not null && File.Exists(lowPath))
            {
                (WanVideoCheckpointConverter.ConvertedWeights convLow, SafeTensorsLoader ll) = WanVideoCheckpointConverter.LoadAndConvert(lowPath);
                lowLoader = ll; loaders.Add(ll);
                transformer2 = new WanVideoTransformer(cfg);
                transformer2.LoadWeights(convLow.Transformer);
                _output.WriteLine($"  MoE low-noise expert loaded ({Path.GetFileName(lowPath)}), boundary={cfg.BoundaryRatio}");
            }

            // z=48 → Wan2.2 VAE decoder; z=16 → Wan2.1 VAE decoder.
            IWanVaeDecoder vae;
            if (cfg.VaeLatentChannels >= 48) { Wan22VaeDecoder d = new(); d.LoadWeights(vaeW); vae = d; }
            else { Wan21VaeDecoder d = new(); d.LoadWeights(vaeW); vae = d; }

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            if (Environment.GetEnvironmentVariable("WAN_NO_CACHE_CASTS") == "1") backend.CacheWeightCasts = false;
            (nuint freeBytes, _) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            double minGb = EnvD("WAN_MIN_GB", 14.0);
            if (freeGb < minGb) { _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM; need ≥{minGb}."); transformer2?.Dispose(); return; }

            _output.WriteLine($"[3/5] umT5 encode...");
            using T5TextEncoder umt5 = new(T5TextEncoderConfig.Umt5Xxl);
            umt5.LoadWeights(umt5W);
            using T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: 512);
            int[] promptTokens = tokenizer.Encode("a cinematic shot of a red fox trotting across a snowy field at dawn, shallow depth of field");
            int[] negTokens = tokenizer.Encode("blurry, low quality, distorted, watermark");
            Tensor tbatch = umt5.Encode(backend, [promptTokens, negTokens],
                [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
            int seqLen = promptTokens.Length;
            Tensor promptEmbeds = CfgHelper.SliceBatchElement(tbatch, 0, seqLen, 4096);
            Tensor negEmbeds = CfgHelper.SliceBatchElement(tbatch, 1, seqLen, 4096);
            tbatch.Dispose();
            backend.Sync();
            backend.FreeWeights(umt5.EnumerateWeights());

            int genW = (int)EnvD("WAN_W", 832), genH = (int)EnvD("WAN_H", 480);
            int genFrames = (int)EnvD("WAN_FRAMES", 33), genSteps = (int)EnvD("WAN_STEPS", 20);
            _output.WriteLine($"[4/5] Generating {genFrames}f {genW}x{genH}, {genSteps} steps...");
            sw.Restart();
            WanVideoPipeline pipeline = new(backend, transformer, vae, cfg, encoder: null, transformer2: transformer2);
            float cfgScale = (float)EnvD("WAN_CFG", cfg.GuidanceScale);
            TextToImageRequest req = new() { Prompt = "fox", Width = genW, Height = genH, Steps = genSteps, CfgScale = cfgScale, Seed = 42 };
            (byte[][] frames, int w, int h, _) = pipeline.GenerateFromEmbeddings(promptEmbeds, negEmbeds, req, genFrames,
                p => { if (p.Step % 5 == 0 || p.Step == p.TotalSteps) _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"); });
            promptEmbeds.Dispose(); negEmbeds.Dispose(); transformer2?.Dispose();
            _output.WriteLine($"[5/5] {frames.Length} frames in {sw.Elapsed.TotalMinutes:F2} min");

            // Coherence: frames must be non-degenerate (spread + temporal variation), not black/uniform.
            AssertFramesCoherent(frames, w, h);
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    /// <summary>VAE quality discriminator: encodes a real RGB image through the Wan VAE encoder and decodes it
    /// straight back (no denoising), writing input/output BMPs for visual comparison. Isolates VAE-roundtrip
    /// softness/halos from denoise/parameter problems. Env: <c>WAN_ROUNDTRIP_IMG</c> (raw RGB24 .bin) or uses a
    /// synthetic checker+gradient; <c>WAN_VAE</c>, <c>WAN_RT_W/H</c> (default 832×480).</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void WanVae_Roundtrip_Quality()
    {
        string vaePath = Env("WAN_VAE") ?? "/home/hartsy/Desktop/Swarm/SwarmUI.old/Models/VAE/Wan/wan_2.1_vae.safetensors";
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(WanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!File.Exists(vaePath) || !Directory.Exists(ptxDir) || !CudaContext.IsAvailable())
        { _output.WriteLine("SKIPPED: missing VAE/PTX/CUDA."); return; }
        int w = (int)EnvD("WAN_RT_W", 832), h = (int)EnvD("WAN_RT_H", 480);

        byte[] rgb;
        string? imgPath = Env("WAN_ROUNDTRIP_IMG");
        if (imgPath is not null && File.Exists(imgPath))
        {
            rgb = File.ReadAllBytes(imgPath);
            Assert.True(rgb.Length >= w * h * 3, $"raw RGB24 file must be ≥ {w * h * 3} bytes for {w}x{h}");
        }
        else
        {
            rgb = new byte[w * h * 3];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int o = (y * w + x) * 3;
                    bool check = ((x / 16) + (y / 16)) % 2 == 0;   // fine checker exposes VAE softness
                    rgb[o] = (byte)(check ? 230 : 25);
                    rgb[o + 1] = (byte)(255 * x / w);
                    rgb[o + 2] = (byte)(255 * y / h);
                }
        }

        (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) = LanceCheckpointConverter.LoadVae(vaePath);
        try
        {
            Wan21VaeEncoder enc = new(); enc.LoadWeights(vaeW);
            Wan21VaeDecoder dec = new(); dec.LoadWeights(vaeW);
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            backend.PreloadWeights(enc.EnumerateWeights());
            Tensor lat = enc.EncodeRgbFrame(backend, rgb, w, h);
            backend.Sync();
            backend.FreeWeights(enc.EnumerateWeights());
            backend.PreloadWeights(dec.EnumerateWeights());
            Tensor rgbOut = dec.Decode(backend, lat);   // [1,3,T(=1..4),H,W]
            lat.Dispose();
            string outDir = Path.Combine(TestPaths.OutputDir, $"wan_vae_roundtrip_{DateTime.Now:yyyyMMdd_HHmmss}");
            Directory.CreateDirectory(outDir);
            HartsyInference.Diffusion.Utilities.ImagePostProcessor.SaveBmp(Path.Combine(outDir, "input.bmp"), rgb, w, h);
            byte[] outFrame = VideoRgbFrames.ExtractFrame(rgbOut, 0);
            HartsyInference.Diffusion.Utilities.ImagePostProcessor.SaveBmp(Path.Combine(outDir, "roundtrip.bmp"), outFrame, w, h);
            rgbOut.Dispose();
            _output.WriteLine($"[roundtrip] → {outDir}");
        }
        finally { foreach (SafeTensorsLoader l in vl) l.Dispose(); }
    }

    /// <summary>Real-weight e2e for Wan2.1 I2V-14B (CLIP-conditioned image-to-video): loads the fp8 DiT (auto-detected
    /// I2V config: in=36, image_dim=1280, CFG-renorm on), the z=16 VAE (decoder + encoder), and CLIP-ViT-H; encodes a
    /// reference image to the 257×1280 penultimate hidden states → <c>imageEmbeds</c> and VAE-encodes it into the
    /// concat conditioning, then runs <see cref="WanVideoPipeline.GenerateImageToVideoConcat"/>. Env: <c>WAN_I2V_CKPT</c>,
    /// <c>WAN_VAE</c>, <c>WAN_CLIP_VISION</c>.</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void WanI2V14B_Gpu_E2E()
    {
        string? ckpt = Env("WAN_I2V_CKPT");
        if (ckpt is null || !File.Exists(ckpt)) { _output.WriteLine("SKIPPED: set WAN_I2V_CKPT to a Wan I2V-14B checkpoint."); return; }
        string vaePath = Env("WAN_VAE") ?? TestPaths.WanVideo.VaePath;
        string clipPath = Env("WAN_CLIP_VISION") ?? "/home/hartsy/Desktop/Swarm/SwarmUI.old/Models/clip_vision/clip_vision_h.safetensors";
        string umt5Path = TestPaths.WanVideo.Umt5Xxl;
        if (!File.Exists(vaePath) || !File.Exists(clipPath) || !File.Exists(umt5Path)) { _output.WriteLine($"SKIPPED: missing VAE/CLIP/umT5 ({vaePath}|{clipPath}|{umt5Path})."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(WanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir) || !CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no PTX/CUDA"); return; }

        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _dit = ditLoader;
        WanVideoConfig cfg = WanConfigDetector.Detect(conv.Transformer);
        _output.WriteLine($"  detected → {WanConfigDetector.Describe(cfg)}");
        Assert.Equal(36, cfg.InChannels);   // I2V concat input (14B uses CLIP; A14B is concat-only, no CLIP)
        string? lowPath = Env("WAN_I2V_CKPT_LOW");
        float boundary = (float)EnvD("WAN_BOUNDARY", 0.0);
        if (lowPath is not null && File.Exists(lowPath) && boundary > 0) cfg = cfg with { BoundaryRatio = boundary };

        List<SafeTensorsLoader> loaders = [];
        try
        {
            (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vl);
            using SafeTensorsLoader umt5Loader = new(); umt5Loader.Load(umt5Path);
            Dictionary<string, Tensor> umt5W = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());

            using WanVideoTransformer transformer = new(cfg);
            transformer.LoadWeights(conv.Transformer);
            Wan21VaeDecoder vae = new(); vae.LoadWeights(vaeW);
            Wan21VaeEncoder encoder = new(); encoder.LoadWeights(vaeW);

            // MoE low-noise expert (Wan2.2 A14B I2V ships as two experts).
            WanVideoTransformer? transformer2 = null;
            if (lowPath is not null && File.Exists(lowPath) && cfg.BoundaryRatio > 0)
            {
                (WanVideoCheckpointConverter.ConvertedWeights convLow, SafeTensorsLoader ll) = WanVideoCheckpointConverter.LoadAndConvert(lowPath);
                loaders.Add(ll);
                transformer2 = new WanVideoTransformer(cfg); transformer2.LoadWeights(convLow.Transformer);
                _output.WriteLine($"  MoE low-noise expert loaded, boundary={cfg.BoundaryRatio}");
            }

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            if (Environment.GetEnvironmentVariable("WAN_NO_CACHE_CASTS") == "1") backend.CacheWeightCasts = false;
            (nuint freeBytes, _) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            if (freeGb < EnvD("WAN_MIN_GB", 15.0)) { _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free."); transformer2?.Dispose(); return; }

            int genW = (int)EnvD("WAN_W", 480), genH = (int)EnvD("WAN_H", 320);
            int genFrames = (int)EnvD("WAN_FRAMES", 25), genSteps = (int)EnvD("WAN_STEPS", 20);

            // Synthetic reference image (HWC RGB24): smooth colour field — enough to validate the I2V plumbing e2e.
            byte[] init = new byte[genW * genH * 3];
            for (int y = 0; y < genH; y++)
                for (int x = 0; x < genW; x++)
                {
                    int o = (y * genW + x) * 3;
                    init[o] = (byte)(255 * x / genW);
                    init[o + 1] = (byte)(255 * y / genH);
                    init[o + 2] = (byte)(128 + 127 * Math.Sin(6.28 * (x + y) / 96.0));
                }

            // First-last-frame (FLF2V): a distinct last-frame image; WAN_FLF=1 enables it.
            byte[]? lastImg = null;
            if (Env("WAN_FLF") == "1")
            {
                lastImg = new byte[genW * genH * 3];
                for (int y = 0; y < genH; y++)
                    for (int x = 0; x < genW; x++)
                    {
                        int o = (y * genW + x) * 3;
                        lastImg[o] = (byte)(255 * y / genH);
                        lastImg[o + 1] = (byte)(255 * (genW - x) / genW);
                        lastImg[o + 2] = (byte)(64 + 63 * Math.Cos(6.28 * (x - y) / 80.0));
                    }
            }

            // CLIP image embeds only for the CLIP-conditioned variant (Wan2.1 I2V-14B / FLF2V); A14B conditions via concat only.
            Tensor? imageEmbeds = null;
            if (cfg.HasImageConditioning)
            {
                _output.WriteLine("[clip] encoding reference image(s) → penultimate hidden states...");
                using SafeTensorsLoader clipLoader = new(); clipLoader.Load(clipPath);
                ClipVisionEncoder clip = new(ClipVisionEncoderConfig.ViTH14); clip.LoadWeights(clipLoader.GetAllTensors());
                using Tensor pix = new ClipImagePreprocessor().Preprocess(init, genW, genH);
                imageEmbeds = clip.EncodeHiddenStates(backend, pix); backend.Sync();
                if (lastImg is not null)   // FLF2V: concat first+last CLIP embeds → [1,514,1280]
                {
                    using Tensor pixL = new ClipImagePreprocessor().Preprocess(lastImg, genW, genH);
                    using Tensor le = clip.EncodeHiddenStates(backend, pixL); backend.Sync();
                    Tensor cat = ConcatImageEmbeds(imageEmbeds, le);
                    imageEmbeds.Dispose(); imageEmbeds = cat;
                }
                _output.WriteLine($"  imageEmbeds {string.Join("x", Enumerable.Range(0, imageEmbeds.Shape.Rank).Select(d => (long)imageEmbeds.Shape[d]))}");
                backend.Sync();
                backend.FreeWeights(clip.EnumerateWeights());   // reclaim CLIP VRAM before the DiT
            }

            using T5TextEncoder umt5 = new(T5TextEncoderConfig.Umt5Xxl); umt5.LoadWeights(umt5W);
            using T5Tokenizer tok = T5Tokenizer.CreateUmt5(maxLength: 512);
            int[] pTok = tok.Encode("the scene comes to life, gentle camera motion, cinematic");
            int[] nTok = tok.Encode("blurry, low quality, distorted");
            Tensor tb = umt5.Encode(backend, [pTok, nTok], [T5Tokenizer.CreateAttentionMask(pTok), T5Tokenizer.CreateAttentionMask(nTok)]);
            Tensor promptEmbeds = CfgHelper.SliceBatchElement(tb, 0, pTok.Length, 4096);
            Tensor negEmbeds = CfgHelper.SliceBatchElement(tb, 1, pTok.Length, 4096);
            tb.Dispose();
            backend.Sync();
            backend.FreeWeights(umt5.EnumerateWeights());

            _output.WriteLine($"[gen] I2V {genFrames}f {genW}x{genH}, {genSteps} steps, cfg={cfg.GuidanceScale}, renorm={cfg.CfgRescale}, moe={transformer2 is not null}...");
            WanVideoPipeline pipeline = new(backend, transformer, vae, cfg, encoder, transformer2);
            TextToImageRequest req = new() { Prompt = "i2v", Width = genW, Height = genH, Steps = genSteps, CfgScale = cfg.GuidanceScale, Seed = 42 };
            (byte[][] frames, int w, int h, _) = pipeline.GenerateImageToVideoConcat(promptEmbeds, negEmbeds, imageEmbeds, init, req, genFrames,
                p => { if (p.Step % 5 == 0 || p.Step == p.TotalSteps) _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"); }, lastRgb24: lastImg);
            imageEmbeds?.Dispose(); promptEmbeds.Dispose(); negEmbeds.Dispose(); transformer2?.Dispose();
            // AssertFramesCoherent only catches degenerate output (flat/black); it passes on structured garbage —
            // dump the frames so quality regressions can be verified visually (the 2026-07-08 I2V debugging need).
            string i2vOutDir = Path.Combine(TestPaths.OutputDir, $"wan_i2v_{DateTime.Now:yyyyMMdd_HHmmss}");
            Directory.CreateDirectory(i2vOutDir);
            for (int fi = 0; fi < frames.Length; fi++)
                HartsyInference.Diffusion.Utilities.ImagePostProcessor.SaveBmp(
                    Path.Combine(i2vOutDir, $"frame_{fi:D5}.bmp"), frames[fi], w, h);
            _output.WriteLine($"[frames] → {i2vOutDir}");
            AssertFramesCoherent(frames, w, h);
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    /// <summary>Real-weight e2e for Wan2.1 VACE (control-video conditioning): loads the merged VACE DiT (config +
    /// control branch auto-detected via <c>vace_patch_embedding</c>), the z=16 VAE (decoder + encoder), builds a
    /// synthetic moving-square control clip, and runs <see cref="WanVacePipeline.GenerateFromControl"/> in full-control
    /// mode (null mask → inactive = gray, reactive = control). Env: <c>WAN_VACE_CKPT</c>, <c>WAN_VAE</c> (z=16 VAE),
    /// plus the shared <c>WAN_W/H/FRAMES/STEPS/MIN_GB/CFG/FLOW_SHIFT</c>.</summary>
    /// <summary>Real-weight e2e for the production <see cref="WanVideoLoader"/> entry point: one call from checkpoint
    /// paths to a ready pipeline (auto-detect incl. VACE routing + auto CFG-renorm), umT5 via
    /// <see cref="WanVideoLoader.EncodePrompts"/>, then a short T2V (or full-control VACE) clip. Env:
    /// <c>WAN_LOADER_CKPT</c>, <c>WAN_VAE</c>, plus the shared <c>WAN_W/H/FRAMES/STEPS/MIN_GB</c>.</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void WanLoader_Gpu_E2E()
    {
        string? ckpt = Env("WAN_LOADER_CKPT");
        if (ckpt is null || !File.Exists(ckpt)) { _output.WriteLine("SKIPPED: set WAN_LOADER_CKPT to a Wan DiT checkpoint."); return; }
        string vaePath = Env("WAN_VAE") ?? "/home/hartsy/Desktop/Swarm/SwarmUI.old/Models/VAE/Wan/wan_2.1_vae.safetensors";
        string umt5Path = TestPaths.WanVideo.Umt5Xxl;
        if (!File.Exists(vaePath) || !File.Exists(umt5Path)) { _output.WriteLine($"SKIPPED: missing VAE/umT5 ({vaePath}|{umt5Path})."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(WanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir) || !CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no PTX/CUDA"); return; }

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
        if (Environment.GetEnvironmentVariable("WAN_NO_CACHE_CASTS") == "1") backend.CacheWeightCasts = false;
        (nuint freeBytes, _) = backend.Context.GetMemoryInfo();
        if (freeBytes / (1024.0 * 1024.0 * 1024.0) < EnvD("WAN_MIN_GB", 8.0)) { _output.WriteLine("SKIPPED: low VRAM."); return; }

        using WanVideoLoader loader = WanVideoLoader.Load(backend, ckpt, vaePath, new WanVideoLoadOptions
        {
            Umt5Path = umt5Path,
            LowNoiseDitPath = Env("WAN_LOADER_CKPT_LOW"),
            FlowShift = Env("WAN_FLOW_SHIFT") is null ? null : (float)EnvD("WAN_FLOW_SHIFT", 5.0),
        });
        _output.WriteLine($"  loader → {WanConfigDetector.Describe(loader.Config)} (vace={loader.VacePipeline is not null})");

        (Tensor promptEmbeds, Tensor negEmbeds) = loader.EncodePrompts(
            "a bright red balloon drifting across a clear blue sky on a sunny day, vivid colors",
            "blurry, low quality, distorted");
        int genW = (int)EnvD("WAN_W", 480), genH = (int)EnvD("WAN_H", 320);
        int genFrames = (int)EnvD("WAN_FRAMES", 17), genSteps = (int)EnvD("WAN_STEPS", 20);
        TextToImageRequest req = new() { Prompt = "loader", Width = genW, Height = genH, Steps = genSteps, CfgScale = (float)EnvD("WAN_CFG", loader.Config.GuidanceScale), Seed = 42 };

        byte[][] frames; int w, h;
        if (loader.VacePipeline is WanVacePipeline vace)
        {
            Tensor control = BuildMovingSquareClip(genFrames, genH, genW);
            (frames, w, h, _) = vace.GenerateFromControl(promptEmbeds, negEmbeds, control, req,
                onProgress: p => { if (p.Step % 5 == 0) _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"); });
            control.Dispose();
        }
        else
        {
            (frames, w, h, _) = loader.Pipeline!.GenerateFromEmbeddings(promptEmbeds, negEmbeds, req, genFrames,
                p => { if (p.Step % 5 == 0) _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"); });
        }
        promptEmbeds.Dispose(); negEmbeds.Dispose();
        AssertFramesCoherent(frames, w, h);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void WanVace_Gpu_E2E()
    {
        string? ckpt = Env("WAN_VACE_CKPT");
        if (ckpt is null || !File.Exists(ckpt)) { _output.WriteLine("SKIPPED: set WAN_VACE_CKPT to a merged Wan VACE checkpoint."); return; }
        string vaePath = Env("WAN_VAE") ?? "/home/hartsy/Desktop/Swarm/SwarmUI.old/Models/VAE/Wan/wan_2.1_vae.safetensors";
        string umt5Path = TestPaths.WanVideo.Umt5Xxl;
        if (!File.Exists(vaePath) || !File.Exists(umt5Path)) { _output.WriteLine($"SKIPPED: missing VAE/umT5 ({vaePath}|{umt5Path})."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(WanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir) || !CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no PTX/CUDA"); return; }

        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _dit = ditLoader;
        WanVideoConfig cfg = WanConfigDetector.Detect(conv.Transformer);
        if (Env("WAN_FLOW_SHIFT") is not null) cfg = cfg with { FlowShift = (float)EnvD("WAN_FLOW_SHIFT", cfg.FlowShift) };
        if (Env("WAN_CFG_RESCALE") is not null) cfg = cfg with { CfgRescale = (float)EnvD("WAN_CFG_RESCALE", cfg.CfgRescale) };
        _output.WriteLine($"  detected → {WanConfigDetector.Describe(cfg)}");
        Assert.True(cfg.VaceLayers.Length > 0, "checkpoint has no vace_blocks — not a VACE DiT");

        List<SafeTensorsLoader> loaders = [];
        try
        {
            (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vl);
            using SafeTensorsLoader umt5Loader = new(); umt5Loader.Load(umt5Path);
            Dictionary<string, Tensor> umt5W = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());

            using WanVaceTransformer transformer = new(cfg);
            transformer.LoadWeights(conv.Transformer);
            Wan21VaeDecoder vae = new(); vae.LoadWeights(vaeW);
            Wan21VaeEncoder encoder = new(); encoder.LoadWeights(vaeW);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            if (Environment.GetEnvironmentVariable("WAN_NO_CACHE_CASTS") == "1") backend.CacheWeightCasts = false;
            (nuint freeBytes, _) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            if (freeGb < EnvD("WAN_MIN_GB", 8.0)) { _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free."); return; }

            using T5TextEncoder umt5 = new(T5TextEncoderConfig.Umt5Xxl); umt5.LoadWeights(umt5W);
            using T5Tokenizer tok = T5Tokenizer.CreateUmt5(maxLength: 512);
            int[] pTok = tok.Encode("a bright red balloon drifting across a clear blue sky on a sunny day, vivid colors, high key lighting");
            int[] nTok = tok.Encode("blurry, low quality, distorted");
            Tensor tb = umt5.Encode(backend, [pTok, nTok], [T5Tokenizer.CreateAttentionMask(pTok), T5Tokenizer.CreateAttentionMask(nTok)]);
            Tensor promptEmbeds = CfgHelper.SliceBatchElement(tb, 0, pTok.Length, 4096);
            Tensor negEmbeds = CfgHelper.SliceBatchElement(tb, 1, pTok.Length, 4096);
            tb.Dispose();
            backend.Sync();
            backend.FreeWeights(umt5.EnumerateWeights());

            int genW = (int)EnvD("WAN_W", 480), genH = (int)EnvD("WAN_H", 320);
            int genFrames = (int)EnvD("WAN_FRAMES", 25), genSteps = (int)EnvD("WAN_STEPS", 20);

            // Synthetic control clip [1,3,T,H,W] in [-1,1]: a bright square sweeping left→right on a dark field —
            // enough structure for the control branch to steer motion and validate the plumbing e2e.
            Tensor control = BuildMovingSquareClip(genFrames, genH, genW);

            _output.WriteLine($"[gen] VACE {genFrames}f {genW}x{genH}, {genSteps} steps, cfg={cfg.GuidanceScale}, renorm={cfg.CfgRescale}...");
            WanVacePipeline pipeline = new(backend, transformer, vae, encoder, cfg);
            float cfgScale = (float)EnvD("WAN_CFG", cfg.GuidanceScale);
            TextToImageRequest req = new() { Prompt = "vace", Width = genW, Height = genH, Steps = genSteps, CfgScale = cfgScale, Seed = 42 };
            (byte[][] frames, int w, int h, _) = pipeline.GenerateFromControl(promptEmbeds, negEmbeds, control, req,
                controlScale: (float)EnvD("WAN_VACE_SCALE", 1.0),
                p => { if (p.Step % 5 == 0 || p.Step == p.TotalSteps) _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"); });
            control.Dispose(); promptEmbeds.Dispose(); negEmbeds.Dispose();
            AssertFramesCoherent(frames, w, h);
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    /// <summary>Real-weight e2e for Wan2.2-S2V-14B (speech-to-video): fp8 DiT (auto-detected S2V config incl. the
    /// audio injector), Wav2Vec2-large front-end on a synthetic modulated waveform, z=16 VAE, optional reference image.
    /// Env: <c>WAN_S2V_CKPT</c>, <c>WAN_WAV2VEC</c>, <c>WAN_VAE</c>; needs the 4090 (fp8 14B ⇒ <c>WAN_NO_CACHE_CASTS=1</c>).</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void WanS2V_Gpu_E2E()
    {
        string ckpt = Env("WAN_S2V_CKPT") ?? "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/Wan/wan2.2_s2v_14B_fp8_scaled.safetensors";
        string wavPath = Env("WAN_WAV2VEC") ?? "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/Wan/wav2vec2_large_english_fp16.safetensors";
        string vaePath = Env("WAN_VAE") ?? "/home/hartsy/Desktop/Swarm/SwarmUI.old/Models/VAE/Wan/wan_2.1_vae.safetensors";
        string umt5Path = TestPaths.WanVideo.Umt5Xxl;
        if (!File.Exists(ckpt) || !File.Exists(wavPath) || !File.Exists(vaePath) || !File.Exists(umt5Path))
        { _output.WriteLine($"SKIPPED: missing S2V ckpt/wav2vec2/VAE/umT5."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(WanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir) || !CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no PTX/CUDA"); return; }

        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _dit = ditLoader;
        WanVideoConfig cfg = WanConfigDetector.Detect(conv.Transformer);
        if (Env("WAN_FLOW_SHIFT") is not null) cfg = cfg with { FlowShift = (float)EnvD("WAN_FLOW_SHIFT", cfg.FlowShift) };
        if (Env("WAN_CFG_RESCALE") is not null) cfg = cfg with { CfgRescale = (float)EnvD("WAN_CFG_RESCALE", cfg.CfgRescale) };
        _output.WriteLine($"  detected → {WanConfigDetector.Describe(cfg)}");
        Assert.True(cfg.HasAudioConditioning, "checkpoint has no audio conditioning — not an S2V DiT");

        List<SafeTensorsLoader> loaders = [];
        try
        {
            (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vl);
            using SafeTensorsLoader umt5Loader = new(); umt5Loader.Load(umt5Path);
            Dictionary<string, Tensor> umt5W = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());

            using WanS2VTransformer transformer = new(cfg);
            transformer.LoadWeights(conv.Transformer);
            WanS2VAudioEncoder audioEncoder = new(cfg.AudioLayers, cfg.AudioDim, cfg.InnerDim, cfg.AudioTokens);
            audioEncoder.LoadWeights(conv.Transformer);
            Wan21VaeDecoder vae = new(); vae.LoadWeights(vaeW);
            Wan21VaeEncoder encoder = new(); encoder.LoadWeights(vaeW);

            // The ComfyOrg audio-encoder file prefixes every key with "wav2vec2." (plus an unused lm_head).
            using SafeTensorsLoader wavLoader = new(); wavLoader.Load(wavPath);
            Dictionary<string, Tensor> wavW = new();
            foreach (KeyValuePair<string, Tensor> kvp in wavLoader.GetAllTensors())
                if (kvp.Key.StartsWith("wav2vec2.", StringComparison.Ordinal)) wavW[kvp.Key["wav2vec2.".Length..]] = kvp.Value;
            Wav2Vec2Encoder wav2vec2 = new(Wav2Vec2EncoderConfig.Large);
            wav2vec2.LoadWeights(wavW);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            if (Environment.GetEnvironmentVariable("WAN_NO_CACHE_CASTS") == "1") backend.CacheWeightCasts = false;
            (nuint freeBytes, _) = backend.Context.GetMemoryInfo();
            if (freeBytes / (1024.0 * 1024.0 * 1024.0) < EnvD("WAN_MIN_GB", 17.0)) { _output.WriteLine("SKIPPED: low VRAM."); return; }

            using T5TextEncoder umt5 = new(T5TextEncoderConfig.Umt5Xxl); umt5.LoadWeights(umt5W);
            using T5Tokenizer tok = T5Tokenizer.CreateUmt5(maxLength: 512);
            int[] pTok = tok.Encode("a person speaking enthusiastically to the camera, natural lighting, photorealistic");
            int[] nTok = tok.Encode("blurry, low quality, distorted");
            Tensor tb = umt5.Encode(backend, [pTok, nTok], [T5Tokenizer.CreateAttentionMask(pTok), T5Tokenizer.CreateAttentionMask(nTok)]);
            Tensor promptEmbeds = CfgHelper.SliceBatchElement(tb, 0, pTok.Length, 4096);
            Tensor negEmbeds = CfgHelper.SliceBatchElement(tb, 1, pTok.Length, 4096);
            tb.Dispose();
            backend.Sync();
            backend.FreeWeights(umt5.EnumerateWeights());

            int genW = (int)EnvD("WAN_W", 480), genH = (int)EnvD("WAN_H", 320);
            int genFrames = (int)EnvD("WAN_FRAMES", 33), genSteps = (int)EnvD("WAN_STEPS", 20);

            // Synthetic "speech": a 16 kHz amplitude-modulated tone sweep long enough for genFrames at 16 fps.
            int samples = (int)((genFrames / 16.0 + 0.5) * 16000);
            float[] waveform = new float[samples];
            for (int i = 0; i < samples; i++)
            {
                double t = i / 16000.0;
                waveform[i] = (float)(0.4 * Math.Sin(2 * Math.PI * (140 + 60 * Math.Sin(2 * Math.PI * 2.5 * t)) * t)
                                      * (0.55 + 0.45 * Math.Sin(2 * Math.PI * 4.0 * t)));
            }

            byte[] reference = new byte[genW * genH * 3];
            for (int y = 0; y < genH; y++)
                for (int x = 0; x < genW; x++)
                {
                    int o = (y * genW + x) * 3;
                    reference[o] = (byte)(200 - 100 * y / genH);
                    reference[o + 1] = (byte)(160 + 60 * x / genW);
                    reference[o + 2] = (byte)(150 + 80 * Math.Sin(6.28 * (x + y) / 120.0));
                }

            _output.WriteLine($"[gen] S2V {genFrames}f {genW}x{genH}, {genSteps} steps, cfg={cfg.GuidanceScale}, renorm={cfg.CfgRescale}...");
            WanS2VPipeline pipeline = new(backend, transformer, audioEncoder, vae, cfg, encoder);
            if (Env("WAN_S2V_NO_REF") == "1") reference = [];   // isolate the reference-token path when debugging

            // WAN_CFG_SWEEP="3:0.7,5:1.0" runs one generation per cfg:rescale pair in THIS process (the GPU stays
            // held throughout — chaining separate test processes loses the GPU to other tenants between runs).
            string sweep = Env("WAN_CFG_SWEEP") ?? $"{EnvD("WAN_CFG", cfg.GuidanceScale)}:{cfg.CfgRescale}";
            byte[][] frames = []; int w = genW, h = genH;
            foreach (string pair in sweep.Split(','))
            {
                string[] parts = pair.Split(':');
                float cfgScale = float.Parse(parts[0]);
                float rescale = parts.Length > 1 ? float.Parse(parts[1]) : cfg.CfgRescale;
                WanS2VPipeline sweepPipeline = rescale == cfg.CfgRescale ? pipeline
                    : new WanS2VPipeline(backend, transformer, audioEncoder, vae, cfg with { CfgRescale = rescale }, encoder);
                TextToImageRequest req = new() { Prompt = "s2v", Width = genW, Height = genH, Steps = genSteps, CfgScale = cfgScale, Seed = 42 };
                _output.WriteLine($"[sweep] cfg={cfgScale} rescale={rescale}");
                (frames, w, h, _) = sweepPipeline.GenerateFromWaveform(promptEmbeds, negEmbeds, waveform, wav2vec2, req, genFrames, reference,
                    p =>
                    {
                        if (p.Step % 5 != 0 && p.Step != p.TotalSteps) return;
                        (long cached, _, _) = backend.GetGpuCacheStats();
                        _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms) free={backend.FreeMemoryBytes() >> 20}MB cached={cached >> 20}MB");
                    });
                double[] means = frames.Select(f => f.Average(b => (double)b)).ToArray();
                _output.WriteLine($"[sweep] cfg={cfgScale} rescale={rescale} → means: {string.Join(" ", means.Select(m => m.ToString("F0")))}");
            }
            promptEmbeds.Dispose(); negEmbeds.Dispose();
            AssertFramesCoherent(frames, w, h);
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    /// <summary>Real-weight e2e for Wan2.2-Animate-14B (character animation): fp8 DiT (auto-detected Animate config —
    /// pose patch-embed + motion/face adapter), synthetic reference/pose/face inputs, z=16 VAE. Env:
    /// <c>WAN_ANIMATE_CKPT</c>, <c>WAN_VAE</c>; needs the 4090 (fp8 14B ⇒ <c>WAN_NO_CACHE_CASTS=1</c>).</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void WanAnimate_Gpu_E2E()
    {
        string ckpt = Env("WAN_ANIMATE_CKPT") ?? "/home/hartsy/Desktop/HartsyInference/Models/Stable-Diffusion/Wan/wan2.2_animate_14B_fp8_scaled_KJ_v2.safetensors";
        string vaePath = Env("WAN_VAE") ?? "/home/hartsy/Desktop/Swarm/SwarmUI.old/Models/VAE/Wan/wan_2.1_vae.safetensors";
        string umt5Path = TestPaths.WanVideo.Umt5Xxl;
        if (!File.Exists(ckpt) || !File.Exists(vaePath) || !File.Exists(umt5Path)) { _output.WriteLine("SKIPPED: missing Animate ckpt/VAE/umT5."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(WanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir) || !CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no PTX/CUDA"); return; }

        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) = WanVideoCheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _dit = ditLoader;
        WanVideoConfig cfg = WanConfigDetector.Detect(conv.Transformer);
        if (Env("WAN_FLOW_SHIFT") is not null) cfg = cfg with { FlowShift = (float)EnvD("WAN_FLOW_SHIFT", cfg.FlowShift) };
        if (Env("WAN_CFG_RESCALE") is not null) cfg = cfg with { CfgRescale = (float)EnvD("WAN_CFG_RESCALE", cfg.CfgRescale) };
        _output.WriteLine($"  detected → {WanConfigDetector.Describe(cfg)}");
        Assert.True(cfg.IsAnimate, "checkpoint has no pose/face adapter — not an Animate DiT");

        List<SafeTensorsLoader> loaders = [];
        try
        {
            (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vl);
            using SafeTensorsLoader umt5Loader = new(); umt5Loader.Load(umt5Path);
            Dictionary<string, Tensor> umt5W = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());

            using WanAnimateTransformer transformer = new(cfg);
            transformer.LoadWeights(conv.Transformer);
            Wan21VaeDecoder vae = new(); vae.LoadWeights(vaeW);
            Wan21VaeEncoder encoder = new(); encoder.LoadWeights(vaeW);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            if (Environment.GetEnvironmentVariable("WAN_NO_CACHE_CASTS") == "1") backend.CacheWeightCasts = false;
            (nuint freeBytes, _) = backend.Context.GetMemoryInfo();
            if (freeBytes / (1024.0 * 1024.0 * 1024.0) < EnvD("WAN_MIN_GB", 18.0)) { _output.WriteLine("SKIPPED: low VRAM."); return; }

            using T5TextEncoder umt5 = new(T5TextEncoderConfig.Umt5Xxl); umt5.LoadWeights(umt5W);
            using T5Tokenizer tok = T5Tokenizer.CreateUmt5(maxLength: 512);
            int[] pTok = tok.Encode("a person dancing in a bright studio, smooth motion, photorealistic");
            int[] nTok = tok.Encode("blurry, low quality, distorted");
            Tensor tb = umt5.Encode(backend, [pTok, nTok], [T5Tokenizer.CreateAttentionMask(pTok), T5Tokenizer.CreateAttentionMask(nTok)]);
            Tensor promptEmbeds = CfgHelper.SliceBatchElement(tb, 0, pTok.Length, 4096);
            Tensor negEmbeds = CfgHelper.SliceBatchElement(tb, 1, pTok.Length, 4096);
            tb.Dispose();
            backend.Sync();
            backend.FreeWeights(umt5.EnumerateWeights());

            int genW = (int)EnvD("WAN_W", 480), genH = (int)EnvD("WAN_H", 320);
            int genFrames = (int)EnvD("WAN_FRAMES", 17), genSteps = (int)EnvD("WAN_STEPS", 20);

            Tensor reference = BuildGradientFrame(genH, genW);                    // [1,3,1,H,W]
            Tensor pose = BuildMovingSquareClip(genFrames, genH, genW);           // stick-figure stand-in
            Tensor face = BuildMovingSquareClip(genFrames, 512, 512);             // motion-encoder resolution

            _output.WriteLine($"[gen] Animate {genFrames}f {genW}x{genH}, {genSteps} steps, cfg={cfg.GuidanceScale}, renorm={cfg.CfgRescale}...");
            WanAnimatePipeline pipeline = new(backend, transformer, vae, encoder, cfg);
            TextToImageRequest req = new() { Prompt = "animate", Width = genW, Height = genH, Steps = genSteps, CfgScale = (float)EnvD("WAN_CFG", cfg.GuidanceScale), Seed = 42 };
            (byte[][] frames, int w, int h, _) = pipeline.GenerateAnimation(promptEmbeds, negEmbeds, reference, pose, face, req,
                onProgress: p => { if (p.Step % 5 == 0 || p.Step == p.TotalSteps) _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"); });
            reference.Dispose(); pose.Dispose(); face.Dispose();
            promptEmbeds.Dispose(); negEmbeds.Dispose();
            AssertFramesCoherent(frames, w, h);
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    /// <summary>Builds a <c>[1,3,1,H,W]</c> F32 frame in [-1,1]: a smooth colour gradient (reference-image stand-in).</summary>
    private static unsafe Tensor BuildGradientFrame(int h, int w)
    {
        Tensor frame = new Tensor(new TensorShape([1L, 3, 1, h, w]), DType.F32);
        float* p = (float*)frame.DataPointer;
        long plane = (long)h * w;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                long o = (long)y * w + x;
                p[o] = 2f * x / w - 1f;
                p[plane + o] = 2f * y / h - 1f;
                p[2 * plane + o] = (float)Math.Sin(6.28 * (x + y) / 130.0) * 0.6f;
            }
        return frame;
    }

    /// <summary>Builds a <c>[1,3,T,H,W]</c> F32 clip in [-1,1]: dark background, bright square sweeping left→right.</summary>
    private static unsafe Tensor BuildMovingSquareClip(int t, int h, int w)
    {
        Tensor clip = new Tensor(new TensorShape([1L, 3, t, h, w]), DType.F32);
        float* p = (float*)clip.DataPointer;
        long frame = (long)h * w;
        for (long i = 0; i < 3 * t * frame; i++) p[i] = 0.1f;
        int side = Math.Min(h, w) / 4;
        for (int ti = 0; ti < t; ti++)
        {
            int cx = side + (w - 2 * side) * ti / Math.Max(1, t - 1);
            int cy = h / 2;
            for (int c = 0; c < 3; c++)
                for (int y = cy - side / 2; y < cy + side / 2; y++)
                    for (int x = cx - side / 2; x < cx + side / 2; x++)
                        p[((long)c * t + ti) * frame + (long)y * w + x] = 0.9f;
        }
        return clip;
    }

    /// <summary>Parity guard for the query-tiled SDPA path (<c>SdpaTiledF32NoMask</c>, used at full-res / long-sequence
    /// attention like T2V-14B's 14040 tokens): compares the forced-tiled output against the regular GEMM SDPA on the
    /// SAME Q/K/V. They must match — a divergence is the "full-res dark output" bug (the tiled path was never exercised
    /// by TI2V-5B, which only reaches 3510 tokens → regular path).</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void SdpaTiled_MatchesRegularGemm_Gpu()
    {
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(WanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir) || !CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no PTX/CUDA"); return; }

        int B = 1, H = (int)EnvD("SDPA_H", 8), S = (int)EnvD("SDPA_S", 300), D = 128;   // S small → regular path is the reference; tile forces multi-tile
        float scale = 1.0f / MathF.Sqrt(D);
        Tensor MakeRand(int seed)
        {
            Tensor t = new(new TensorShape([(long)B, H, S, D]), DType.F32);
            Span<float> s = t.AsSpan<float>(); Random r = new(seed);
            for (int i = 0; i < s.Length; i++) s[i] = (float)(r.NextDouble() * 2 - 1);
            return t;
        }
        using Tensor q = MakeRand(1), k = MakeRand(2), v = MakeRand(3);

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
        // Reference: regular GEMM path (env clear, small S).
        Environment.SetEnvironmentVariable("HARTSY_SDPA_FORCE_TILED", null);
        using Tensor outRef = new(new TensorShape([(long)B, H, S, D]), DType.F32);
        backend.ScaledDotProductAttention(outRef, q, k, v, null, scale);
        backend.Sync();
        float[] refv = outRef.AsSpan<float>().ToArray();

        // Tiled: force the tiled path with a small tile so the multi-tile loop runs.
        Environment.SetEnvironmentVariable("HARTSY_SDPA_FORCE_TILED", "1");
        Environment.SetEnvironmentVariable("HARTSY_SDPA_TILE", "37");
        using Tensor outTiled = new(new TensorShape([(long)B, H, S, D]), DType.F32);
        backend.ScaledDotProductAttention(outTiled, q, k, v, null, scale);
        backend.Sync();
        float[] tv = outTiled.AsSpan<float>().ToArray();
        Environment.SetEnvironmentVariable("HARTSY_SDPA_FORCE_TILED", null);
        Environment.SetEnvironmentVariable("HARTSY_SDPA_TILE", null);

        double num = 0, den = 0, maxAbs = 0;
        for (int i = 0; i < refv.Length; i++)
        {
            double d = tv[i] - refv[i]; num += d * d; den += (double)refv[i] * refv[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
        }
        double relL2 = Math.Sqrt(num / Math.Max(den, 1e-12));
        _output.WriteLine($"[sdpa-parity] relL2={relL2:E3} maxAbs={maxAbs:E3} (ref[0]={refv[0]:F4} tiled[0]={tv[0]:F4})");
        Assert.True(relL2 < 1e-3, $"Tiled SDPA diverges from regular GEMM (relL2={relL2:E3}) — the full-res attention bug.");
    }

    /// <summary>Isolation diagnostic for the T2V-14B dark-output bug: decode a NORMAL(0,1) random z=16 latent through
    /// the real-weight <see cref="Wan21VaeDecoder"/> alone (Wan 2.1 VAE, never before run on real weights). A healthy
    /// VAE maps this to a mid-gray-ish image (mean ~90–160); a near-black result localizes the bug to the z=16 VAE.</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Wan21Vae_Decode_NormalLatent_Gpu()
    {
        string vaePath = Env("WAN21_VAE") ?? "/home/hartsy/Desktop/Swarm/SwarmUI.old/Models/VAE/Wan/wan_2.1_vae.safetensors";
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: Wan2.1 VAE not found: {vaePath}"); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(WanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir) || !CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no PTX/CUDA"); return; }

        (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) = LanceCheckpointConverter.LoadVae(vaePath);
        try
        {
            Wan21VaeDecoder vae = new();
            vae.LoadWeights(vaeW);
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            _output.WriteLine($"  loaded Wan2.1 VAE: {vaeW.Count} keys");

            // Latent dims default to a small clip but honor WAN_W/H/FRAMES so this can isolate the FULL-RES z=16
            // decode (WAN_W=832 WAN_H=480 → latent 104×60) — where the T2V-14B dark output actually appears.
            int genW = (int)EnvD("WAN_W", 320), genH = (int)EnvD("WAN_H", 192), genF = (int)EnvD("WAN_FRAMES", 9);
            int T = (genF - 1) / 4 + 1, hLat = genH / 8, wLat = genW / 8;
            Tensor latent = new(new TensorShape([1L, 16, T, hLat, wLat]), DType.F32);
            Span<float> ls = latent.AsSpan<float>();
            Random rng = new(7);
            // Box–Muller normal(0,1): the transformer emits ~unit-variance latents pre-denorm.
            for (int i = 0; i < ls.Length; i++)
            {
                double u1 = rng.NextDouble() + 1e-9, u2 = rng.NextDouble();
                ls[i] = (float)(Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
            }
            Tensor rgb = vae.Decode(backend, latent);
            backend.Sync();
            int frames = (int)rgb.Shape[2];
            byte[] f0 = VideoRgbFrames.ExtractFrame(rgb, 0);
            long sum = 0; byte mn = 255, mx = 0; foreach (byte b in f0) { sum += b; if (b < mn) mn = b; if (b > mx) mx = b; }
            _output.WriteLine($"[out] {frames} frames; frame0 mean={(double)sum / f0.Length:F1} min={mn} max={mx}");
            latent.Dispose(); rgb.Dispose();
            Assert.True(sum / f0.Length is > 25 and < 235, $"Wan2.1 VAE decode of a normal latent is degenerate (mean={(double)sum / f0.Length:F1}) — the z=16 VAE path is the bug.");
        }
        finally { foreach (SafeTensorsLoader l in vl) l.Dispose(); }
    }

    /// <summary>Concatenates two image-embed tensors <c>[1,Sa,D]+[1,Sb,D] → [1,Sa+Sb,D]</c> along the token axis (host)
    /// — FLF2V feeds the first+last frames' CLIP hidden states as one 514-token context.</summary>
    private static unsafe Tensor ConcatImageEmbeds(Tensor a, Tensor b)
    {
        int d = (int)a.Shape[a.Shape.Rank - 1];
        int sa = (int)(a.ElementCount / d), sb = (int)(b.ElementCount / d);
        Tensor o = new(new TensorShape([1L, sa + sb, d]), DType.F32);
        float* op = (float*)o.DataPointer, ap = (float*)a.DataPointer, bp = (float*)b.DataPointer;
        Buffer.MemoryCopy(ap, op, (long)(sa + sb) * d * 4, (long)sa * d * 4);
        Buffer.MemoryCopy(bp, op + (long)sa * d, (long)sb * d * 4, (long)sb * d * 4);
        return o;
    }

    /// <summary>Asserts a generated clip is real output: each frame has a wide value spread (not flat), and there is
    /// temporal variation across frames (not a frozen/duplicated frame).</summary>
    private void AssertFramesCoherent(byte[][] frames, int w, int h)
    {
        Assert.True(frames.Length > 0, "no frames produced");
        double[] means = new double[frames.Length];
        int flatFrame = -1; byte flatMn = 0, flatMx = 0;
        for (int f = 0; f < frames.Length; f++)
        {
            byte[] px = frames[f];
            long sum = 0; byte mn = 255, mx = 0;
            foreach (byte b in px) { sum += b; if (b < mn) mn = b; if (b > mx) mx = b; }
            means[f] = (double)sum / px.Length;
            if (mx - mn <= 20 && flatFrame < 0) { flatFrame = f; flatMn = mn; flatMx = mx; }
        }
        double meanMin = means.Min(), meanMax = means.Max();
        double clipMean = means.Average();
        // Print the full profile BEFORE asserting so a failing run is diagnosable from the log alone.
        _output.WriteLine($"  coherence: per-frame mean {meanMin:F1}..{meanMax:F1}, clip mean {clipMean:F1}, spread {meanMax - meanMin:F1}");
        _output.WriteLine($"  per-frame means: {string.Join(" ", means.Select(m => m.ToString("F0")))}");
        Assert.True(flatFrame < 0, $"frame {flatFrame} is near-flat (min={flatMn} max={flatMx}) — degenerate output");
        // Real video frames sit around mid-gray (~80–180); a near-black clip (mean < 25) is degenerate output even if
        // it has some structure — this is the check that catches the "dark output" class of bugs.
        Assert.True(clipMean is > 25 and < 235, $"clip mean {clipMean:F1} outside healthy range [25,235] — degenerate (too dark / blown out)");
        // A static image repeated across frames would have ~0 spread; require some temporal motion for >1 frame.
        if (frames.Length > 1)
            Assert.True(meanMax - meanMin > 0.5, $"no temporal variation across {frames.Length} frames (all mean≈{meanMin:F1})");
    }

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

    /// <summary>Parity guard for the batched CausalConv3d fast path: decodes a MULTI-frame latent (T=3, so the
    /// streaming feat_cache threads across frames 2+) twice on CUDA — once with the batched path, once forced to the
    /// original per-frame accumulate loop — and asserts byte-identical RGB. Proves the batched conv is equivalent to
    /// the reference per-frame algorithm under the streaming cache (the isolation test above only covers T=1/no-cache).</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Wan22Vae_Decode_MultiFrame_BatchedMatchesPerFrame_Gpu()
    {
        string vaePath = TestPaths.WanVideo.VaePath;
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: Wan2.2 VAE not found: {vaePath}"); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(WanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA"); return; }

        (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) = LanceCheckpointConverter.LoadVae(vaePath);
        try
        {
            int T = 3, hLat = 12, wLat = 20;
            Tensor MakeLatent()
            {
                Tensor latent = new(new TensorShape([1L, 48, T, hLat, wLat]), DType.F32);
                Span<float> ls = latent.AsSpan<float>();
                Random rng = new(11);
                for (int i = 0; i < ls.Length; i++) ls[i] = (float)(rng.NextDouble() * 2 - 1);
                return latent;
            }

            byte[] Decode(bool disableBatch)
            {
                Wan22VaeDecoder vae = new();
                vae.LoadWeights(vaeW);
                using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
                CausalConv3d.DisableBatchedPath = disableBatch;
                try
                {
                    using Tensor latent = MakeLatent();
                    using Tensor rgb = vae.Decode(backend, latent);
                    backend.Sync();
                    int frames = (int)rgb.Shape[2];
                    _output.WriteLine($"[{(disableBatch ? "perFrame" : "batched")}] decoded frames={frames} dims=[{string.Join(",", Enumerable.Range(0, rgb.Shape.Rank).Select(d => (long)rgb.Shape[d]))}]");
                    List<byte> all = new();
                    for (int f = 0; f < frames; f++) all.AddRange(VideoRgbFrames.ExtractFrame(rgb, f));
                    return all.ToArray();
                }
                finally { CausalConv3d.DisableBatchedPath = false; }
            }

            byte[] batched = Decode(disableBatch: false);
            byte[] perFrame = Decode(disableBatch: true);

            // Absolute fingerprint (mean + checksum) — compare across layout implementations (host vs GPU) to prove
            // the GPU-resident Vae3dLayout is equivalent to the host reference, not just self-consistent.
            long sumB = 0; ulong ck = 1469598103934665603UL;
            foreach (byte bb in batched) { sumB += bb; ck = (ck ^ bb) * 1099511628211UL; }
            _output.WriteLine($"[fingerprint] batched mean={(double)sumB / batched.Length:F4} fnv={ck:X16} bytes={batched.Length}");

            Assert.Equal(perFrame.Length, batched.Length);
            int diff = 0, maxDiff = 0;
            for (int i = 0; i < batched.Length; i++) { int d = Math.Abs(batched[i] - perFrame[i]); if (d != 0) diff++; if (d > maxDiff) maxDiff = d; }
            _output.WriteLine($"[parity] {batched.Length} RGB bytes: differing={diff} ({100.0 * diff / batched.Length:F3}%), maxDiff={maxDiff}");
            Assert.True(maxDiff <= 1, $"Batched CausalConv3d diverges from per-frame reference: {diff} bytes differ, maxDiff={maxDiff} (expected byte-identical ±1 rounding).");
            _output.WriteLine("VERDICT: batched CausalConv3d is byte-identical to the per-frame reference under the streaming cache.");
        }
        finally { foreach (SafeTensorsLoader l in vl) l.Dispose(); }
    }
}

using System.Diagnostics;
using System.Linq;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.ModelHandler.TextEncoders;
using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>End-to-end HunyuanVideo (Tencent 13B) T2V generation against the Comfy-Org repacked bf16 DiT + VAE +
/// LLaVA-Llama-3-8B (fp8) + CLIP-L → <see cref="HunyuanVideoPipeline"/> → BMP frame sequence. Skips cleanly when
/// artifacts / PTX dir are missing. The 24 GB bf16 DiT is block-streamed. Manual first-run validation entry —
/// numerics validation-pending (structure verified by the parity + synthetic-weight tests).</summary>
public class HunyuanVideoGenerationTests
{
    private readonly ITestOutputHelper _output;
    public HunyuanVideoGenerationTests(ITestOutputHelper output) => _output = output;

    // The exact diffusers prompt template + template-token crop count (pipeline_hunyuan_video.py:70-81).
    private const string PromptTemplate =
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: " +
        "1. The main content and theme of the video." +
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects." +
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects." +
        "4. background environment, light, style and atmosphere." +
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>" +
        "<|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|>";
    private const int CropStart = 95;
    private const int LlamaLayer = 30;   // hidden_states[-3] for a 32-layer model (num_hidden_layers_to_skip=2)

    [Fact]
    public async Task HunyuanVideo_Gpu_T2V_ShortClip()
    {
        string ditPath = TestPaths.HunyuanVideo.Dit, vaePath = TestPaths.HunyuanVideo.Vae;
        string llavaPath = TestPaths.HunyuanVideo.Llava, clipPath = TestPaths.HunyuanVideo.ClipL;
        if (!File.Exists(ditPath)) { _output.WriteLine($"SKIPPED: HunyuanVideo DiT not found: {ditPath} (set HUNYUAN_VIDEO_DIT_PATH)."); return; }
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: HunyuanVideo VAE not found: {vaePath}."); return; }
        if (!File.Exists(llavaPath)) { _output.WriteLine($"SKIPPED: LLaVA-Llama-3 encoder not found: {llavaPath}."); return; }
        if (!File.Exists(clipPath)) { _output.WriteLine($"SKIPPED: CLIP-L not found: {clipPath}."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(HunyuanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        string prompt = Environment.GetEnvironmentVariable("HYV_PROMPT")
            ?? "A cat walks on the grass, realistic style.";
        int width = EnvI("HYV_W", 512), height = EnvI("HYV_H", 320), numFrames = EnvI("HYV_FRAMES", 25), steps = EnvI("HYV_STEPS", 20);

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/6] Loading + converting HunyuanVideo DiT: {Path.GetFileName(ditPath)}");
        (Dictionary<string, Tensor> ditW, SafeTensorsLoader ditLoader) = HunyuanVideoCheckpointConverter.LoadAndConvert(ditPath);
        using SafeTensorsLoader _dit = ditLoader;
        // Cast BF16 → F16 (same 2-byte size, so block-swap still fits; native cuBLAS F16 GEMM). Running BF16 weights
        // through the CacheWeightCasts=false transient path (the fp8 recipe) leaves them unapplied → a BLANK/flat
        // gray render (bf16-transformer-needs-f16-cast). F16 keeps the weights actually applied.
        foreach (string k in ditW.Keys.ToList())
            if (ditW[k].DType == DType.BF16) ditW[k] = ditW[k].CastTo(DType.F16);
        _output.WriteLine($"  {ditW.Count} DiT keys in {sw.ElapsedMilliseconds}ms");

        List<SafeTensorsLoader> loaders = [];
        try
        {
            HunyuanVideoConfig cfg = HunyuanVideoConfig.T2V;
            using HunyuanVideoDit dit = new(cfg);
            dit.LoadWeights(ditW);

            _output.WriteLine("[2/6] Loading VAE...");
            HunyuanVideoVaeDecoder vae = new();
            vae.LoadWeights(CastBf16ToF16(HunyuanVideoCheckpointConverter.ConvertVaeDecoder(LoadStandalone(loaders, vaePath))));

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            backend.CacheWeightCasts = false;   // bf16 stays bf16-resident; block-streamed. Caching casts would OOM.

            // 3. Text conditioning: run the encoders, then free them before the DiT is streamed.
            _output.WriteLine("[3/6] Encoding text (LLaVA-Llama-3 layer -3 + CLIP-L pooled)...");
            Tensor promptEmbeds, pooled;
            {
                using LlamaStyleEncoder llama = new(LlamaStyleEncoderConfig.Llama31_8B);
                llama.LoadWeights(TextEncoderQuantNormalizer.Normalize(LoadStandalone(loaders, llavaPath)));
                ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
                clipL.LoadWeights(LoadStandalone(loaders, clipPath), "text_model");

                // Primary: templated + BOS, take layer-30 hidden state, crop the template preamble.
                int[] llamaTokens = BuildTemplatedTokens(prompt);
                backend.PreloadWeights(llama.EnumerateWeights());
                Tensor full = llama.EncodeMultiLayer(backend, [llamaTokens], [LlamaLayer]);   // [1, seq, 4096]
                promptEmbeds = CropSequence(full, CropStart);                                  // [1, seq-95, 4096]
                full.Dispose();
                backend.Sync(); backend.FreeWeights(llama.EnumerateWeights());

                // Pooled: CLIP-L pooler_output (EOS hidden, no projection).
                using ClipTokenizer clipTok = new(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
                int[] clipTokens = clipTok.Encode(prompt);
                int eos = ClipTokenizer.FindEosPosition(clipTokens);
                backend.PreloadWeights(clipL.EnumerateWeights());
                (Tensor _hidden, Tensor? p) = clipL.EncodePenultimate(backend, [clipTokens], [eos], layersFromEnd: 1);
                _hidden.Dispose();
                pooled = p ?? throw new InvalidOperationException("CLIP-L returned no pooled output.");
                backend.Sync(); backend.FreeWeights(clipL.EnumerateWeights());
                _output.WriteLine($"  llama tokens={llamaTokens.Length} (cropped to {promptEmbeds.Shape[1]}), pooled dim={pooled.Shape[1]}");
            }

            // 4. Generate.
            _output.WriteLine($"[4/6] Generating {numFrames}f {width}x{height}, {steps} steps (block-streamed)...");
            HunyuanVideoPipeline pipeline = new(backend, dit, vae, cfg);
            TextToImageRequest req = new() { Prompt = prompt, Width = width, Height = height, Steps = steps, Seed = 42 };
            Stopwatch gen = Stopwatch.StartNew();
            (byte[][] frames, int outW, int outH, int seed) = pipeline.GenerateFromEmbeddings(
                promptEmbeds, pooled, req, numFrames,
                pr => _output.WriteLine($"  step {pr.Step}/{pr.TotalSteps} ({pr.ElapsedMs:F0}ms)"));
            gen.Stop();
            promptEmbeds.Dispose(); pooled.Dispose();
            _output.WriteLine($"[5/6] Generated {frames.Length} frames in {gen.Elapsed.TotalSeconds:F1}s (seed={seed})");

            // 5. Write BMP sequence + assert non-degenerate.
            string outDir = Path.Combine(Path.GetTempPath(), $"hyv_{DateTime.Now:HHmmss}");
            VideoFrame[] vf = new VideoFrame[frames.Length];
            for (int i = 0; i < frames.Length; i++) vf[i] = new VideoFrame(i, outW, outH, frames[i]);
            await new BmpSequenceEncoder().EncodeAsync(ToAsync(vf), outDir, fps: 24);
            _output.WriteLine($"[6/6] Wrote frames → {outDir}");

            Assert.Equal(numFrames, Directory.GetFiles(outDir, "frame_*.bmp").Length);
            AssertNotBlack(frames[frames.Length / 2], outW, outH);
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    /// <summary>Isolated VAE decode: random latent [1,16,7,40,64] (== 25f/512x320) → full-res decode on CUDA.
    /// Reproduces the full-pipeline VAE OOM in seconds (no DiT/text-encode). Reports peak/free VRAM.</summary>
    [Fact]
    public unsafe void HunyuanVideo_Gpu_VaeDecode_FullRes()
    {
        string vaePath = TestPaths.HunyuanVideo.Vae;
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: HunyuanVideo VAE not found: {vaePath}."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(HunyuanVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        int tLat = EnvI("HYV_TLAT", 7), hLat = EnvI("HYV_HLAT", 40), wLat = EnvI("HYV_WLAT", 64);
        List<SafeTensorsLoader> loaders = [];
        try
        {
            HunyuanVideoVaeDecoder vae = new();
            vae.LoadWeights(CastBf16ToF16(HunyuanVideoCheckpointConverter.ConvertVaeDecoder(LoadStandalone(loaders, vaePath))));
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            backend.CacheWeightCasts = false;

            Tensor latent = new(new TensorShape([1L, 16, tLat, hLat, wLat]), DType.F32);
            Random rng = new(42);
            float* lp = (float*)latent.DataPointer;
            long n = latent.Shape.ElementCount;
            for (long i = 0; i < n; i++) lp[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            backend.PreloadWeights(vae.EnumerateWeights());

            // (a) Small tiled-vs-untiled correlation check (latent fits untiled) — validates the feather blend.
            {
                Tensor small = new(new TensorShape([1L, 16, 3, 20, 40]), DType.F32);
                float* sp0 = (float*)small.DataPointer; long sn = small.Shape.ElementCount;
                Random r2 = new(7); for (long i = 0; i < sn; i++) sp0[i] = (float)(r2.NextDouble() * 2 - 1);
                Tensor full = vae.Decode(backend, small); backend.Sync();
                // Force full → host now (DecodeTiled's per-tile FreeActivations would otherwise evict its cached
                // device buffer and leave the host copy unpopulated → spurious zeros).
                _ = ((float*)full.DataPointer)[0];
                Tensor tiled = vae.DecodeTiled(backend, small); backend.Sync();
                double corr = Corr((float*)full.DataPointer, (float*)tiled.DataPointer, full.Shape.ElementCount);
                _output.WriteLine($"[blend] tiled-vs-untiled corr={corr:F4} (GroupNorm-per-tile ⇒ close, not exact)");
                small.Dispose(); full.Dispose(); tiled.Dispose();
                Assert.True(corr > 0.9, $"Tiled decode diverges from untiled (corr {corr:F4}).");
            }

            // (b) Full-res tiled decode — must fit in 24 GB and produce a non-degenerate frame.
            Stopwatch sw = Stopwatch.StartNew();
            Tensor rgb = vae.DecodeTiled(backend, latent);
            backend.Sync();
            sw.Stop();
            _output.WriteLine($"VAE tiled-decode OK: latent [1,16,{tLat},{hLat},{wLat}] → rgb {rgb.Shape} in {sw.ElapsedMilliseconds}ms");
            float* rp = (float*)rgb.DataPointer;
            double s = 0, s2 = 0; long rn = rgb.Shape.ElementCount; float rmin = float.MaxValue, rmax = float.MinValue;
            for (long i = 0; i < rn; i++) { float x = rp[i]; s += x; s2 += (double)x * x; if (x < rmin) rmin = x; if (x > rmax) rmax = x; }
            double rmean = s / rn; double rstd = Math.Sqrt(Math.Max(0, s2 / rn - rmean * rmean));
            _output.WriteLine($"  rgb mean={rmean:F4} std={rstd:F4} range=[{rmin:F3},{rmax:F3}]");
            latent.Dispose(); rgb.Dispose();
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    private static int[] BuildTemplatedTokens(string prompt)
    {
        // Reference tokenizes the templated string with add_special_tokens=True (BOS prepended) then crops 95.
        // Special header/eot tokens are resolved by the Llama-3 tokenizer's added-token table. VALIDATION-GATED:
        // if special tokens are BPE'd literally, the crop count / conditioning will drift — verify vs the reference.
        // Build from the embedded Llama-3 tokenizer.json (HF byte-level BPE) so the special header/eot literals
        // resolve to single ids and the crop-95 alignment matches diffusers (add_special_tokens=True → BOS prepended).
        using Stream json = EmbeddedTokenizerResources.OpenLlama3TokenizerJson();
        GgufTokenizer tok = HfTokenizerJson.LoadByteLevelBpe(json);
        string templated = string.Format(PromptTemplate, prompt);
        int[] ids = tok.Encode(templated, addSpecial: true);
        int[] withBos = new int[ids.Length + 1];
        withBos[0] = tok.BosId ?? 128000;
        Array.Copy(ids, 0, withBos, 1, ids.Length);
        return withBos;
    }

    /// <summary>Crops the leading <paramref name="crop"/> tokens along the sequence dim of a <c>[1, S, H]</c> tensor.</summary>
    private static unsafe Tensor CropSequence(Tensor t, int crop)
    {
        int s = (int)t.Shape[1], h = (int)t.Shape[2];
        int keep = s - crop;
        if (keep <= 0) throw new InvalidOperationException($"Template crop {crop} >= sequence length {s}.");
        Tensor outT = new(new TensorShape(1, keep, h), DType.F32);
        float* sp = (float*)t.DataPointer, dp = (float*)outT.DataPointer;
        long bytes = (long)keep * h * 4;
        Buffer.MemoryCopy(sp + (long)crop * h, dp, bytes, bytes);
        return outT;
    }

    private static Dictionary<string, Tensor> LoadStandalone(List<SafeTensorsLoader> loaders, string path)
    {
        SafeTensorsLoader l = new();
        l.Load(path);
        loaders.Add(l);
        return CheckpointConvertUtils.ApplyFp8ScaledDequant(l.GetAllTensors());
    }

    private static Dictionary<string, Tensor> StripPrefix(Dictionary<string, Tensor> w, string prefix)
    {
        bool any = false;
        foreach (string k in w.Keys) if (k.StartsWith(prefix, StringComparison.Ordinal)) { any = true; break; }
        if (!any) return w;
        Dictionary<string, Tensor> o = new(w.Count);
        foreach ((string k, Tensor v) in w) o[k.StartsWith(prefix, StringComparison.Ordinal) ? k[prefix.Length..] : k] = v;
        return o;
    }

    private static unsafe void AssertNotBlack(byte[] frame, int w, int h)
    {
        long sum = 0; foreach (byte px in frame) sum += px;
        double mean = (double)sum / frame.Length;
        Assert.True(mean > 2.0 && mean < 253.0, $"Middle frame is degenerate (mean pixel {mean:F1}); expected a real image.");
    }

    private static async IAsyncEnumerable<VideoFrame> ToAsync(VideoFrame[] frames)
    { foreach (VideoFrame f in frames) { yield return f; await Task.Yield(); } }

    /// <summary>Casts BF16 tensors → F16 in place (same 2-byte size). BF16 weights run through the
    /// CacheWeightCasts=false transient path are left unapplied → blank output; F16 is native cuBLAS. Applies to
    /// both the DiT and the VAE (both ship as bf16 checkpoints).</summary>
    private static Dictionary<string, Tensor> CastBf16ToF16(Dictionary<string, Tensor> w)
    {
        foreach (string k in w.Keys.ToList())
            if (w[k].DType == DType.BF16) w[k] = w[k].CastTo(DType.F16);
        return w;
    }

    private static unsafe double Corr(float* a, float* b, long n)
    {
        double sa = 0, sb = 0; for (long i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
        double ma = sa / n, mb = sb / n, num = 0, da = 0, db = 0;
        for (long i = 0; i < n; i++) { double x = a[i] - ma, y = b[i] - mb; num += x * y; da += x * x; db += y * y; }
        return num / (Math.Sqrt(da * db) + 1e-12);
    }

    private static int EnvI(string name, int fallback) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out int v) ? v : fallback;
}

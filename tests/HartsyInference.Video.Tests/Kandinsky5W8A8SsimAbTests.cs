using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>W8A8 stage 4a/4c authoritative gate (docs/Checklists/W8A8_HANDOFF.md "What's left" #1/#3):
/// end-to-end SSIM on decoded frames — the same ship-metric gate every other fleet feature (Flux/SDXL/
/// Ideogram/Krea2 step-cache, Chroma fused-QKV) is measured against — NOT the block-level velocity relL2
/// proxy (<see cref="Kandinsky5W8A8MeasurementTests"/>).
///
/// <para><b>One generation per process, by design.</b> An earlier version of this test ran
/// floorA/floorB/candidate back-to-back on one <c>CudaBackend</c>/pipeline instance and hit
/// <c>CUDA_ERROR_ILLEGAL_ADDRESS</c> on the SECOND generation (both baseline, W8A8 never engaged) —
/// a multi-generation-reuse issue in the Kandinsky5 Free/Preload weight cycle, unexercised by the
/// existing single-gen <c>Kandinsky5_Gpu_T2V_ShortClip</c> test. An illegal-address error poisons the
/// whole CUDA context, so a second in-process attempt (even on a fresh backend) can inherit the
/// corruption. This test instead runs exactly ONE generation per process (arm selected by
/// <c>K5V_ARM</c>) and dumps raw RGB frames to disk; <see cref="Ssim_Compare_Dumps"/> is a separate,
/// CPU-only, no-CUDA test that SSIM-compares the dumps across process runs. See
/// <c>docs/Checklists/W8A8_HANDOFF.md</c> for the reuse-crash writeup (flagged, not fixed here — off
/// the W8A8 critical path, needs its own investigation).</para>
///
/// Run all three arms then the comparison (CVD=1 = the 3060, the IMMA target class):
/// <code>
/// CUDA_VISIBLE_DEVICES=1 K5V_ARM=floorA dotnet test --filter "FullyQualifiedName~W8A8_E2E_SingleGen"
/// CUDA_VISIBLE_DEVICES=1 K5V_ARM=floorB dotnet test --filter "FullyQualifiedName~W8A8_E2E_SingleGen"
/// CUDA_VISIBLE_DEVICES=1 K5V_ARM=w8a8    dotnet test --filter "FullyQualifiedName~W8A8_E2E_SingleGen"
/// dotnet test --filter "FullyQualifiedName~Ssim_Compare_Dumps"
/// </code></summary>
[Trait("Category", "W8A8Bench")]
public sealed unsafe class Kandinsky5W8A8SsimAbTests
{
    private readonly ITestOutputHelper _output;
    public Kandinsky5W8A8SsimAbTests(ITestOutputHelper output) => _output = output;

    private static string T2VDir => Environment.GetEnvironmentVariable("KANDINSKY5_T2V_DIR")
        ?? Path.Combine(TestPaths.ModelsDir, "Stable-Diffusion", "Kandinsky5", "Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers");

    private static string DumpRoot => Path.Combine(Path.GetTempPath(), "k5w8a8_ab");

    /// <summary>Runs exactly one generation for the arm named by <c>K5V_ARM</c> (<c>floorA</c>/<c>floorB</c>
    /// pick baseline W8A8=off twice for the determinism-floor control, <c>w8a8</c> picks W8A8=on) and
    /// dumps interleaved-RGB frames + a manifest to <see cref="DumpRoot"/>/&lt;arm&gt;/.</summary>
    [Fact]
    public void W8A8_E2E_SingleGen()
    {
        string arm = Environment.GetEnvironmentVariable("K5V_ARM") ?? "";
        if (arm is not ("floorA" or "floorB" or "w8a8"))
        { _output.WriteLine("SKIPPED: set K5V_ARM=floorA|floorB|w8a8 to run one generation."); return; }

        string transformerDir = Path.Combine(T2VDir, "transformer");
        string vaePath = Path.Combine(T2VDir, "vae", "diffusion_pytorch_model.safetensors");
        if (!Directory.Exists(transformerDir)) { _output.WriteLine($"SKIPPED: T2V transformer dir not found: {transformerDir} (set KANDINSKY5_T2V_DIR)."); return; }
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: T2V VAE not found: {vaePath}."); return; }
        if (!File.Exists(TestPaths.Kandinsky5.PromptQwenEmbeds) || !File.Exists(TestPaths.Kandinsky5.PromptClipPooled)
            || !File.Exists(TestPaths.Kandinsky5.NegPromptQwenEmbeds) || !File.Exists(TestPaths.Kandinsky5.NegPromptClipPooled))
        { _output.WriteLine("SKIPPED: pre-computed Qwen/CLIP embeddings missing (see dump_kandinsky5_embeddings.py)."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Kandinsky5W8A8SsimAbTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }
        if (!File.Exists(Path.Combine(ptxDir, "w8a8.ptx"))) { _output.WriteLine("SKIPPED: w8a8.ptx missing"); return; }
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        int width = EnvI("K5V_W", 512), height = EnvI("K5V_H", 512);
        int numFrames = EnvI("K5V_FRAMES", 25), steps = EnvI("K5V_STEPS", 30);
        const float cfg = 5.0f;
        bool w8a8 = arm == "w8a8";
        // Diagnostic toggle (2026-07-23): the real e2e crash (see W8A8_HANDOFF.md) is a Tensor.Dispose()
        // illegal-address inside CausalConv3d.Forward's batched fast path. DisableBatchedPath is the
        // test-only escape hatch to the older per-frame loop, provided specifically to prove the two
        // paths produce identical output — reuse it here to isolate whether the fast path is the culprit.
        CausalConv3d.DisableBatchedPath = Environment.GetEnvironmentVariable("K5V_DISABLE_BATCHED_CONV") == "1";
        if (CausalConv3d.DisableBatchedPath)
            _output.WriteLine("DIAGNOSTIC: CausalConv3d.DisableBatchedPath=true (per-frame VAE conv loop)");

        (Kandinsky5CheckpointConverter.ConvertedWeights converted, List<SafeTensorsLoader> loaders) =
            Kandinsky5CheckpointConverter.LoadDiffusersFolder(transformerDir);
        try
        {
            Dictionary<string, Tensor> tw = new(converted.Transformer.Count);
            foreach ((string k, Tensor v) in converted.Transformer)
                tw[k] = v.DType == DType.BF16 ? v.CastTo(DType.F16) : v;
            Kandinsky5Config config = Kandinsky5Config.VideoLite2B;
            using Kandinsky5Transformer transformer = new(config);
            transformer.LoadWeights(tw);

            using SafeTensorsLoader vaeLoader = new();
            vaeLoader.Load(vaePath);
            Dictionary<string, Tensor> vw = CastBf16ToF16(vaeLoader.GetAllTensors());
            HunyuanVideoVaeDecoder vae = new();
            vae.LoadWeights(vw);

            // Explicit construct/dispose (NOT `using`) so a real mid-generation exception isn't masked by
            // a second exception thrown from Dispose() on the now-broken CUDA context — `using`'s implicit
            // Dispose() call during exception unwind replaces the original exception with whatever Dispose
            // throws, and prior runs of this test showed exactly that pattern (illegal-address always
            // reported at Dispose, real call site unknown). Log the REAL exception here before it can be
            // masked, then dispose defensively.
            CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            try
            {
                backend.EnableW8A8 = w8a8;
                _output.WriteLine($"Device: {backend.Capabilities.Name} (CUDA_VISIBLE_DEVICES=" +
                    $"{Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") ?? "unset"}), arm={arm}, W8A8={w8a8}");

                using Tensor qwen = LoadF32Tensor(TestPaths.Kandinsky5.PromptQwenEmbeds, config.InTextDim);
                using Tensor clip = LoadPooled(TestPaths.Kandinsky5.PromptClipPooled, config.InTextDim2);
                using Tensor negQwen = LoadF32Tensor(TestPaths.Kandinsky5.NegPromptQwenEmbeds, config.InTextDim);
                using Tensor negClip = LoadPooled(TestPaths.Kandinsky5.NegPromptClipPooled, config.InTextDim2);

                // K5V_SKIP_CALIBRATION=1: same-build w8a8 arm WITHOUT SmoothQuant, isolating the build
                // change (w8a8.ptx recompiled 11.5->13.0.88 this session) from the smoothing effect itself
                // in the SSIM comparison — the prior 0.9211 baseline was measured on the old PTX build.
                if (w8a8 && Environment.GetEnvironmentVariable("K5V_SKIP_CALIBRATION") != "1")
                {
                    System.Diagnostics.Stopwatch calSw = System.Diagnostics.Stopwatch.StartNew();
                    int applied = CalibrateSmoothQuant(backend, transformer, config, qwen, clip, width, height, numFrames, steps, _output);
                    calSw.Stop();
                    _output.WriteLine($"SmoothQuant calibration: {applied} weights smoothed in {calSw.Elapsed.TotalSeconds:F1}s");
                }
                else if (w8a8)
                {
                    _output.WriteLine("K5V_SKIP_CALIBRATION=1: running w8a8 WITHOUT SmoothQuant (build-drift control).");
                }

                Kandinsky5VideoPipeline pipeline = new(backend, transformer, vae, config);
                TextToImageRequest req = new()
                { Prompt = "(embeddings)", Width = width, Height = height, Steps = steps, CfgScale = cfg, Seed = 42 };

                System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
                (byte[][] frames, int outW, int outH, _) = pipeline.GenerateFromEmbeddings(qwen, clip, negQwen, negClip, req, numFrames,
                    pr => { if (pr.Step % 10 == 0 || pr.Step == pr.TotalSteps) _output.WriteLine($"  step {pr.Step}/{pr.TotalSteps}"); });
                sw.Stop();
                _output.WriteLine($"[{arm}] {frames.Length} frames in {sw.Elapsed.TotalSeconds:F1}s (W8A8={w8a8})");

                string dumpArm = w8a8 && Environment.GetEnvironmentVariable("K5V_SKIP_CALIBRATION") == "1" ? "w8a8nosmooth" : arm;
                string dumpDir = Path.Combine(DumpRoot, dumpArm);
                Directory.CreateDirectory(dumpDir);
                foreach (string f in Directory.GetFiles(dumpDir, "*.rgb")) File.Delete(f);
                for (int i = 0; i < frames.Length; i++)
                    File.WriteAllBytes(Path.Combine(dumpDir, $"frame_{i:D3}.rgb"), frames[i]);
                File.WriteAllText(Path.Combine(dumpDir, "manifest.txt"), $"{outW} {outH} {frames.Length}");
                _output.WriteLine($"Dumped raw RGB frames -> {dumpDir}");

                // Human-viewable BMP sequence for eyeball quality inspection (same pattern as
                // Kandinsky5VideoGenerationTests) — separate from the raw dumps SSIM reads.
                string bmpDir = Path.Combine(dumpDir, "bmp");
                VideoFrame[] vf = new VideoFrame[frames.Length];
                for (int i = 0; i < frames.Length; i++) vf[i] = new VideoFrame(i, outW, outH, frames[i]);
#pragma warning disable xUnit1031 // debug-artifact write, not test logic; console test host has no sync context to deadlock on
                new BmpSequenceEncoder().EncodeAsync(ToAsync(vf), bmpDir, fps: 24).GetAwaiter().GetResult();
#pragma warning restore xUnit1031
                _output.WriteLine($"Wrote BMP frames -> {bmpDir}");
            }
            catch (Exception realEx)
            {
                _output.WriteLine($"REAL EXCEPTION (pre-Dispose): {realEx}");
                try { _output.WriteLine($"FreeMemoryBytes at throw: {backend.FreeMemoryBytes() >> 20} MB"); }
                catch (Exception diagEx) { _output.WriteLine($"(FreeMemoryBytes itself threw: {diagEx.Message})"); }
                throw;
            }
            finally
            {
                try { backend.Dispose(); }
                catch (Exception disposeEx) { _output.WriteLine($"DISPOSE-TIME EXCEPTION (masked in earlier runs): {disposeEx}"); }
            }
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    /// <summary>SmoothQuant calibration (W8A8_HANDOFF.md item 1, offline-gate-confirmed 2026-07-24 on real
    /// Kandinsky5 layers: relL2 drops ~40%, alpha~0.7-0.8 optimal, not layer-idiosyncratic; Pearson r of
    /// per-channel absmax profiles drops to 0.43 between schedule extremes, so calibration MUST
    /// max-aggregate across the schedule, not a single sample). Runs a few representative timesteps
    /// (early/mid/late) through the real transformer with the capture hook always re-arming (so every
    /// W8A8-eligible Linear in the model gets captured, not just one), accumulates per-input-channel
    /// activation absmax (max across BOTH rows-within-a-capture and timesteps) and per-weight
    /// per-input-channel weight absmax (t-invariant, computed once), then calls
    /// <see cref="CudaBackend.SetW8A8SmoothingScale"/> per weight with s_j = (actMax_j/wMax_j)^alpha.
    /// Calibration necessarily runs with EnableW8A8=true (the capture hook piggybacks on the same
    /// eligibility gate as the real W8A8 dispatch — there is no separate calibration-mode bypass), so the
    /// captured activations reflect layers AFTER unsmoothed-int8 noise from earlier blocks in the same
    /// forward pass, not pure F16 — a minor, accepted approximation (the noise doesn't change WHICH
    /// channels are outliers, just adds jitter to the magnitude estimate). Any already-quantized weight
    /// cache entries this pre-generation pass creates get evicted by SetW8A8SmoothingScale and
    /// re-quantized smoothed on the real generation's first use.</summary>
    private static unsafe int CalibrateSmoothQuant(CudaBackend backend, Kandinsky5Transformer transformer,
        Kandinsky5Config config, Tensor qwen, Tensor clip, int width, int height, int numFrames, int steps,
        ITestOutputHelper output)
    {
        int tLat = (numFrames - 1) / 4 + 1, hLat = height / 8, wLat = width / 8;
        int latCh = config.InVisualDim;
        TensorShape latentShape = new TensorShape([1L, latCh, tLat, hLat, wLat]);
        TensorShape maskShape = new TensorShape([1L, 1, tLat, hLat, wLat]);
        (float scaleT, float scaleH, float scaleW) = Kandinsky5VideoPipeline.GetRopeScaleFactor(height, width);

        FlowMatchEulerDiscreteScheduler scheduler = new(5.0f);
        scheduler.SetTimesteps(steps);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        int[] probeSteps = steps >= 3 ? [0, steps / 2, steps - 1] : [0];

        using Tensor noisy = SeedGenerator.CreateNoise(latentShape, seed: 42);
        using Tensor condLatent = ZerosF32(latentShape);
        using Tensor condMask = ZerosF32(maskShape);
        using Tensor packed = PackVisualCondCalib(noisy, condLatent, condMask, latCh);

        Dictionary<Tensor, float[]> actMax = new();
        Dictionary<Tensor, float[]> wMax = new();
        Dictionary<Tensor, int> kOf = new();

        backend.PreloadWeights(transformer.EnumerateWeights());
        backend.EnableW8A8 = true;
        foreach (int stepIdx in probeSteps)
        {
            float t = timesteps[stepIdx];
            Action<float[], int, int, float[], int, Tensor> hook = null!;
            hook = (inArr, m, k, wArr, n, wT) =>
            {
                if (!actMax.TryGetValue(wT, out float[]? am))
                {
                    am = new float[k];
                    actMax[wT] = am;
                    kOf[wT] = k;
                }
                for (int r = 0; r < m; r++)
                {
                    int baseIdx = r * k;
                    for (int c = 0; c < k; c++)
                    {
                        float a = MathF.Abs(inArr[baseIdx + c]);
                        if (a > am[c]) am[c] = a;
                    }
                }
                if (!wMax.ContainsKey(wT))
                {
                    float[] wm = new float[k];
                    for (int ni = 0; ni < n; ni++)
                    {
                        int baseIdx = ni * k;
                        for (int c = 0; c < k; c++)
                        {
                            float a = MathF.Abs(wArr[baseIdx + c]);
                            if (a > wm[c]) wm[c] = a;
                        }
                    }
                    wMax[wT] = wm;
                }
                CudaBackend.CaptureW8A8Operands = hook; // keep watching — capture EVERY eligible Linear this pass
            };
            CudaBackend.CaptureW8A8Operands = hook;
            Tensor outp = transformer.ForwardVideo(backend, packed, t, qwen, clip, scaleT, scaleH, scaleW);
            backend.Sync();
            outp.Dispose();
            backend.FreeActivations(trimPool: false);
            CudaBackend.CaptureW8A8Operands = null;
            output.WriteLine($"  calibration pass step {stepIdx} (t={t:F1}): {actMax.Count} distinct weights seen so far");
        }
        backend.FreeWeights(transformer.EnumerateWeights());

        const double alpha = 0.7; // offline gate: relL2 optimum for real Kandinsky5 layers at alpha~0.7-0.8
        int applied = 0;
        foreach ((Tensor w, float[] am) in actMax)
        {
            float[] wm = wMax[w];
            int k = kOf[w];
            float[] s = new float[k];
            for (int c = 0; c < k; c++)
            {
                double sv = am[c] > 0 && wm[c] > 0 ? Math.Pow(am[c], alpha) / Math.Pow(wm[c], 1.0 - alpha) : 1.0;
                s[c] = (float)Math.Clamp(sv, 1e-3, 1e3);
            }
            backend.SetW8A8SmoothingScale(w, s);
            applied++;
        }
        return applied;
    }

    private static Tensor ZerosF32(TensorShape shape)
    {
        Tensor t = new Tensor(shape, DType.F32);
        new Span<float>((float*)t.DataPointer, checked((int)shape.ElementCount)).Clear();
        return t;
    }

    /// <summary>Ports <c>Kandinsky5VideoPipeline.PackVisualCond</c> (private): concat <c>[noisy(16), cond(16), mask(1)]</c> along the channel axis.</summary>
    private static Tensor PackVisualCondCalib(Tensor noisy, Tensor condLatent, Tensor condMask, int latCh)
    {
        long b = noisy.Shape[0], t = noisy.Shape[2], h = noisy.Shape[3], w = noisy.Shape[4];
        Tensor packed = new Tensor(new TensorShape([b, 2L * latCh + 1, t, h, w]), DType.F32);
        long per = t * h * w;
        float* dst = (float*)packed.DataPointer;
        float* pn = (float*)noisy.DataPointer;
        float* pc = (float*)condLatent.DataPointer;
        float* pm = (float*)condMask.DataPointer;
        long chOut = 2L * latCh + 1;
        for (long bi = 0; bi < b; bi++)
        {
            long dstBase = bi * chOut * per;
            Buffer.MemoryCopy(pn + bi * latCh * per, dst + dstBase, latCh * per * 4, latCh * per * 4);
            Buffer.MemoryCopy(pc + bi * latCh * per, dst + dstBase + latCh * per, latCh * per * 4, latCh * per * 4);
            Buffer.MemoryCopy(pm + bi * per, dst + dstBase + 2L * latCh * per, per * 4, per * 4);
        }
        return packed;
    }

    private static async IAsyncEnumerable<VideoFrame> ToAsync(VideoFrame[] frames)
    { foreach (VideoFrame f in frames) { yield return f; await Task.Yield(); } }

    /// <summary>CPU-only, no CUDA: SSIM-compares the frame dumps written by three separate
    /// <see cref="W8A8_E2E_SingleGen"/> process runs (floorA/floorB/w8a8). Determinism-floor SSIM
    /// (floorA vs floorB, both W8A8=off) establishes the noise ceiling; W8A8 vs floorA must clear the
    /// fleet's 0.95 SSIM gate relative to that floor.</summary>
    [Fact]
    public void Ssim_Compare_Dumps()
    {
        (byte[][] frames, int w, int h)? floorA = LoadDump("floorA");
        (byte[][] frames, int w, int h)? floorB = LoadDump("floorB");
        (byte[][] frames, int w, int h)? candidate = LoadDump("w8a8");
        if (floorA is null || floorB is null || candidate is null)
        { _output.WriteLine($"SKIPPED: run all three W8A8_E2E_SingleGen arms first (dumps expected under {DumpRoot})."); return; }

        Assert.Equal(floorA.Value.frames.Length, floorB.Value.frames.Length);
        Assert.Equal(floorA.Value.frames.Length, candidate.Value.frames.Length);

        double floorSsim = MeanSsim(floorA.Value.frames, floorB.Value.frames, floorA.Value.w, floorA.Value.h);
        double w8a8Ssim = MeanSsim(floorA.Value.frames, candidate.Value.frames, floorA.Value.w, floorA.Value.h);
        _output.WriteLine($"Determinism floor SSIM (floorA vs floorB, both W8A8=off): {floorSsim:F4}");
        _output.WriteLine($"W8A8 vs floorA SSIM: {w8a8Ssim:F4}");

        Assert.True(w8a8Ssim >= 0.95, $"W8A8 e2e SSIM {w8a8Ssim:F4} below the fleet 0.95 gate (floor {floorSsim:F4}).");
    }

    /// <summary>Build-drift control (advisor-directed 2026-07-24): disambiguates "SmoothQuant regressed
    /// e2e SSIM" from "the w8a8.ptx recompile (11.5->13.0.88, this session) shifted the baseline" by
    /// comparing floorA against a w8a8 run with calibration SKIPPED (K5V_SKIP_CALIBRATION=1) — same build,
    /// no smoothing. Not a pass/fail gate, just reports the number for comparison against the
    /// smoothed 0.9144 and the prior-session pre-recompile 0.9211.</summary>
    [Fact]
    public void Ssim_Compare_NoSmoothControl()
    {
        (byte[][] frames, int w, int h)? floorA = LoadDump("floorA");
        (byte[][] frames, int w, int h)? noSmooth = LoadDump("w8a8nosmooth");
        if (floorA is null || noSmooth is null)
        { _output.WriteLine($"SKIPPED: run floorA and (K5V_ARM=w8a8 K5V_SKIP_CALIBRATION=1) first (dumps expected under {DumpRoot})."); return; }

        Assert.Equal(floorA.Value.frames.Length, noSmooth.Value.frames.Length);
        double ssim = MeanSsim(floorA.Value.frames, noSmooth.Value.frames, floorA.Value.w, floorA.Value.h);
        _output.WriteLine($"W8A8 (no SmoothQuant, this session's PTX build) vs floorA SSIM: {ssim:F4}");
    }

    private static (byte[][] frames, int w, int h)? LoadDump(string arm)
    {
        string dir = Path.Combine(DumpRoot, arm);
        string manifestPath = Path.Combine(dir, "manifest.txt");
        if (!File.Exists(manifestPath)) return null;
        string[] parts = File.ReadAllText(manifestPath).Split(' ');
        int w = int.Parse(parts[0]), h = int.Parse(parts[1]), count = int.Parse(parts[2]);
        byte[][] frames = new byte[count][];
        for (int i = 0; i < count; i++)
            frames[i] = File.ReadAllBytes(Path.Combine(dir, $"frame_{i:D3}.rgb"));
        return (frames, w, h);
    }

    private static double MeanSsim(byte[][] a, byte[][] b, int width, int height)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
            sum += Helpers.Ssim.Compute(a[i], b[i], width, height);
        return sum / a.Length;
    }

    private static Dictionary<string, Tensor> CastBf16ToF16(Dictionary<string, Tensor> w)
    {
        foreach (string k in w.Keys.ToList())
            if (w[k].DType == DType.BF16) w[k] = w[k].CastTo(DType.F16);
        return w;
    }

    /// <summary>Raw headerless F32 <c>[seq, dim]</c> → <c>[1, seq, dim]</c>.</summary>
    private static Tensor LoadF32Tensor(string path, int embedDim)
    {
        byte[] data = File.ReadAllBytes(path);
        long totalFloats = data.Length / sizeof(float);
        if (totalFloats % embedDim != 0)
            throw new InvalidOperationException($"{path}: {totalFloats} floats not a multiple of {embedDim}.");
        int seqLen = (int)(totalFloats / embedDim);
        Tensor result = new Tensor(new TensorShape(1, seqLen, embedDim), DType.F32);
        fixed (byte* src = data) Buffer.MemoryCopy(src, (void*)result.DataPointer, data.Length, data.Length);
        return result;
    }

    /// <summary>Raw headerless F32 <c>[dim]</c> → <c>[1, dim]</c>.</summary>
    private static Tensor LoadPooled(string path, int embedDim)
    {
        byte[] data = File.ReadAllBytes(path);
        if (data.Length / sizeof(float) != embedDim)
            throw new InvalidOperationException($"{path}: expected {embedDim} floats.");
        Tensor result = new Tensor(new TensorShape(1, embedDim), DType.F32);
        fixed (byte* src = data) Buffer.MemoryCopy(src, (void*)result.DataPointer, data.Length, data.Length);
        return result;
    }

    private static int EnvI(string name, int fallback) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out int v) ? v : fallback;
}

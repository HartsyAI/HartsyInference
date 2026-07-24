using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
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

                Kandinsky5VideoPipeline pipeline = new(backend, transformer, vae, config);
                TextToImageRequest req = new()
                { Prompt = "(embeddings)", Width = width, Height = height, Steps = steps, CfgScale = cfg, Seed = 42 };

                System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
                (byte[][] frames, int outW, int outH, _) = pipeline.GenerateFromEmbeddings(qwen, clip, negQwen, negClip, req, numFrames,
                    pr => { if (pr.Step % 10 == 0 || pr.Step == pr.TotalSteps) _output.WriteLine($"  step {pr.Step}/{pr.TotalSteps}"); });
                sw.Stop();
                _output.WriteLine($"[{arm}] {frames.Length} frames in {sw.Elapsed.TotalSeconds:F1}s (W8A8={w8a8})");

                string dumpDir = Path.Combine(DumpRoot, arm);
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

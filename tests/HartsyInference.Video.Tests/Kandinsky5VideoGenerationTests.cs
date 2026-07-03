using System.Diagnostics;
using System.Linq;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>End-to-end Kandinsky 5.0 T2V Lite (2B) generation against the real diffusers checkpoint
/// (<c>kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers</c>: transformer + the shared HunyuanVideo
/// 3D causal VAE) → <see cref="Kandinsky5VideoPipeline"/> → BMP frame sequence. The dual text-encoder
/// stack (Qwen2.5-VL + CLIP-L) is supplied as the same pre-computed embeddings the T2I e2e uses
/// (snow-leopard prompt, dumped via <c>tests/python-reference/dump_kandinsky5_embeddings.py</c>).
/// Skips cleanly when any artifact is missing. Numerics validation-pending vs the reference pipeline.</summary>
public class Kandinsky5VideoGenerationTests
{
    private readonly ITestOutputHelper _output;
    public Kandinsky5VideoGenerationTests(ITestOutputHelper output) => _output = output;

    private static string T2VDir => Environment.GetEnvironmentVariable("KANDINSKY5_T2V_DIR")
        ?? Path.Combine(TestPaths.ModelsDir, "Stable-Diffusion", "Kandinsky5", "Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers");

    [Fact]
    public async Task Kandinsky5_Gpu_T2V_ShortClip()
    {
        string transformerDir = Path.Combine(T2VDir, "transformer");
        string vaePath = Path.Combine(T2VDir, "vae", "diffusion_pytorch_model.safetensors");
        if (!Directory.Exists(transformerDir)) { _output.WriteLine($"SKIPPED: T2V transformer dir not found: {transformerDir} (set KANDINSKY5_T2V_DIR)."); return; }
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: T2V VAE not found: {vaePath}."); return; }
        if (!File.Exists(TestPaths.Kandinsky5.PromptQwenEmbeds) || !File.Exists(TestPaths.Kandinsky5.PromptClipPooled)
            || !File.Exists(TestPaths.Kandinsky5.NegPromptQwenEmbeds) || !File.Exists(TestPaths.Kandinsky5.NegPromptClipPooled))
        { _output.WriteLine("SKIPPED: pre-computed Qwen/CLIP embeddings missing (see dump_kandinsky5_embeddings.py)."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(Kandinsky5VideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        int width = EnvI("K5V_W", 512), height = EnvI("K5V_H", 512);
        int numFrames = EnvI("K5V_FRAMES", 25), steps = EnvI("K5V_STEPS", 30);
        float cfg = 5.0f;

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/5] Loading T2V transformer (diffusers dir): {transformerDir}");
        (Kandinsky5CheckpointConverter.ConvertedWeights converted, List<SafeTensorsLoader> loaders) =
            Kandinsky5CheckpointConverter.LoadDiffusersFolder(transformerDir);
        try
        {
            // BF16 → F16 (native cuBLAS; the BF16 transient path leaves weights unapplied → blank output).
            Dictionary<string, Tensor> tw = new(converted.Transformer.Count);
            foreach ((string k, Tensor v) in converted.Transformer)
                tw[k] = v.DType == DType.BF16 ? v.CastTo(DType.F16) : v;
            Kandinsky5Config config = Kandinsky5Config.VideoLite2B;
            using Kandinsky5Transformer transformer = new(config);
            transformer.LoadWeights(tw);
            _output.WriteLine($"  transformer ready in {sw.ElapsedMilliseconds}ms ({tw.Count} keys)");

            _output.WriteLine("[2/5] Loading HunyuanVideo VAE (diffusers naming)...");
            using SafeTensorsLoader vaeLoader = new();
            vaeLoader.Load(vaePath);
            Dictionary<string, Tensor> vw = CastBf16ToF16(vaeLoader.GetAllTensors());
            HunyuanVideoVaeDecoder vae = new();
            vae.LoadWeights(vw);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);

            _output.WriteLine("[3/5] Loading pre-computed text embeddings...");
            Tensor qwen = LoadF32Tensor(TestPaths.Kandinsky5.PromptQwenEmbeds, config.InTextDim);
            Tensor clip = LoadPooled(TestPaths.Kandinsky5.PromptClipPooled, config.InTextDim2);
            Tensor negQwen = LoadF32Tensor(TestPaths.Kandinsky5.NegPromptQwenEmbeds, config.InTextDim);
            Tensor negClip = LoadPooled(TestPaths.Kandinsky5.NegPromptClipPooled, config.InTextDim2);
            _output.WriteLine($"  qwen seq={qwen.Shape[1]}, neg seq={negQwen.Shape[1]}");

            _output.WriteLine($"[4/5] Generating {numFrames}f {width}x{height}, {steps} steps, cfg={cfg}...");
            Kandinsky5VideoPipeline pipeline = new(backend, transformer, vae, config);
            TextToImageRequest req = new()
            { Prompt = "(embeddings)", Width = width, Height = height, Steps = steps, CfgScale = cfg, Seed = 42 };
            Stopwatch gen = Stopwatch.StartNew();
            (byte[][] frames, int outW, int outH, int seed) = pipeline.GenerateFromEmbeddings(
                qwen, clip, negQwen, negClip, req, numFrames,
                pr => _output.WriteLine($"  step {pr.Step}/{pr.TotalSteps} ({pr.ElapsedMs:F0}ms)"));
            gen.Stop();
            qwen.Dispose(); clip.Dispose(); negQwen.Dispose(); negClip.Dispose();
            _output.WriteLine($"[5/5] Generated {frames.Length} frames in {gen.Elapsed.TotalSeconds:F1}s (seed={seed})");
            (long cachedBytes, long hits, long misses) = backend.GetGpuCacheStats();
            _output.WriteLine($"[gpu-cache] hits={hits} misses={misses} cached={cachedBytes / (1024 * 1024)}MB");

            string outDir = Path.Combine(Path.GetTempPath(), $"k5v_{DateTime.Now:HHmmss}");
            VideoFrame[] vf = new VideoFrame[frames.Length];
            for (int i = 0; i < frames.Length; i++) vf[i] = new VideoFrame(i, outW, outH, frames[i]);
            await new BmpSequenceEncoder().EncodeAsync(ToAsync(vf), outDir, fps: 24);
            _output.WriteLine($"Wrote frames → {outDir}");

            Assert.Equal(numFrames, frames.Length);
            AssertNotBlack(frames[frames.Length / 2]);
        }
        finally { foreach (SafeTensorsLoader l in loaders) l.Dispose(); }
    }

    private static void AssertNotBlack(byte[] frame)
    {
        long sum = 0; foreach (byte px in frame) sum += px;
        double mean = (double)sum / frame.Length;
        Assert.True(mean > 2.0 && mean < 253.0, $"Middle frame is degenerate (mean pixel {mean:F1}); expected a real image.");
    }

    private static async IAsyncEnumerable<VideoFrame> ToAsync(VideoFrame[] frames)
    { foreach (VideoFrame f in frames) { yield return f; await Task.Yield(); } }

    private static Dictionary<string, Tensor> CastBf16ToF16(Dictionary<string, Tensor> w)
    {
        foreach (string k in w.Keys.ToList())
            if (w[k].DType == DType.BF16) w[k] = w[k].CastTo(DType.F16);
        return w;
    }

    /// <summary>Raw headerless F32 <c>[seq, dim]</c> → <c>[1, seq, dim]</c>.</summary>
    private static unsafe Tensor LoadF32Tensor(string path, int embedDim)
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
    private static unsafe Tensor LoadPooled(string path, int embedDim)
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

using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Cuda;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tests.Common;
using SharpInference.Tokenizers;

namespace SharpInference.Diffusion.Tests;

/// <summary>End-to-end Lance T2I generation against a real checkpoint. Skips cleanly when the variant folder, the converted Wan2.2 VAE safetensors, the Qwen2 tokenizer, or the PTX dir are missing — none are bundled. This is the manual-validation entry point: the pipeline is structurally verified by <see cref="LanceImagePipelineTests"/>, but numeric output vs the reference is validation-pending. Set <c>LANCE_3B_DIR</c> + <c>LANCE_VAE_PATH</c>.</summary>
public class LanceGenerationTests
{
    private static string OutputDir => TestPaths.OutputDir;
    private readonly ITestOutputHelper _output;
    public LanceGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Lance3B_Gpu_T2I_512()
    {
        string dir = TestPaths.Lance.Dir;
        string vaePath = TestPaths.Lance.VaePath;
        if (!Directory.Exists(dir))
        {
            _output.WriteLine($"SKIPPED: Lance variant folder not found: {dir}");
            _output.WriteLine("  Download bytedance-research/Lance (Lance_3B/), convert Wan2.2_VAE.pth → safetensors, set LANCE_3B_DIR + LANCE_VAE_PATH.");
            return;
        }
        if (!File.Exists(vaePath))
        {
            _output.WriteLine($"SKIPPED: Wan2.2 VAE safetensors not found: {vaePath} (convert Wan2.2_VAE.pth first).");
            return;
        }
        string vocab = TestPaths.Tokenizers.Qwen3Vocab, merges = TestPaths.Tokenizers.Qwen3Merges;
        if (!File.Exists(vocab) || !File.Exists(merges))
        {
            _output.WriteLine($"SKIPPED: Qwen tokenizer not found ({vocab} / {merges}).");
            return;
        }
        string assemblyDir = Path.GetDirectoryName(typeof(LanceGenerationTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        List<SafeTensorsLoader> loaders = [];
        try
        {
            _output.WriteLine("[1/4] Loading + bucketing model.safetensors...");
            (LanceCheckpointConverter.ConvertedWeights conv, IReadOnlyList<SafeTensorsLoader> tl) =
                LanceCheckpointConverter.LoadAndConvert(dir);
            loaders.AddRange(tl);
            (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) =
                LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vl);

            _output.WriteLine("[2/4] Building transformer + VAE...");
            LanceConfig cfg = LanceConfig.Image;
            using LanceTransformer transformer = new(cfg);
            transformer.LoadWeights(conv.Transformer);
            Wan22VaeDecoder vae = new();
            vae.LoadWeights(vaeW);

            _output.WriteLine("[3/4] CUDA backend (VRAM probe)...");
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            (nuint freeBytes, nuint totalBytes) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            const double MinGb = 16.0;
            if (freeGb < MinGb)
            {
                _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM; Lance_3B needs ≥{MinGb} GB at FP16. The pipeline is ready; run on a larger card.");
                return;
            }

            using Qwen3Tokenizer tokenizer = new(vocab, merges, maxLength: 512);
            int[] prompt = tokenizer.EncodeChat("A photograph of an astronaut riding a horse on the moon");
            int[] neg = tokenizer.EncodeChat("");

            LanceImagePipeline pipeline = new(backend, transformer, vae, cfg);
            TextToImageRequest req = new()
            {
                Prompt = "astronaut", NegativePrompt = "", Width = 512, Height = 512,
                Steps = cfg.NumTimesteps, CfgScale = cfg.CfgTextScale, Seed = 42,
            };

            _output.WriteLine("[4/4] Generating 512x512 (NUMERIC OUTPUT VALIDATION-PENDING)...");
            (byte[] rgb, int w, int h, int seed) = pipeline.GenerateFromTokens(prompt, neg, req,
                p => _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));

            Assert.Equal(512 * 512 * 3, rgb.Length);
            Directory.CreateDirectory(OutputDir);
            string outPath = Path.Combine(OutputDir, $"lance_3b_t2i_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outPath, rgb, w, h);
            _output.WriteLine($"Saved: {outPath} ({sw.Elapsed.TotalSeconds:F1}s, seed={seed})");
        }
        finally
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
        }
    }
}

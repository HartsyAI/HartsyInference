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
using HartsyInference.Tokenizers;
using HartsyInference.Video;
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>End-to-end Lance T2V generation against a real checkpoint, streamed to a frame sequence. Skips cleanly when <c>LANCE_3B_VIDEO_DIR</c> / <c>LANCE_VAE_PATH</c> / Qwen tokenizer / PTX dir are missing, or VRAM is insufficient. The manual-validation entry point — numerics are validation-pending (the pipeline is structurally verified by <see cref="LanceVideoPipelineTests"/>).</summary>
[Trait("Category", "Integration")]
public class LanceVideoGenerationTests
{
    private readonly ITestOutputHelper _output;
    public LanceVideoGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task Lance3BVideo_Gpu_T2V_480p_ShortClip()
    {
        string dir = TestPaths.Lance.VideoDir, vaePath = TestPaths.Lance.VaePath;
        if (!Directory.Exists(dir)) { _output.WriteLine($"SKIPPED: Lance video folder not found: {dir}"); return; }
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: Wan2.2 VAE safetensors not found: {vaePath}."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(LanceVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        List<SafeTensorsLoader> loaders = [];
        try
        {
            (LanceCheckpointConverter.ConvertedWeights conv, IReadOnlyList<SafeTensorsLoader> tl) = LanceCheckpointConverter.LoadAndConvert(dir);
            loaders.AddRange(tl);
            (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vl) = LanceCheckpointConverter.LoadVae(vaePath);
            loaders.AddRange(vl);

            LanceConfig cfg = LanceConfig.Video;
            using LanceTransformer transformer = new(cfg);
            transformer.LoadWeights(conv.Transformer);
            Wan22VaeDecoder vae = new();
            vae.LoadWeights(vaeW);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            (nuint freeBytes, _) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            const double MinGb = 24.0;
            if (freeGb < MinGb) { _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM; Lance video needs ≥{MinGb} GB."); return; }

            string tokenizerJson = Path.Combine(dir, "tokenizer.json");
            if (!File.Exists(tokenizerJson)) { _output.WriteLine($"SKIPPED: tokenizer.json not found in {dir}."); return; }
            using FileStream tokFs = File.OpenRead(tokenizerJson);
            GgufTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokFs);
            HartsyInference.Diffusion.Pipelines.LancePromptTemplate template =
                HartsyInference.Diffusion.Pipelines.LancePromptTemplate.Create(tokenizer.EncodeOrdinary, cfg, video: true);
            int[] prompt = tokenizer.EncodeOrdinary("a cinematic shot of a cat walking through a garden");
            int[] neg = [];

            LanceVideoPipeline pipeline = new(backend, transformer, vae, cfg, template);
            TextToImageRequest req = new() { Prompt = "cat", NegativePrompt = "", Width = 512, Height = 512, Steps = cfg.NumTimesteps, CfgScale = cfg.CfgTextScale, Seed = 42 };

            string outDir = Path.Combine(TestPaths.OutputDir, $"lance_video_{DateTime.Now:yyyyMMdd_HHmmss}");
            _output.WriteLine("Generating 9-frame 512x512 clip (NUMERIC OUTPUT VALIDATION-PENDING)...");
            await new BmpSequenceEncoder().EncodeAsync(
                pipeline.GenerateFramesAsync(prompt, neg, req, numFrames: 9,
                    p => _output.WriteLine($"  step {p.Step}/{p.TotalSteps}")),
                outDir, fps: 16);
            _output.WriteLine($"Wrote frames → {outDir}");
            Assert.True(Directory.GetFiles(outDir, "frame_*.bmp").Length == 9);
        }
        finally
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
        }
    }
}

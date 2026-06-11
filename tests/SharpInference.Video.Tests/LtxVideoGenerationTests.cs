using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Cuda;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tests.Common;
using SharpInference.Tokenizers;
using SharpInference.Video.Encoding;
using SharpInference.Video.Pipelines;

namespace SharpInference.Video.Tests;

/// <summary>End-to-end LTX-Video T2V generation against the <c>Lightricks/LTX-Video</c> 0.9 single file (bundled
/// DiT + VAE): T5-XXL encode (extracted from the SD3.5 bundle, like Chroma) → <see cref="LtxVideoPipeline"/> →
/// BMP frame sequence. Skips cleanly when artifacts are missing or VRAM is insufficient. The manual first-run
/// validation entry — numerics are validation-pending (structure is verified by <see cref="LtxVideoPipelineTests"/>).</summary>
public class LtxVideoGenerationTests
{
    private readonly ITestOutputHelper _output;
    public LtxVideoGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task LtxVideo09_Gpu_T2V_480p_ShortClip()
    {
        string ckpt = TestPaths.LtxVideo.SingleFile;
        string t5Source = TestPaths.LtxVideo.T5XxlSource, spiece = TestPaths.LtxVideo.T5XxlSpiece;
        if (!File.Exists(ckpt)) { _output.WriteLine($"SKIPPED: LTX-Video checkpoint not found: {ckpt} (set LTX_VIDEO_PATH)."); return; }
        if (!File.Exists(t5Source)) { _output.WriteLine($"SKIPPED: T5-XXL source not found: {t5Source} (set LTX_T5XXL_SOURCE; default is the SD3.5 bundle)."); return; }
        if (!File.Exists(spiece)) { _output.WriteLine($"SKIPPED: T5-XXL SentencePiece not found: {spiece}."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(LtxVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/5] Loading + converting LTX single file: {Path.GetFileName(ckpt)}");
        (LtxVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ckptLoader) =
            LtxVideoCheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _ckpt = ckptLoader;
        _output.WriteLine($"  DiT {conv.Transformer.Count} keys, VAE {conv.Vae.Count} keys in {sw.ElapsedMilliseconds}ms");

        _output.WriteLine($"[2/5] Extracting T5-XXL from {Path.GetFileName(t5Source)}...");
        sw.Restart();
        (Sd3CheckpointConverter.ConvertedWeights sd3Bundle, SafeTensorsLoader t5Loader) =
            Sd3CheckpointConverter.LoadAndConvert(t5Source);
        try
        {
            if (sd3Bundle.T5.Count == 0)
            { _output.WriteLine($"SKIPPED: no T5 weights bundled in {Path.GetFileName(t5Source)}."); return; }
            _output.WriteLine($"  {sd3Bundle.T5.Count} T5 tensors in {sw.ElapsedMilliseconds}ms");

            LtxVideoConfig cfg = LtxVideoConfig.V09;
            using LtxVideoTransformer transformer = new(cfg);
            transformer.LoadWeights(conv.Transformer);
            LtxVideoVaeDecoder vae = new();
            vae.LoadWeights(CastWeightsToF32(conv.Vae));
            using T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(sd3Bundle.T5);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            (nuint freeBytes, _) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            const double MinGb = 10.0;
            if (freeGb < MinGb) { _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM; LTX-Video needs ≥{MinGb} GB."); return; }

            _output.WriteLine($"[3/5] T5 encode...");
            sw.Restart();
            using T5Tokenizer tokenizer = new(spiece, maxLength: 128);
            int[] promptTokens = tokenizer.Encode("a cinematic shot of a cat walking through a sunlit garden, shallow depth of field");
            int[] negTokens = tokenizer.Encode("blurry, low quality, distorted, watermark");
            Tensor batch = t5.Encode(backend,
                [promptTokens, negTokens],
                [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
            Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, promptTokens.Length, 4096);
            Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, negTokens.Length, 4096);
            batch.Dispose();
            // Reclaim T5 VRAM before the DiT preload (the pipeline preloads the transformer itself).
            backend.Sync();
            backend.FreeWeights(t5.EnumerateWeights());
            _output.WriteLine($"  encoded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine($"[4/5] Generating 25-frame 704x480 clip (NUMERIC OUTPUT VALIDATION-PENDING)...");
            LtxVideoPipeline pipeline = new(backend, transformer, vae, cfg);
            TextToImageRequest req = new() { Prompt = "cat", Width = 704, Height = 480, Steps = 30, CfgScale = cfg.GuidanceScale, Seed = 42 };
            string outDir = Path.Combine(TestPaths.OutputDir, $"ltx_video_{DateTime.Now:yyyyMMdd_HHmmss}");
            await new BmpSequenceEncoder().EncodeAsync(
                pipeline.GenerateFramesAsync(promptEmbeds, negEmbeds, req, numFrames: 25, frameRate: 25,
                    p => _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)")),
                outDir, fps: 25);
            promptEmbeds.Dispose();
            negEmbeds.Dispose();

            _output.WriteLine($"[5/5] Wrote frames → {outDir}");
            Assert.Equal(25, Directory.GetFiles(outDir, "frame_*.bmp").Length);
        }
        finally
        {
            t5Loader.Dispose();
        }
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

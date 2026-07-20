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
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.TextEncoders;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>End-to-end LTX-Video T2V generation against the <c>Lightricks/LTX-Video</c> 0.9 single file (bundled
/// DiT + VAE): T5-XXL encode (extracted from the SD3.5 bundle, like Chroma) → <see cref="LtxVideoPipeline"/> →
/// BMP frame sequence. Skips cleanly when artifacts are missing or VRAM is insufficient. The manual first-run
/// validation entry — numerics are validation-pending (structure is verified by <see cref="LtxVideoPipelineTests"/>).</summary>
[Trait("Category", "Integration")]
public class LtxVideoGenerationTests
{
    private readonly ITestOutputHelper _output;
    public LtxVideoGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task LtxVideo09_Gpu_T2V_480p_ShortClip()
    {
        string ckpt = TestPaths.LtxVideo.SingleFile;
        string t5Standalone = TestPaths.LtxVideo.T5XxlStandalone, spiece = TestPaths.LtxVideo.T5XxlSpiece;
        if (!File.Exists(ckpt)) { _output.WriteLine($"SKIPPED: LTX-Video checkpoint not found: {ckpt} (set LTX_VIDEO_PATH)."); return; }
        if (!File.Exists(t5Standalone)) { _output.WriteLine($"SKIPPED: standalone T5-XXL not found: {t5Standalone} (set LTX_T5XXL_STANDALONE)."); return; }
        if (!File.Exists(spiece)) { _output.WriteLine($"SKIPPED: T5-XXL SentencePiece not found: {spiece}."); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(LtxVideoGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/5] Loading + converting LTX single file: {Path.GetFileName(ckpt)}");
        (LtxVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ckptLoader) =
            LtxVideoCheckpointConverter.LoadAndConvert(ckpt);
        using SafeTensorsLoader _ckpt = ckptLoader;
        _output.WriteLine($"  DiT {conv.Transformer.Count} keys, VAE {conv.Vae.Count} keys in {sw.ElapsedMilliseconds}ms");

        _output.WriteLine($"[2/5] Loading standalone T5-XXL from {Path.GetFileName(t5Standalone)}...");
        sw.Restart();
        using SafeTensorsLoader t5Loader = new();
        t5Loader.Load(t5Standalone);
        try
        {
            Dictionary<string, Tensor> t5Weights = TextEncoderQuantNormalizer.Normalize(t5Loader.GetAllTensors());
            if (t5Weights.Count == 0)
            { _output.WriteLine($"SKIPPED: no T5 weights in {Path.GetFileName(t5Standalone)}."); return; }
            _output.WriteLine($"  {t5Weights.Count} T5 tensors in {sw.ElapsedMilliseconds}ms");

            // Variant selection (filename or LTX_VARIANT override): 13B/0.9.7 = V097 transformer (48 layers, head_dim
            // 128, cross 4096); 0.9.5 = V09 transformer. Both 0.9.5 and 13B share the timestep-conditioned VAE
            // (residual channel-changing upsamplers, decoder_block_out_channels (256,512,1024), 5 layers/block). 0.9 is
            // the base non-timestep VAE.
            string variant = Environment.GetEnvironmentVariable("LTX_VARIANT")
                ?? (ckpt.Contains("13b", StringComparison.OrdinalIgnoreCase) || ckpt.Contains("0.9.7", StringComparison.Ordinal) ? "0.9.7"
                    : ckpt.Contains("0.9.5", StringComparison.Ordinal) ? "0.9.5" : "0.9");
            bool timestepVae = variant is "0.9.5" or "0.9.7";
            LtxVideoConfig cfg = variant switch { "0.9.7" => LtxVideoConfig.V097, "0.9.5" => LtxVideoConfig.V095, _ => LtxVideoConfig.V09 };
            using LtxVideoTransformer transformer = new(cfg);
            transformer.LoadWeights(conv.Transformer);
            LtxVideoVaeDecoder vae = timestepVae
                ? new LtxVideoVaeDecoder(blockOutChannels: [256, 512, 1024], spatioTemporalScaling: [true, true, true],
                    layersPerBlock: [5, 5, 5, 5], patchSize: 4, isCausal: false, timestepConditioned: true,
                    upsampleFactor: [2, 2, 2], upsampleResidual: [true, true, true])
                : new LtxVideoVaeDecoder();
            vae.LoadWeights(CastWeightsToF32(conv.Vae));
            _output.WriteLine($"  variant: {variant} ({(timestepVae ? "timestep" : "base")} VAE)");
            using T5TextEncoder t5 = new(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(t5Weights);

            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            // 13B fp8 weights stay fp8-resident (~13 GB); caching their F16 casts would add ~26 GB → OOM on 24 GB.
            // Dequant transiently per GEMM instead (matches the Wan fp8 recipe).
            if (variant == "0.9.7") backend.CacheWeightCasts = false;
            (nuint freeBytes, _) = backend.Context.GetMemoryInfo();
            double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
            const double MinGb = 10.0;
            if (freeGb < MinGb) { _output.WriteLine($"SKIPPED: only {freeGb:F1} GB free VRAM; LTX-Video needs ≥{MinGb} GB."); return; }

            _output.WriteLine($"[3/5] T5 encode...");
            sw.Restart();
            using T5Tokenizer tokenizer = new(spiece, maxLength: 128);
            int[] promptTokens = tokenizer.Encode("a cinematic shot of a cat walking through a sunlit garden, shallow depth of field");
            int[] negTokens = tokenizer.Encode("blurry, low quality, distorted, watermark");
            int[] promptMask = T5Tokenizer.CreateAttentionMask(promptTokens);
            int[] negMask = T5Tokenizer.CreateAttentionMask(negTokens);
            Tensor batch = t5.Encode(backend, [promptTokens, negTokens], [promptMask, negMask]);
            // Drop right-padding: feed cross-attention only the real (non-pad) T5 tokens. Attending the ~120 pad rows
            // unmasked dilutes the caption (LTX/diffusers mask them); truncation is the equivalent for a prefix-padded seq.
            int promptLen = promptMask.Sum(), negLen = negMask.Sum();
            Tensor promptEmbeds = CfgHelper.SliceBatchElementPrefix(batch, 0, promptTokens.Length, promptLen, 4096);
            Tensor negEmbeds = CfgHelper.SliceBatchElementPrefix(batch, 1, negTokens.Length, negLen, 4096);
            _output.WriteLine($"  real tokens: prompt {promptLen}/{promptTokens.Length}, neg {negLen}/{negTokens.Length}");
            batch.Dispose();
            // Reclaim T5 VRAM before the DiT preload (the pipeline preloads the transformer itself).
            backend.Sync();
            backend.FreeWeights(t5.EnumerateWeights());
            _output.WriteLine($"  encoded in {sw.ElapsedMilliseconds}ms");

            _output.WriteLine($"[4/5] Generating 25-frame 704x480 clip (NUMERIC OUTPUT VALIDATION-PENDING)...");
            LtxVideoPipeline pipeline = new(backend, transformer, vae, cfg);
            int steps = int.TryParse(Environment.GetEnvironmentVariable("LTX_STEPS"), out int st) ? st : 30;
            TextToImageRequest req = new() { Prompt = "cat", Width = 704, Height = 480, Steps = steps, CfgScale = cfg.GuidanceScale, Seed = 42 };
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
            // t5Loader is a `using` declaration — disposed at scope exit.
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

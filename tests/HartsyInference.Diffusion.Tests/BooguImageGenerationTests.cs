using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>End-to-end Boogu-Image-0.1 text-to-image. Loads the fp8_scaled transformer + full Qwen3-VL-8B language tower
/// (text encoder) + the FLUX.1 VAE from <c>Boogu/{Base,Turbo}/</c> via <see cref="BooguImageCheckpointConverter"/>.
///
/// <para>Conditioning (per <c>docs/Research/BOOGU_IMAGE.md §5</c>): the instruction is chat-templated with Boogu's T2I
/// system prompt and encoded through Qwen3-VL-8B; <c>num_instruction_feature_layers=1, reduce_type=mean</c> ⇒ the
/// conditioning is simply the <b>final decoder hidden state</b> (<c>[1, T, 4096]</c>) — NO system-prefix drop (unlike
/// Qwen-Image / Krea 2). The negative stream is the empty string under the same template, used for single-CFG
/// (<c>text_guidance_scale</c>). The 10.6 GB TE is freed right after encode so the 10.3 GB fp8 transformer fits the
/// 4090 alongside the VAE. Skips cleanly when weights / PTX / VRAM are absent.</para></summary>
public sealed class BooguImageGenerationTests
{
    private readonly ITestOutputHelper _output;
    public BooguImageGenerationTests(ITestOutputHelper output) => _output = output;

    // Verbatim from pipeline_boogu.py (BOOGU_IMAGE.md §5).
    private const string T2ISystem =
        "You are a helpful assistant that generates high-quality images based on user instructions. " +
        "The instructions are as follows.";

    [Fact]
    public void BooguImage_Base_Gpu_1024_Cfg() =>
        RunGenerationTest(TestPaths.Boogu.BaseDir, "boogu_base_1024", steps: 28, textGuidance: 4.0f);

    [Fact]
    public void BooguImage_Turbo_Gpu_1024_NoCfg() =>
        RunGenerationTest(TestPaths.Boogu.TurboDir, "boogu_turbo_1024", steps: 4, textGuidance: 1.0f);

    private void RunGenerationTest(string rootDir, string outputName, int steps, float textGuidance)
    {
        if (!Directory.Exists(rootDir)) { _output.WriteLine($"SKIPPED: Boogu dir not found: {rootDir}"); return; }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(BooguImageGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX dir not found: {ptxDir}"); return; }

        Stopwatch sw = Stopwatch.StartNew();
        _output.WriteLine($"[1/6] Loading + converting Boogu from {rootDir}...");
        (Dictionary<string, Tensor> txWeights, IReadOnlyList<SafeTensorsLoader> txLoaders) = BooguImageCheckpointConverter.LoadTransformer(rootDir);
        (Dictionary<string, Tensor> teWeights, IReadOnlyList<SafeTensorsLoader> teLoaders) = BooguImageCheckpointConverter.LoadTextEncoder(rootDir);
        (Dictionary<string, Tensor> vaeWeights, IReadOnlyList<SafeTensorsLoader> vaeLoaders) = BooguImageCheckpointConverter.LoadVae(rootDir);
        _output.WriteLine($"  transformer={txWeights.Count} te={teWeights.Count} vae={vaeWeights.Count} keys");

        try
        {
            _output.WriteLine("[2/6] Building transformer + TE (Qwen3-VL-8B) + VAE (FLUX.1)...");
            BooguImageConfig config = BooguImageConfig.V01;
            BooguImageTransformer transformer = new(config);
            transformer.LoadWeights(txWeights);
            LlamaStyleEncoder textEncoder = new(LlamaStyleEncoderConfig.Qwen3_VL_8B);
            textEncoder.LoadWeights(teWeights);
            VaeDecoder vae = new(VaeConfig.Flux);
            vae.LoadWeights(CastF32(vaeWeights));

            _output.WriteLine("[3/6] Tokenizing (Qwen3-VL chat template, Boogu T2I system, NO drop)...");
            using Qwen2Tokenizer tokenizer = new();
            string prompt = "A photograph of an astronaut riding a horse";
            int[] promptTokens = tokenizer.EncodeChat(prompt, systemPrompt: T2ISystem, addGenerationPrompt: true);
            int[] negTokens = tokenizer.EncodeChat("", systemPrompt: T2ISystem, addGenerationPrompt: true);
            _output.WriteLine($"  prompt tokens={promptTokens.Length} neg tokens={negTokens.Length}");

            _output.WriteLine("[4/6] CUDA backend...");
            using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            backend.CacheWeightCasts = false;   // fp8 (~10.3 GB transformer + 10.6 GB TE) → transient dequant
            (nuint freeB, nuint totalB) = backend.Context.GetMemoryInfo();
            double freeGb = freeB / (1024.0 * 1024.0 * 1024.0);
            if (freeGb < 16.0) { _output.WriteLine($"SKIPPED: {freeGb:F1} GB free; need ≥16 GB."); transformer.Dispose(); textEncoder.Dispose(); return; }

            // ── Encode the instruction (final hidden state, [1, T, 4096]); free the TE before the transformer runs. ──
            _output.WriteLine("[5/6] Encoding instruction via Qwen3-VL-8B (final hidden state, mean over 1 layer)...");
            Stopwatch encSw = Stopwatch.StartNew();
            backend.PreloadWeights(textEncoder.EnumerateWeights());
            Tensor instr = textEncoder.Encode(backend, [promptTokens]);
            Tensor? negInstr = textGuidance > 1.0f ? textEncoder.Encode(backend, [negTokens]) : null;
            backend.FreeWeights(textEncoder.EnumerateWeights());
            backend.Sync();
            textEncoder.Dispose();   // free the 10.6 GB TE host weights too
            foreach (SafeTensorsLoader l in teLoaders) l.Dispose();
            encSw.Stop();
            _output.WriteLine($"  Encoded instr={instr.Shape} neg={(negInstr is null ? "(none)" : negInstr.Shape.ToString())} in {encSw.Elapsed.TotalSeconds:F1}s");

            using BooguImagePipeline pipeline = new(backend, transformer, vae, vaeEncoder: null, config);
            TextToImageRequest request = new() { Prompt = prompt, Width = 1024, Height = 1024, Steps = steps, Seed = 42 };

            _output.WriteLine($"[6/6] Generating 1024x1024, {steps} steps, tg={textGuidance}...");
            Stopwatch gen = Stopwatch.StartNew();
            (byte[] rgb, int w, int h, int seed) = pipeline.GenerateFromEmbeddings(instr, request, textGuidance, negInstr,
                p => _output.WriteLine($"  Step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
            gen.Stop();
            _output.WriteLine($"  Generated in {gen.Elapsed.TotalSeconds:F1}s (seed={seed})");

            instr.Dispose();
            negInstr?.Dispose();

            Assert.Equal(w * h * 3, rgb.Length);
            int nonZero = rgb.Count(b => b != 0), nonFF = rgb.Count(b => b != 255);
            _output.WriteLine($"  Non-zero {nonZero / (float)rgb.Length * 100:F1}%, Non-255 {nonFF / (float)rgb.Length * 100:F1}%");
            Assert.True(nonZero > rgb.Length * 0.1, "all black");
            Assert.True(nonFF > rgb.Length * 0.1, "all white");

            string outDir = TestPaths.OutputDir; Directory.CreateDirectory(outDir);
            string outPath = Path.Combine(outDir, $"{outputName}_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outPath, rgb, w, h);
            _output.WriteLine($"  Saved: {outPath} ({sw.Elapsed.TotalSeconds:F1}s total)");
            transformer.Dispose();
        }
        finally
        {
            foreach (SafeTensorsLoader l in txLoaders) l.Dispose();
            foreach (SafeTensorsLoader l in vaeLoaders) l.Dispose();
        }
    }

    private static Dictionary<string, Tensor> CastF32(Dictionary<string, Tensor> w)
    {
        Dictionary<string, Tensor> o = new(w.Count);
        foreach (KeyValuePair<string, Tensor> kv in w)
            o[kv.Key] = (kv.Value.DType == DType.F16 || kv.Value.DType == DType.BF16) ? kv.Value.CastTo(DType.F32) : kv.Value;
        return o;
    }
}

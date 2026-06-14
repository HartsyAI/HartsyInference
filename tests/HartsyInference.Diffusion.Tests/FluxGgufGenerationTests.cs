using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.Gguf;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>End-to-end Flux Schnell generation through a Q4_K_S GGUF transformer (`city96/FLUX.1-schnell-gguf/flux1-schnell-Q4_K_S.gguf`, 6.78 GB on disk vs 23.8 GB FP16). Uses the Phase A-D codec stack: GGUF reader → key mapper → BFL→diffusers convert (with quant-aware QKV split) → FluxTransformer.LoadWeights → CudaBackend.Linear (with on-device PTX dequant for Q4_K weights at GEMM time).
///
/// <para><b>Memory profile</b>: peak ~20 GB anon-RSS during the BFL→diffusers QKV split + FP8 T5/CLIP/VAE load. Won't fit a 32 GB system if the calling environment has a tight cgroup limit (e.g. when dotnet inherits the browser's session memcg). <see cref="SkipWhenMemoryConstrained"/> probes for this and skips early with a clear message.</para>
///
/// <para>The Phase D wiring (CudaBackend.Linear handling quantized weights via GPU dequant kernels) is exercised by <see cref="FluxGgufLinearTests.Linear_RealQ4_K_FromCity96Gguf_ProducesSaneOutput"/> against real city96 data — that test runs cleanly in any environment.</para></summary>
public sealed class FluxGgufGenerationTests
{
    private const long MinAnonRamRequiredGb = 25;

    private readonly ITestOutputHelper _output;
    public FluxGgufGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void Schnell_FromGguf_GeneratesImage()
    {
        if (!File.Exists(TestPaths.Flux.SchnellQ4KS))
        {
            _output.WriteLine($"SKIPPED: GGUF not found at {TestPaths.Flux.SchnellQ4KS}.");
            _output.WriteLine($"  curl -L -o {TestPaths.Flux.SchnellQ4KS} 'https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q4_K_S.gguf?download=true'");
            return;
        }
        if (!File.Exists(TestPaths.Flux.Schnell))
        {
            _output.WriteLine($"SKIPPED: Flux Schnell FP8 single-file not found at {TestPaths.Flux.Schnell} (needed for T5/CLIP/VAE).");
            return;
        }
        if (!File.Exists(TestPaths.Tokenizers.ClipVocab) || !File.Exists(TestPaths.Tokenizers.ClipMerges))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found.");
            return;
        }
        if (!File.Exists(TestPaths.Tokenizers.T5Spiece))
        {
            _output.WriteLine("SKIPPED: T5 SentencePiece model not found.");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable.");
            return;
        }
        if (SkipWhenMemoryConstrained(out string memReason))
        {
            _output.WriteLine($"SKIPPED: {memReason}");
            return;
        }

        Stopwatch totalSw = Stopwatch.StartNew();

        _output.WriteLine($"[1/8] Loading Flux Schnell Q4_K_S GGUF (LAZY): {Path.GetFileName(TestPaths.Flux.SchnellQ4KS)}");
        Stopwatch sw = Stopwatch.StartNew();
        GgufModelLoader.LoadedGgufModel ggufHandle = GgufModelLoader.Load(TestPaths.Flux.SchnellQ4KS);
        sw.Stop();
        _output.WriteLine($"  GGUF mmap loaded in {sw.ElapsedMilliseconds}ms ({ggufHandle.Weights.Count} tensors).");

        sw.Restart();
        Dictionary<string, Tensor> ggufRawDict = ggufHandle.Weights.ToDictionary(kv => kv.Key, kv => kv.Value);
        FluxCheckpointConverter.ConvertedWeights ggufConverted = FluxCheckpointConverter.Convert(ggufRawDict);
        sw.Stop();
        _output.WriteLine($"  BFL→diffusers + QKV split in {sw.ElapsedMilliseconds}ms — transformer keys: {ggufConverted.Transformer.Count}");
        Assert.NotEmpty(ggufConverted.Transformer);

        _output.WriteLine($"[2/8] Loading Flux Schnell FP8 single-file for T5/CLIP/VAE: {Path.GetFileName(TestPaths.Flux.Schnell)}");
        sw.Restart();
        (FluxCheckpointConverter.ConvertedWeights fp8Converted, SafeTensorsLoader fp8Loader) =
            FluxCheckpointConverter.LoadAndConvert(TestPaths.Flux.Schnell);
        sw.Stop();
        _output.WriteLine($"  FP8 file loaded in {sw.ElapsedMilliseconds}ms");

        try
        {
            using (ggufHandle)
            using (fp8Loader)
            {
                Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(fp8Converted.Vae);

                string assemblyDir = Path.GetDirectoryName(typeof(FluxGgufGenerationTests).Assembly.Location)!;
                string ptxDir = Path.Combine(assemblyDir, "Ptx");
                using CudaBackend backend = Directory.Exists(ptxDir)
                    ? new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir)
                    : new CudaBackend();

                FluxConfig config = FluxConfig.Schnell;

                _output.WriteLine("[3/8] Loading FluxTransformer from GGUF-derived weights (quant tensors stay quantized)...");
                sw.Restart();
                FluxTransformer transformer = new FluxTransformer(config);
                transformer.LoadWeights(ggufConverted.Transformer);
                sw.Stop();
                _output.WriteLine($"  Transformer loaded in {sw.ElapsedMilliseconds}ms");

                _output.WriteLine("[4/8] Loading CLIP-L (FP8 single-file)...");
                ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
                clipL.LoadWeights(fp8Converted.ClipL, "text_model");

                _output.WriteLine("[5/8] Loading T5-XXL (FP8 native)...");
                T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
                t5.LoadWeights(fp8Converted.T5);

                _output.WriteLine("[6/8] Loading VAE...");
                VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Flux);
                vaeDecoder.LoadWeights(vaeF32);

                _output.WriteLine("[7/8] Tokenize prompt...");
                using ClipTokenizer clipTokenizer = new ClipTokenizer(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
                string prompt = "A photograph of an astronaut riding a horse";
                int[] clipIds = clipTokenizer.Encode(prompt);
                int eosPos = ClipTokenizer.FindEosPosition(clipIds);
                using T5Tokenizer t5Tok = new T5Tokenizer(TestPaths.Tokenizers.T5Spiece, maxLength: 256);
                int[] t5Ids = t5Tok.Encode(prompt);
                int[] t5Mask = T5Tokenizer.CreateAttentionMask(t5Ids);

                TextToImageRequest req = new TextToImageRequest
                {
                    Prompt = prompt,
                    Width = 512,
                    Height = 512,
                    Steps = 4,
                    Seed = 42,
                };

                _output.WriteLine("[8/8] Generate (512×512, 4 steps, GGUF transformer with on-device dequant)...");
                FluxPipeline pipeline = new FluxPipeline(backend, clipL, t5, transformer, vaeDecoder, config);
                sw.Restart();
                (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                    clipIds, eosPos, t5Ids, t5Mask, req, guidanceScale: 0.0f,
                    onProgress: p => _output.WriteLine($"  step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
                sw.Stop();
                backend.Sync();
                _output.WriteLine($"Generation done in {sw.ElapsedMilliseconds}ms (seed={seed})");

                Assert.Equal(512, width);
                Assert.Equal(512, height);
                ValidateImageNotDegenerate(rgbData);

                Directory.CreateDirectory(TestPaths.OutputDir);
                string outputPath = Path.Combine(TestPaths.OutputDir, $"flux_schnell_q4_k_s_gguf_{width}x{height}_seed{seed}.bmp");
                ImagePostProcessor.SaveBmp(outputPath, rgbData, width, height);
                _output.WriteLine($"Saved generated image: {outputPath}");

                pipeline.Dispose();
                transformer.Dispose();
                t5.Dispose();
            }
        }
        finally
        {
            totalSw.Stop();
            _output.WriteLine($"Total elapsed: {totalSw.ElapsedMilliseconds}ms");
        }
    }

    /// <summary>Probes /proc/meminfo and skips when free anon-RAM is below the threshold this test needs (Q4_K Flux Schnell + FP8 T5/CLIP/VAE peak ~20 GB during the QKV split + GGUF mmap residency).</summary>
    private static bool SkipWhenMemoryConstrained(out string reason)
    {
        try
        {
            string meminfo = File.ReadAllText("/proc/meminfo");
            long memAvailableKb = 0;
            foreach (string line in meminfo.Split('\n'))
            {
                if (line.StartsWith("MemAvailable:", StringComparison.Ordinal))
                {
                    string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2) memAvailableKb = long.Parse(parts[1]);
                    break;
                }
            }
            long memAvailableGb = memAvailableKb / (1024 * 1024);
            if (memAvailableGb < MinAnonRamRequiredGb)
            {
                reason = $"Only {memAvailableGb} GB available; need ≥{MinAnonRamRequiredGb} GB to fit Q4_K Flux Schnell + FP8 T5/CLIP/VAE without OOM. Close other apps or run on a larger host. Phase D wiring itself is validated by FluxGgufLinearTests with real city96 Q4_K data — that test runs in any environment.";
                return true;
            }
        }
        catch (Exception ex)
        {
            reason = $"Could not probe /proc/meminfo: {ex.Message}";
            return false;
        }
        reason = "";
        return false;
    }

    private static unsafe Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> result = new();
        foreach (KeyValuePair<string, Tensor> kv in weights)
        {
            Tensor src = kv.Value;
            result[kv.Key] = src.DType == DType.F32 ? src : src.CastTo(DType.F32);
        }
        return result;
    }

    private static void ValidateImageNotDegenerate(byte[] rgb)
    {
        long sum = 0;
        long sumSq = 0;
        long n = rgb.Length;
        for (int i = 0; i < n; i++)
        {
            int v = rgb[i];
            sum += v;
            sumSq += (long)v * v;
        }
        double mean = sum / (double)n;
        double variance = (sumSq / (double)n) - (mean * mean);
        double stddev = Math.Sqrt(variance);
        Assert.True(mean > 5 && mean < 250, $"Image is mostly black or mostly white (mean={mean:F1}).");
        Assert.True(stddev > 8, $"Image has too little variance to be real content (stddev={stddev:F1}).");
    }
}

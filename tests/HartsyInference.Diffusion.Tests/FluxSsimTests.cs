using System.Diagnostics;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>SSIM acceptance gate for Flux Dev / Flux Schnell. Compares pipeline output against PNGs produced by `tests/python-reference/dump_flux_reference_image.py`. Strict gate (0.85) when the matching <c>init_noise_seed{N}.bin</c> binary is present (16-channel unpacked F32 noise feeds <see cref="TextToImageRequest.InitialNoise"/>); loose gate (0.30) fallback otherwise.
///
/// Note: Flux Schnell's strict SSIM is harder to hit than Dev's because the 4-step distilled scheduler amplifies any small numerical drift over fewer steps. The 0.85 threshold leaves headroom for that.</summary>
[Trait("Category", "Integration")]
public sealed class FluxSsimTests
{
    private const double StrictSsimThreshold = 0.85;
    private const double LooseSsimThreshold = 0.30;

    private static string ReferenceDir =>
        Environment.GetEnvironmentVariable("FLUX_REFERENCE_IMAGE_DIR")
        ?? Path.Combine(RepoRoot.Path, "tests", "python-reference", "flux_reference_images");

    private readonly ITestOutputHelper _output;
    public FluxSsimTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void FluxDev_Generates_WithinSsimThreshold() => RunVariant("dev", TestPaths.Flux.Dev, FluxConfig.Dev, numSteps: 10, cfg: 3.5f);

    [Fact]
    public void FluxSchnell_Generates_WithinSsimThreshold_FourStep() => RunVariant("schnell", TestPaths.Flux.Schnell, FluxConfig.Schnell, numSteps: 4, cfg: 0.0f);

    private void RunVariant(string variantTag, string checkpointPath, FluxConfig config, int numSteps, float cfg)
    {
        string metaPath = Path.Combine(ReferenceDir, "meta.json");
        if (!File.Exists(metaPath))
        {
            _output.WriteLine($"SKIPPED: Flux reference dir not found at {ReferenceDir}.");
            _output.WriteLine("Generate references first via: python tests/python-reference/dump_flux_reference_image.py --output <ReferenceDir>");
            return;
        }
        if (!File.Exists(checkpointPath))
        {
            _output.WriteLine($"SKIPPED: Flux {variantTag} checkpoint not found at {checkpointPath}.");
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
            _output.WriteLine("SKIPPED: CUDA not available.");
            return;
        }

        ReferenceMeta meta = JsonSerializer.Deserialize<ReferenceMeta>(File.ReadAllText(metaPath))
            ?? throw new InvalidDataException("Reference meta.json is empty or malformed.");
        ReferencePrompt? prompt = meta.Prompts?.FirstOrDefault(p => p.Variant == variantTag);
        if (prompt is null)
        {
            _output.WriteLine($"SKIPPED: no '{variantTag}' prompts in reference meta.json.");
            return;
        }
        string rgbPath = Path.Combine(ReferenceDir, prompt.Rgb);
        if (string.IsNullOrEmpty(prompt.Rgb) || !File.Exists(rgbPath))
        {
            _output.WriteLine($"SKIPPED: reference RGB binary missing for {variantTag} (looked at {rgbPath}).");
            return;
        }
        byte[] referenceRgb = File.ReadAllBytes(rgbPath);
        int expectedLen = meta.Width * meta.Height * 3;
        if (referenceRgb.Length != expectedLen)
        {
            _output.WriteLine($"SKIPPED: reference RGB length {referenceRgb.Length} != {expectedLen}.");
            return;
        }

        string ptxDir = Path.Combine(RepoRoot.Path, "native", "cuda", "build");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        (FluxCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            FluxCheckpointConverter.LoadAndConvert(checkpointPath);
        sw.Stop();
        _output.WriteLine($"Loaded {variantTag} checkpoint in {sw.ElapsedMilliseconds}ms.");

        using (loader)
        using (CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir))
        {
            FluxTransformer transformer = new FluxTransformer(config);
            transformer.LoadWeights(converted.Transformer);

            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(converted.ClipL, "text_model");

            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(converted.T5);

            VaeDecoder vae = new VaeDecoder(VaeConfig.Flux);
            vae.LoadWeights(converted.Vae);

            using ClipTokenizer clipTok = new ClipTokenizer(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
            using T5Tokenizer t5Tok = new T5Tokenizer(TestPaths.Tokenizers.T5Spiece, maxLength: 256);

            int[] clipIds = clipTok.Encode(prompt.Prompt);
            int eosL = ClipTokenizer.FindEosPosition(clipIds);
            int[] t5Ids = t5Tok.Encode(prompt.Prompt);
            int[] t5Mask = T5Tokenizer.CreateAttentionMask(t5Ids);

            string noisePath = Path.Combine(ReferenceDir, $"init_noise_seed{meta.Seed}.bin");
            Tensor? injectedNoise = TryLoadFluxNoise(noisePath, meta.Width, meta.Height);
            double threshold = injectedNoise is not null ? StrictSsimThreshold : LooseSsimThreshold;
            _output.WriteLine(injectedNoise is not null
                ? $"Loaded reference noise from {Path.GetFileName(noisePath)}; using strict SSIM gate {StrictSsimThreshold}."
                : $"No reference noise binary at {noisePath}; falling back to loose SSIM gate {LooseSsimThreshold}.");

            TextToImageRequest req = new TextToImageRequest
            {
                Prompt = prompt.Prompt,
                Width = meta.Width,
                Height = meta.Height,
                Steps = numSteps,
                Seed = meta.Seed,
                InitialNoise = injectedNoise,
            };

            FluxPipeline pipeline = new FluxPipeline(backend, clipL, t5, transformer, vae, config);

            sw.Restart();
            (byte[] rgbData, int width, int height, int seed) = pipeline.GenerateFromTokens(
                clipIds, eosL, t5Ids, t5Mask, req, guidanceScale: cfg,
                onProgress: p => _output.WriteLine($"  {variantTag} step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));
            sw.Stop();
            _output.WriteLine($"Flux {variantTag} generated in {sw.ElapsedMilliseconds}ms (seed={seed}).");

            Assert.Equal(meta.Width, width);
            Assert.Equal(meta.Height, height);

            double ssim = Ssim.Compute(referenceRgb, rgbData, width, height);
            _output.WriteLine($"Flux {variantTag} SSIM vs reference: {ssim:F4} (threshold: {threshold})");
            Assert.True(ssim > threshold, $"Flux {variantTag} SSIM {ssim:F4} below threshold {threshold}.");

            pipeline.Dispose();
        }
    }

    private static unsafe Tensor? TryLoadFluxNoise(string path, int width, int height)
    {
        if (!File.Exists(path)) return null;
        byte[] raw = File.ReadAllBytes(path);
        int latentH = height / 8;
        int latentW = width / 8;
        long expectedFloats = (long)1 * 16 * latentH * latentW;
        if (raw.Length != expectedFloats * 4) return null;
        TensorShape shape = new TensorShape(1, 16, latentH, latentW);
        Tensor noise = new Tensor(shape, DType.F32);
        Span<byte> dst = new Span<byte>((void*)noise.DataPointer, raw.Length);
        raw.AsSpan().CopyTo(dst);
        return noise;
    }

    private sealed class ReferenceMeta
    {
        public int Width { get; set; }
        public int Height { get; set; }
        public int Seed { get; set; }
        public List<ReferencePrompt>? Prompts { get; set; }
    }

    private sealed class ReferencePrompt
    {
        public int Index { get; set; }
        public string Variant { get; set; } = string.Empty;
        public string Prompt { get; set; } = string.Empty;
        public string Image { get; set; } = string.Empty;
        public string Rgb { get; set; } = string.Empty;
    }
}

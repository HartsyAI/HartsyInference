using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>
/// Wiring tests for <see cref="SdxlRefinerPipeline"/> and the cross-model refining pattern.
///
/// <list type="bullet">
/// <item>Pipeline construction with refiner config.</item>
/// <item>Strength=0 byte-identical pass-through (validates the cross-model handoff plumbing without needing real weights).</item>
/// <item>SdxlRefinerCheckpointConverter — refiner-specific 4-level UNet block layout.</item>
/// <item>Cross-model refining demo: SD1.5 → SDXL refiner using <see cref="ImagePostProcessor.RgbBytesToTensor"/> handoff.</item>
/// </list>
/// </summary>
public sealed class SdxlRefinerPipelineTests
{
    private readonly ITestOutputHelper _output;

    public SdxlRefinerPipelineTests(ITestOutputHelper output) => _output = output;

    // ── Construction & wiring ───────────────────────────────────────────

    [Fact]
    public void Constructor_AcceptsRefinerConfig()
    {
        using CpuBackend backend = new();
        ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
        UNet refinerUnet = new(UNetConfig.SdxlRefiner);
        VaeEncoder vaeEncoder = new(VaeConfig.Sdxl);
        VaeDecoder vaeDecoder = new(VaeConfig.Sdxl);

        using SdxlRefinerPipeline pipeline = new(backend, clipG, refinerUnet, vaeEncoder, vaeDecoder);
        Assert.NotNull(pipeline);
    }

    [Fact]
    public void RefineFromTokens_WrongSourceShape_Throws()
    {
        using CpuBackend backend = new();
        ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
        UNet refinerUnet = new(UNetConfig.SdxlRefiner);
        VaeEncoder vaeEncoder = new(VaeConfig.Sdxl);
        VaeDecoder vaeDecoder = new(VaeConfig.Sdxl);

        using SdxlRefinerPipeline pipeline = new(backend, clipG, refinerUnet, vaeEncoder, vaeDecoder);

        Tensor source = new Tensor(new TensorShape(1, 3, 32, 32), DType.F32);
        SdxlRefinerRequest request = new()
        {
            Prompt = "test",
            Width = 64,
            Height = 64,
            SourceImage = source,
        };

        Assert.Throws<ArgumentException>(() =>
            pipeline.RefineFromTokens(
                promptTokenIdsG: [],
                negativePromptTokenIdsG: [],
                promptEosPositionG: 0,
                negativeEosPositionG: 0,
                request));

        source.Dispose();
    }

    [Fact]
    public void RefineFromTokens_Strength0_PassesSourceThrough()
    {
        // Strength=0 short-circuits before any model code, so empty token arrays + uninitialized weights are fine.
        // This validates the pipeline plumbing without needing a real SDXL refiner checkpoint.
        using CpuBackend backend = new();
        ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
        UNet refinerUnet = new(UNetConfig.SdxlRefiner);
        VaeEncoder vaeEncoder = new(VaeConfig.Sdxl);
        VaeDecoder vaeDecoder = new(VaeConfig.Sdxl);

        using SdxlRefinerPipeline pipeline = new(backend, clipG, refinerUnet, vaeEncoder, vaeDecoder);

        const int w = 64, h = 64;
        byte[] sourceBytes = new byte[w * h * 3];
        for (int i = 0; i < sourceBytes.Length; i++) sourceBytes[i] = (byte)(i * 7 & 0xFF);
        Tensor source = ImagePostProcessor.RgbBytesToTensor(sourceBytes, w, h);

        SdxlRefinerRequest request = new()
        {
            Prompt = "ignored",
            Width = w,
            Height = h,
            Steps = 10,
            CfgScale = 7.5f,
            Seed = 42,
            SourceImage = source,
            Strength = 0.0f,
            AestheticScore = 6.0f,
            NegativeAestheticScore = 2.5f,
        };

        (byte[] outBytes, int outW, int outH, int seed) = pipeline.RefineFromTokens(
            promptTokenIdsG: [],
            negativePromptTokenIdsG: [],
            promptEosPositionG: 0,
            negativeEosPositionG: 0,
            request);

        Assert.Equal(w, outW);
        Assert.Equal(h, outH);
        Assert.Equal(42, seed);
        Assert.Equal(sourceBytes, outBytes);
        source.Dispose();
    }

    // ── Cross-model handoff plumbing test ───────────────────────────────

    [Fact]
    public void CrossModelHandoff_AnyBaseToSdxlRefiner_BytesRoundTrip()
    {
        // Simulates the cross-model refining pattern: a base pipeline produces RGB bytes,
        // those bytes get converted to a tensor and fed to the SDXL refiner's strength=0 pass-through.
        // The roundtrip (bytes → tensor → bytes via TensorToRgbBytes) must be exact.
        using CpuBackend backend = new();
        ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
        UNet refinerUnet = new(UNetConfig.SdxlRefiner);
        VaeEncoder vaeEncoder = new(VaeConfig.Sdxl);
        VaeDecoder vaeDecoder = new(VaeConfig.Sdxl);

        using SdxlRefinerPipeline refiner = new(backend, clipG, refinerUnet, vaeEncoder, vaeDecoder);

        // Imagine these came from a different base pipeline (SD1.5, Flux, Z-Image, ...).
        const int w = 128, h = 128;
        byte[] basePipelineRgb = new byte[w * h * 3];
        Random rng = new Random(42);
        rng.NextBytes(basePipelineRgb);

        // The cross-model handoff: bytes from base → tensor → refiner.RefineFromTokens
        Tensor sourceTensor = ImagePostProcessor.RgbBytesToTensor(basePipelineRgb, w, h);

        SdxlRefinerRequest request = new()
        {
            Prompt = "polish",
            Width = w,
            Height = h,
            Steps = 20,
            Seed = 0,
            SourceImage = sourceTensor,
            Strength = 0.0f,  // 0 → exact pass-through, isolates the conversion plumbing
        };

        (byte[] refinedRgb, _, _, _) = refiner.RefineFromTokens(
            promptTokenIdsG: [],
            negativePromptTokenIdsG: [],
            promptEosPositionG: 0,
            negativeEosPositionG: 0,
            request);

        // The byte→tensor→byte roundtrip is exact for in-range values.
        Assert.Equal(basePipelineRgb, refinedRgb);
        sourceTensor.Dispose();
    }

    // ── SdxlRefinerCheckpointConverter sanity ───────────────────────────

    [Fact]
    public void SdxlRefinerCheckpointConverter_Convert_AcceptsEmptyDict()
    {
        // Smoke: converter handles an empty input dict without throwing and returns empty per-component dicts.
        SdxlRefinerCheckpointConverter.ConvertedWeights w = SdxlRefinerCheckpointConverter.Convert([]);
        Assert.Empty(w.UNet);
        Assert.Empty(w.ClipG);
        Assert.Empty(w.Vae);
    }

    [Fact]
    public void SdxlRefinerCheckpointConverter_Convert_RoutesUnetVaeKeysCorrectly()
    {
        // Verify the routing logic catches the major LDM key prefixes and dispatches them to the right buckets.
        // We don't need real tensor shapes; placeholder 1-element F32 tensors are enough to verify routing.
        Tensor placeholder() => new Tensor(new TensorShape(1), DType.F32);

        Dictionary<string, Tensor> input = new()
        {
            // UNet: input_blocks.0.0 (conv_in)
            ["model.diffusion_model.input_blocks.0.0.weight"] = placeholder(),
            // UNet: a level-3 resnet (refiner-specific — base SDXL has no level 3)
            ["model.diffusion_model.input_blocks.10.0.in_layers.0.weight"] = placeholder(),
            // CLIP-G at conditioner index 0 (refiner uses 0, not 1 like base)
            ["conditioner.embedders.0.model.token_embedding.weight"] = placeholder(),
            // VAE
            ["first_stage_model.encoder.conv_in.weight"] = placeholder(),
            ["first_stage_model.decoder.conv_out.weight"] = placeholder(),
            // Junk (should be silently dropped)
            ["unrelated_garbage_key"] = placeholder(),
        };

        SdxlRefinerCheckpointConverter.ConvertedWeights w = SdxlRefinerCheckpointConverter.Convert(input);

        Assert.Contains("conv_in.weight", w.UNet.Keys);
        Assert.Contains("down_blocks.3.resnets.0.norm1.weight", w.UNet.Keys);
        Assert.Contains("text_model.embeddings.token_embedding.weight", w.ClipG.Keys);
        Assert.Contains("encoder.conv_in.weight", w.Vae.Keys);
        Assert.Contains("decoder.conv_out.weight", w.Vae.Keys);

        foreach (Tensor t in input.Values) t.Dispose();
    }

    [Fact]
    public void SdxlRefinerCheckpointConverter_Convert_LevelMapping_CoversAllInputBlocks()
    {
        // The 4-level refiner has 12 input_blocks (0=conv_in, 1-9=levels 0-2 with downsamples, 10-11=level 3 resnets).
        // Each LDM index must produce a non-null mapped key.
        Tensor placeholder() => new Tensor(new TensorShape(1), DType.F32);
        Dictionary<string, Tensor> input = new();
        for (int blk = 1; blk <= 11; blk++)
        {
            // Resnet sub-key (in_layers.0 = norm1) for non-downsample blocks; downsample blocks use op.weight
            // We use a generic in_layers form; downsamples will be skipped at this routing path — that's fine,
            // we just verify that they don't blow up. Real downsample keys use a different sub-prefix anyway.
            input[$"model.diffusion_model.input_blocks.{blk}.0.in_layers.0.weight"] = placeholder();
        }

        SdxlRefinerCheckpointConverter.ConvertedWeights w = SdxlRefinerCheckpointConverter.Convert(input);

        // Every level (0,1,2,3) should have at least one mapped resnet key.
        Assert.Contains(w.UNet.Keys, k => k.StartsWith("down_blocks.0.resnets."));
        Assert.Contains(w.UNet.Keys, k => k.StartsWith("down_blocks.1.resnets."));
        Assert.Contains(w.UNet.Keys, k => k.StartsWith("down_blocks.2.resnets."));
        Assert.Contains(w.UNet.Keys, k => k.StartsWith("down_blocks.3.resnets."));

        foreach (Tensor t in input.Values) t.Dispose();
    }
}

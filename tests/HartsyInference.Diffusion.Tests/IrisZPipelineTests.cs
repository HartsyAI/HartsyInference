using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Runs each Iris-Z task end to end on the CPU with a tiny random transformer and VAE and no prompt, so the
/// loops, grid bookkeeping and tensor ownership execute. Random weights say nothing about image quality, and the
/// guided (prompted) path needs a text encoder these tests do not have.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class IrisZPipelineTests
{
    private const int Size = 32;

    [Theory]
    [InlineData(IrisZDepthOutput.None)]
    [InlineData(IrisZDepthOutput.Joint)]
    [InlineData(IrisZDepthOutput.Estimated)]
    public void PromptToImageReturnsAnImageAndTheRequestedDepth(IrisZDepthOutput depth)
    {
        using Fixture fixture = new(skyCeiling: 0.95f);

        IrisZImage image = fixture.Pipeline.TextToImage(null, null, Request(seed: 3), depth);

        Assert.Equal((Size, Size, 3), (image.Width, image.Height, image.Seed));
        Assert.Equal(Size * Size * 3, image.Rgb.Length);
        if (depth == IrisZDepthOutput.None)
        {
            Assert.Null(image.LogDepth);
        }
        else
        {
            AssertLogDepth(image.LogDepth);
        }
    }

    [Fact]
    public void TheSameSeedReproducesTheImage()
    {
        using Fixture fixture = new(skyCeiling: 1f);

        IrisZImage first = fixture.Pipeline.TextToImage(null, null, Request(seed: 5), IrisZDepthOutput.None);
        IrisZImage second = fixture.Pipeline.TextToImage(null, null, Request(seed: 5), IrisZDepthOutput.None);
        IrisZImage other = fixture.Pipeline.TextToImage(null, null, Request(seed: 6), IrisZDepthOutput.None);

        Assert.Equal(first.Rgb, second.Rgb);
        Assert.NotEqual(first.Rgb, other.Rgb);
    }

    [Theory]
    [InlineData(4, 2, 1)]
    [InlineData(4, 0, 1)]
    [InlineData(3, 2, 3)]
    public void ImageToDepthReturnsOneLogDepthPerPixel(int steps, int averageLast, int seedCount)
    {
        using Fixture fixture = new(skyCeiling: 0.95f);
        using Tensor photo = IrisZTestData.Random(new Random(9), 1, 3, Size, Size);
        int[] seeds = new int[seedCount];
        for (int i = 0; i < seedCount; i++) seeds[i] = 20 + i;
        int reported = 0;

        float[] logDepth = fixture.Pipeline.ImageToDepth(photo, steps, averageLast, seeds, _ => reported++);

        AssertLogDepth(logDepth);
        Assert.Equal(steps * seedCount, reported);
    }

    [Fact]
    public void DepthToImageGeneratesAtTheDepthMapsSize()
    {
        using Fixture fixture = new(skyCeiling: 1f);
        float[] encoded = new float[Size * Size];
        for (int i = 0; i < encoded.Length; i++) encoded[i] = -1f + 2f * (i % Size) / (Size - 1);

        IrisZImage image = fixture.Pipeline.DepthToImage(encoded, null, null, Request(seed: 4));

        Assert.Equal(Size * Size * 3, image.Rgb.Length);
        Assert.Null(image.LogDepth);
    }

    [Fact]
    public void InputsOffTheSixteenPixelGridAreRefused()
    {
        using Fixture fixture = new(skyCeiling: 1f);
        using Tensor photo = IrisZTestData.Random(new Random(9), 1, 3, Size, Size + 8);

        Assert.Throws<ArgumentException>(() => fixture.Pipeline.ImageToDepth(photo, 4, 2, [1]));
        Assert.Throws<ArgumentException>(() => fixture.Pipeline.DepthToImage(new float[Size * (Size + 8)], null, null,
            Request(seed: 1) with { Height = Size + 8 }));
    }

    private static TextToImageRequest Request(int seed) => new()
    {
        Prompt = "",
        Width = Size,
        Height = Size,
        Steps = 2,
        CfgScale = 4f,
        Seed = seed,
    };

    private static void AssertLogDepth(float[]? logDepth)
    {
        Assert.NotNull(logDepth);
        Assert.Equal(Size * Size, logDepth.Length);
        // Averaged estimates are aligned by least squares and not clamped again, as in the reference.
        Assert.All(logDepth, value => Assert.InRange(value, -1.5f, 1.5f));
    }

    private sealed class Fixture : IDisposable
    {
        private readonly CpuBackend _backend = new();
        private readonly Flux2Transformer _transformer = new(IrisZTestData.Config(modalities: 4));
        private readonly Tensor _bnMean = IrisZTestData.Filled(0f, IrisZTestData.Channels);
        private readonly Tensor _bnVar = IrisZTestData.Filled(1f, IrisZTestData.Channels);
        private readonly Tensor _emptyPrompt = IrisZTestData.Random(new Random(8), 1, 5, IrisZTestData.Context);

        public Fixture(float skyCeiling)
        {
            _transformer.LoadWeights(IrisZTestData.Converted(zeroModality: false));
            IrisZTestData.VaeWeights vaeWeights = new();
            VaeDecoder decoder = new(IrisZTestData.Vae);
            decoder.LoadWeights(vaeWeights);
            VaeEncoder encoder = new(IrisZTestData.Vae);
            encoder.LoadWeights(vaeWeights);
            Pipeline = new IrisZPipeline(_backend, new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_8B), _transformer, decoder,
                encoder, _bnMean, _bnVar, IrisZTestData.Config(modalities: 4), _emptyPrompt, skyCeiling, [9, 18, 27]);
        }

        public IrisZPipeline Pipeline { get; }

        public void Dispose()
        {
            Pipeline.Dispose();
            _transformer.Dispose();
            _bnMean.Dispose();
            _bnVar.Dispose();
            _emptyPrompt.Dispose();
            _backend.Dispose();
        }
    }
}

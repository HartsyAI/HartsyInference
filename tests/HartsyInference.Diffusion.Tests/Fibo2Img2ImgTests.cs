using HartsyInference.Engine.Recipes;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 img2img and inpaint checks that need no weights: the init-latent, start-sigma, loop and mask-blend steps are
/// internal seams driven with a fake velocity, and the argument rules run before any encoder call.</summary>
public sealed class Fibo2Img2ImgTests
{
    private const int Steps = 10;
    private const int Side = 32;
    private static readonly float Shift = (float)Math.Exp(1.15);

    private static FlowMatchEulerDiscreteScheduler Scheduler()
    {
        FlowMatchEulerDiscreteScheduler scheduler = new(Shift);
        scheduler.SetTimesteps(Steps);
        return scheduler;
    }

    private static Tensor Filled(TensorShape shape, float value)
    {
        Tensor tensor = new(shape, DType.F32);
        tensor.AsSpan<float>().Fill(value);
        return tensor;
    }

    private static ImageToImageRequest Request(Tensor source, float strength, Tensor? mask = null) =>
        new(new TextToImageRequest { Prompt = "p", Width = Side, Height = Side, Steps = Steps, CfgScale = 5f }, source)
        {
            Strength = strength,
            Mask = mask,
        };

    [Fact]
    public void FullStrength_StartsFromTheSameNoiseAsTextToImage()
    {
        FlowMatchEulerDiscreteScheduler scheduler = Scheduler();
        using Tensor source = Filled(new TensorShape(1, 3, Side, Side), 0.3f);
        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(Request(source, 1f), Side, Side, Steps);
        TensorShape latentShape = new(1, 32, 4, 4);
        using Tensor sourceLatent = Filled(latentShape, 0.7f);

        using Tensor img2img = Fibo2Pipeline.BuildStartLatent(scheduler, SeedGenerator.CreateNoise(latentShape, 42), sourceLatent, plan.StartStep);
        using Tensor textToImage = Fibo2Pipeline.BuildStartLatent(scheduler, SeedGenerator.CreateNoise(latentShape, 42), null, 0);

        Assert.Equal(0, plan.StartStep);
        Assert.True(img2img.AsReadOnlySpan<float>().SequenceEqual(textToImage.AsReadOnlySpan<float>()));
    }

    [Fact]
    public void PartialStrength_StartsMidScheduleAtTheShiftedSigma()
    {
        FlowMatchEulerDiscreteScheduler scheduler = Scheduler();
        using Tensor source = Filled(new TensorShape(1, 3, Side, Side), 0.3f);
        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(Request(source, 0.5f), Side, Side, Steps);
        TensorShape latentShape = new(1, 32, 4, 4);
        using Tensor sourceLatent = Filled(latentShape, 0.7f);
        using Tensor noise = SeedGenerator.CreateNoise(latentShape, 5);
        float noiseValue = noise.AsReadOnlySpan<float>()[3];

        using Tensor start = Fibo2Pipeline.BuildStartLatent(scheduler, SeedGenerator.CreateNoise(latentShape, 5), sourceLatent, plan.StartStep);

        // sigma(t) = s·t / (1 + (s − 1)·t) with t = 1 − 5/10 and s = e^1.15.
        float sigma = Shift * 0.5f / (1f + (Shift - 1f) * 0.5f);
        Assert.Equal(5, plan.StartStep);
        Assert.Equal(sigma, scheduler.SigmaAt(plan.StartStep), 5);
        Assert.True(scheduler.SigmaAt(plan.StartStep) < 1f);
        Assert.Equal((1f - sigma) * 0.7f + sigma * noiseValue, start.AsReadOnlySpan<float>()[3], 5);

        List<float> sigmas = [];
        List<(int Step, int Total)> progress = [];
        Tensor? latent = Filled(latentShape, 0f);
        Fibo2Pipeline.SampleFlow(scheduler, ref latent, plan.StartStep, Steps,
            (_, s) => { sigmas.Add(s); return Filled(latentShape, 1f); }, null, p => progress.Add((p.Step, p.TotalSteps)));
        latent!.Dispose();

        Assert.Equal(5, sigmas.Count);
        Assert.Equal(sigma, sigmas[0], 5);
        Assert.Equal((1, 5), progress[0]);
        Assert.Equal((5, 5), progress[^1]);
    }

    [Fact]
    public void MaskedOutRegion_FollowsTheSourceAndTheMaskedRegionDoesNot()
    {
        FlowMatchEulerDiscreteScheduler scheduler = Scheduler();
        TensorShape latentShape = new(1, 32, 4, 4);
        using Tensor sourceLatent = Filled(latentShape, 0.7f);
        // Left half kept (0), right half regenerated (1).
        using Tensor mask = new(new TensorShape(1, 1, 4, 4), DType.F32);
        Span<float> maskValues = mask.AsSpan<float>();
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++)
                maskValues[y * 4 + x] = x < 2 ? 0f : 1f;
        Fibo2Pipeline.LatentInpaint inpaint = new(scheduler, sourceLatent, mask, seed: 11);

        Tensor? latent = Fibo2Pipeline.BuildStartLatent(scheduler, SeedGenerator.CreateNoise(latentShape, 11), sourceLatent, 0);
        Fibo2Pipeline.SampleFlow(scheduler, ref latent, 0, Steps, (_, _) => Filled(latentShape, 0f), inpaint, null);

        ReadOnlySpan<float> result = latent!.AsReadOnlySpan<float>();
        for (int c = 0; c < 32; c++)
        {
            for (int y = 0; y < 4; y++)
            {
                Assert.Equal(0.7f, result[(c * 4 + y) * 4 + 0], 6);
                Assert.Equal(0.7f, result[(c * 4 + y) * 4 + 1], 6);
                Assert.NotEqual(0.7f, result[(c * 4 + y) * 4 + 2], 3);
            }
        }
        latent!.Dispose();
    }

    [Fact]
    public void NormalizeVaeLatent_IsTheExactInverseOfTheDecodeSideDenormalisation()
    {
        Fibo2Config config = Fibo2Config.Checkpoint;
        using Tensor vae = SeedGenerator.CreateNoise(new TensorShape(1, 32, 4, 6), 3);
        using Tensor mean = SeedGenerator.CreateNoise(new TensorShape(128), 4);
        using Tensor variance = new(new TensorShape(128), DType.F32);
        Span<float> varianceValues = variance.AsSpan<float>();
        for (int i = 0; i < varianceValues.Length; i++)
            varianceValues[i] = 0.5f + i * 0.02f;

        using Tensor normalized = Fibo2Pipeline.NormalizeVaeLatent(vae, config, mean, variance, 1e-5f);
        using Tensor patched = Flux2LatentOps.Patchify(normalized, 128, config.PatchSize);
        using Tensor unnormalized = Flux2LatentOps.BnUnNormalize(patched, mean, variance, 1e-5f);
        using Tensor restored = Flux2LatentOps.Unpatchify(unnormalized, config.InChannels, config.PatchSize);

        ReadOnlySpan<float> expected = vae.AsReadOnlySpan<float>();
        ReadOnlySpan<float> actual = restored.AsReadOnlySpan<float>();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 4);
        Assert.False(normalized.AsReadOnlySpan<float>().SequenceEqual(expected));
    }

    [Fact]
    public void ZeroStrength_ReturnsTheSourceWithoutEncodingAnything()
    {
        using Fibo2Pipeline pipeline = UnloadedPipeline(withEncoder: true);
        using Tensor source = Filled(new TensorShape(1, 3, Side, Side), 0.25f);

        (byte[] rgb, int width, int height, int _) = pipeline.GenerateFromTokens([1], [1], Side, Side, Steps, 5f, Shift, 7,
            img2img: Request(source, 0f));

        Assert.Equal((Side, Side), (width, height));
        Assert.Equal(ImagePostProcessor.TensorToRgbBytes(source), rgb);
    }

    [Fact]
    public void Img2Img_WithoutAVaeEncoderIsRefusedClearly()
    {
        using Fibo2Pipeline pipeline = UnloadedPipeline(withEncoder: false);
        using Tensor source = Filled(new TensorShape(1, 3, Side, Side), 0.25f);

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            pipeline.GenerateFromTokens([1], [1], Side, Side, Steps, 5f, Shift, 7, img2img: Request(source, 0.5f)));

        Assert.Contains("encoder", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Img2Img_RejectsASourceOfTheWrongSize()
    {
        using Fibo2Pipeline pipeline = UnloadedPipeline(withEncoder: true);
        using Tensor source = Filled(new TensorShape(1, 3, Side * 2, Side), 0.25f);

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromTokens([1], [1], Side, Side, Steps, 5f, Shift, 7, img2img: Request(source, 0.5f)));
    }

    private static Fibo2Pipeline UnloadedPipeline(bool withEncoder)
    {
        Fibo2Config config = Fibo2Config.Checkpoint with
        {
            Dim = 64, NumHeads = 2, HeadDim = 32, NumLayers = 2, NumRefinerLayers = 1, InjectionLayers = [1], CapFeatDim = 96,
            PerceiverQueries = 8, PerceiverLayers = 1, FfnDim = 96, AxesDims = [8, 12, 12], MaxGist = 8, MinGist = 4, GistStep = 2,
            GistMinTextLen = 16, GistMaxTextLen = 64,
        };
        LlamaStyleEncoderConfig encoderConfig = LlamaStyleEncoderConfig.Qwen3_VL_4B with
        {
            HiddenSize = 32, NumLayers = 36, NumQueryHeads = 2, NumKvHeads = 1, HeadDim = 16, IntermediateSize = 64, VocabSize = 64,
        };
        return new Fibo2Pipeline(new CpuBackend(), new LlamaStyleEncoder(encoderConfig), new Fibo2Transformer(config),
            new VaeDecoder(VaeConfig.Flux2), new Tensor(new TensorShape(128), DType.F32), new Tensor(new TensorShape(128), DType.F32),
            config, vaeEncoder: withEncoder ? new VaeEncoder(VaeConfig.Flux2) : null);
    }
}

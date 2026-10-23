using HartsyInference.Engine.Recipes;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 pipeline and recipe-pipeline checks that need no weights: the argument rules run before any encoder or denoiser
/// call, so each refusal is reached with unloaded components.</summary>
public sealed class Fibo2PipelineTests
{
    [Fact]
    public void Constructor_RejectsBatchNormStatsOfTheWrongWidth()
    {
        CpuBackend backend = new();
        using Fibo2Transformer transformer = new(TinyConfig());
        using LlamaStyleEncoder encoder = new(TinyEncoderConfig());
        VaeDecoder decoder = new(VaeConfig.Flux2);
        using Tensor badMean = new(new TensorShape(127), DType.F32);
        using Tensor badVar = new(new TensorShape(127), DType.F32);

        Assert.Throws<ArgumentException>(() =>
            new Fibo2Pipeline(backend, encoder, transformer, decoder, badMean, badVar, TinyConfig()));
    }

    [Fact]
    public void GenerateFromTokens_RejectsASizeOffTheSixteenPixelGrid()
    {
        CpuBackend backend = new();
        using Fibo2Transformer transformer = new(TinyConfig());
        using LlamaStyleEncoder encoder = new(TinyEncoderConfig());
        VaeDecoder decoder = new(VaeConfig.Flux2);
        using Tensor mean = new(new TensorShape(128), DType.F32);
        using Tensor variance = new(new TensorShape(128), DType.F32);
        using Fibo2Pipeline pipeline = new(backend, encoder, transformer, decoder, mean, variance, TinyConfig());

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromTokens([1], [1], width: 1000, height: 1024, steps: 4, guidanceScale: 5f, sigmaShift: 1.15f, seed: 7));
    }

    [Fact]
    public void Generate_RefusesNegativePromptsBeforeAnyWork()
    {
        using Qwen3Tokenizer tokenizer = new();
        CpuBackend backend = new();
        using Fibo2Transformer transformer = new(TinyConfig());
        using LlamaStyleEncoder encoder = new(TinyEncoderConfig());
        VaeDecoder decoder = new(VaeConfig.Flux2);
        using Tensor mean = new(new TensorShape(128), DType.F32);
        using Tensor variance = new(new TensorShape(128), DType.F32);
        Fibo2Pipeline pipeline = new(backend, encoder, transformer, decoder, mean, variance, TinyConfig());
        Fibo2RecipePipeline recipe = new(pipeline, tokenizer, Fibo2Recipe.FamilyDefaults,
            new ResolvedModelVariant("fibo2", Fibo2Variants.Dev, ModelVariantSource.CallerHint, "test"), []);

        NotSupportedException error = Assert.Throws<NotSupportedException>(() =>
            recipe.Generate(new ImageRequest { Prompt = "a cat", NegativePrompt = "blurry" }, Settings(), null, CancellationToken.None));

        Assert.Contains("negative prompts", error.Message, StringComparison.Ordinal);
        recipe.Dispose();
        pipeline.Dispose();
    }

    /// <summary>A small DiT: dim 64, two heads, two blocks (one with injection), an 8-query Perceiver. Constructors do not allocate
    /// weights, but the full checkpoint sizes would still make these tests heavy for no benefit.</summary>
    private static Fibo2Config TinyConfig() => Fibo2Config.Checkpoint with
    {
        Dim = 64,
        NumHeads = 2,
        HeadDim = 32,
        NumLayers = 2,
        NumRefinerLayers = 1,
        InjectionLayers = [1],
        CapFeatDim = 96,
        PerceiverQueries = 8,
        PerceiverLayers = 1,
        FfnDim = 96,
        AxesDims = [8, 12, 12],
        MaxGist = 8,
        MinGist = 4,
        GistStep = 2,
        GistMinTextLen = 16,
        GistMaxTextLen = 64,
    };

    private static LlamaStyleEncoderConfig TinyEncoderConfig() => LlamaStyleEncoderConfig.Qwen3_VL_4B with
    {
        HiddenSize = 32,
        NumLayers = 36,
        NumQueryHeads = 2,
        NumKvHeads = 1,
        HeadDim = 16,
        IntermediateSize = 64,
        VocabSize = 64,
    };

    [Fact]
    public void Generate_RefusesSamplerSelectionsBeforeAnyWork()
    {
        using Qwen3Tokenizer tokenizer = new();
        CpuBackend backend = new();
        using Fibo2Transformer transformer = new(TinyConfig());
        using LlamaStyleEncoder encoder = new(TinyEncoderConfig());
        VaeDecoder decoder = new(VaeConfig.Flux2);
        using Tensor mean = new(new TensorShape(128), DType.F32);
        using Tensor variance = new(new TensorShape(128), DType.F32);
        Fibo2Pipeline pipeline = new(backend, encoder, transformer, decoder, mean, variance, TinyConfig());
        Fibo2RecipePipeline recipe = new(pipeline, tokenizer, Fibo2Recipe.FamilyDefaults,
            new ResolvedModelVariant("fibo2", Fibo2Variants.Dev, ModelVariantSource.CallerHint, "test"), []);

        Assert.Throws<NotSupportedException>(() =>
            recipe.Generate(new ImageRequest { Prompt = "a cat", Sampler = "euler_ancestral" }, Settings(), null, CancellationToken.None));
        recipe.Dispose();
        pipeline.Dispose();
    }

    private static ImageEffectiveSettings Settings() =>
        new() { Width = 1024, Height = 1024, Steps = 4, CfgScale = 5f };
}

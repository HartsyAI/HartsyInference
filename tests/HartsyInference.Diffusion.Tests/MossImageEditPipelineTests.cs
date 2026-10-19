using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The whole MOSS-Image pipeline with reference images, end to end on the CPU: the tiny transformer fixture, a tiny
/// random VAE in the FLUX.2 key layout (<c>tests/python-reference/flux2_vae_tiny_fixture.py</c>) and a stand-in
/// multimodal encoder. Checks the wiring an edit adds: references reach the encoder for both prompts and the DiT in both
/// attention modes, plain generation stays deterministic, and the references' tensors stay the caller's.</summary>
public sealed unsafe class MossImageEditPipelineTests
{
    private const int Width = 64, Height = 48;
    private static readonly VaeConfig TinyVae = VaeConfig.Flux2 with { LatentChannels = 4, BlockOutChannels = [8, 8, 8, 8], NormNumGroups = 4 };

    [Fact]
    public void Edit_ReachesTheEncoderAndTheDiT_InBothAttentionModes()
    {
        using MossImageFixture fixture = new();
        using SafeTensorsLoader vaeFile = new();
        vaeFile.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "MossImage", "flux2_vae_tiny.safetensors"));
        Dictionary<string, Tensor> vaeWeights = vaeFile.GetAllTensors();
        VaeEncoder vaeEncoder = new(TinyVae);
        vaeEncoder.LoadWeights(vaeWeights);
        VaeDecoder vaeDecoder = new(TinyVae);
        vaeDecoder.LoadWeights(vaeWeights);
        using MossImageTransformer transformer = new(fixture.Transformer.Config with { VaeLatentChannels = 4 });
        transformer.LoadWeights(fixture.Weights);
        StandInEncoder encoder = new();
        using IBackend backend = new CpuBackend();
        using MossImagePipeline pipeline = new(backend, encoder, transformer, vaeDecoder, vaeEncoder,
            vaeWeights["bn.running_mean"], vaeWeights["bn.running_var"]);
        using Tensor latentCopy = Pattern(new TensorShape(1, 3, 32, 48), -1f, 1f);
        using Tensor visionCopy = Pattern(new TensorShape(1, 3, 32, 32), 0f, 1f);

        byte[] plain = Run(pipeline, null);
        Assert.Equal(plain, Run(pipeline, null));
        Assert.Equal(0, encoder.ImageCalls);

        byte[] cached = Run(pipeline, Inputs(latentCopy, visionCopy, ReferenceAttentionMode.CachedKv));
        Assert.Equal(2, encoder.ImageCalls);
        byte[] joint = Run(pipeline, Inputs(latentCopy, visionCopy, ReferenceAttentionMode.Joint));
        Assert.NotEqual(plain, cached);
        Assert.NotEqual(cached, joint);
        Assert.NotEqual(plain, joint);
        Assert.Equal(new TensorShape(1, 3, 32, 48), latentCopy.Shape);
        Assert.Equal(cached, Run(pipeline, Inputs(latentCopy, visionCopy, ReferenceAttentionMode.CachedKv)));
    }

    [Fact]
    public void Edit_RefusesReferencesItCannotUse()
    {
        using MossImageFixture fixture = new();
        using SafeTensorsLoader vaeFile = new();
        vaeFile.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "MossImage", "flux2_vae_tiny.safetensors"));
        Dictionary<string, Tensor> vaeWeights = vaeFile.GetAllTensors();
        VaeDecoder vaeDecoder = new(TinyVae);
        vaeDecoder.LoadWeights(vaeWeights);
        using MossImageTransformer transformer = new(fixture.Transformer.Config with { VaeLatentChannels = 4 });
        transformer.LoadWeights(fixture.Weights);
        using IBackend backend = new CpuBackend();
        using MossImagePipeline decodeOnly = new(backend, new StandInEncoder(), transformer, vaeDecoder, null,
            vaeWeights["bn.running_mean"], vaeWeights["bn.running_var"]);
        using Tensor odd = Pattern(new TensorShape(1, 3, 30, 48), -1f, 1f);
        using Tensor vision = Pattern(new TensorShape(1, 3, 32, 32), 0f, 1f);
        Assert.Throws<InvalidOperationException>(() => Run(decodeOnly, Inputs(odd, vision, ReferenceAttentionMode.CachedKv)));
        ReferenceImageInputs mismatched = new() { Latent = [odd, odd], Vision = [vision] };
        Assert.Throws<ArgumentException>(() => mismatched.Validate("test"));
    }

    private static byte[] Run(MossImagePipeline pipeline, ReferenceImageInputs? references)
    {
        WeightedTokenSequence prompt = Tokens([1, 2, 3, 4, 5, 6, 7], references is null ? 0 : 4);
        WeightedTokenSequence negative = Tokens([9, 8], references is null ? 0 : 4);
        TextToImageRequest request = new() { Prompt = "x", Width = Width, Height = Height, Steps = 3, CfgScale = 3f, Seed = 7 };
        (byte[] rgb, int width, int height, int _) = pipeline.Generate(prompt, negative, request,
            new MossImageGuidance { UncondSkipLayers = [1] }, null, references);
        Assert.Equal((Width, Height), (width, height));
        return rgb;
    }

    private static ReferenceImageInputs Inputs(Tensor latent, Tensor vision, ReferenceAttentionMode mode) => new()
    {
        Latent = [latent],
        Vision = [vision],
        Options = new ReferenceConditioningOptions { Mode = mode },
    };

    /// <summary>Prompt ids with a run of <paramref name="pads"/> image placeholders (id 77) before the text.</summary>
    private static WeightedTokenSequence Tokens(int[] text, int pads)
    {
        int[] ids = [.. Enumerable.Repeat(StandInEncoder.ImagePad, pads), .. text];
        float[] ones = new float[ids.Length];
        Array.Fill(ones, 1f);
        return new WeightedTokenSequence(ids, ones) { UniformWeight = 1f };
    }

    private static Tensor Pattern(TensorShape shape, float low, float high)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = low + (high - low) * (0.5f + 0.5f * MathF.Sin(i * 0.37f));
        return t;
    }

    /// <summary>A deterministic 48-wide "encoder": text rows from the ids, placeholder rows from the image's mean.</summary>
    private sealed class StandInEncoder : IMultimodalFinalHiddenStateEncoder
    {
        public const int ImagePad = 77;

        public int ImageCalls { get; private set; }

        public int HiddenSize => 48;

        public bool HasVision => true;

        public (int Width, int Height) VisionInputSize(int width, int height) => (width, height);

        public int CountImageTokens(int width, int height) => 4;

        public Tensor Encode(IBackend backend, IReadOnlyList<int> tokenIds) => Encode(backend, tokenIds, []);

        public Tensor Encode(IBackend backend, IReadOnlyList<int> tokenIds, IReadOnlyList<Tensor> images)
        {
            if (images.Count > 0) ImageCalls++;
            float imageMean = 0f;
            foreach (Tensor image in images)
            {
                float* p = (float*)image.DataPointer;
                for (long i = 0; i < image.ElementCount; i++) imageMean += p[i] / image.ElementCount;
            }
            Tensor hidden = new(new TensorShape(1, tokenIds.Count, HiddenSize), DType.F32);
            float* h = (float*)hidden.DataPointer;
            for (int t = 0; t < tokenIds.Count; t++)
            {
                for (int d = 0; d < HiddenSize; d++)
                {
                    h[t * HiddenSize + d] = tokenIds[t] == ImagePad
                        ? MathF.Cos(imageMean * 3f + d * 0.21f + t)
                        : MathF.Sin(tokenIds[t] * 0.37f + d * 0.11f);
                }
            }
            return hidden;
        }

        public IEnumerable<Tensor> EnumerateWeights() => [];

        public IEnumerable<Tensor> EnumerateVisionWeights() => [];
    }
}

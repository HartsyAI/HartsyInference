using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Requests;
using SharpInference.Tests.Common;
using SharpInference.Video.Pipelines;

namespace SharpInference.Video.Tests;

/// <summary>End-to-end structural tests for the Kandinsky 5 T2V/I2V pipeline: tiny-config
/// <see cref="Kandinsky5Transformer"/> (visual-cond video forward) + HunyuanVideo VAE driven through
/// <see cref="Kandinsky5VideoPipeline"/> on CPU. Numerics vs the real checkpoint are validation-pending.</summary>
public unsafe class Kandinsky5VideoPipelineTests
{
    private readonly ITestOutputHelper _output;
    public Kandinsky5VideoPipelineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void GenerateFromEmbeddings_T2V_TinyConfig_ProducesExpandedFrames()
    {
        (Kandinsky5VideoPipeline pipeline, Kandinsky5Config config) = BuildPipeline(withEncoder: false);

        Tensor qwen = Rand([1, 5, config.InTextDim], seed: 1);
        Tensor clip = Rand([1, config.InTextDim2], seed: 2);
        Tensor negQwen = Rand([1, 4, config.InTextDim], seed: 3);
        Tensor negClip = Rand([1, config.InTextDim2], seed: 4);

        // 32x32 px → 4x4 latent (8x), patch 2 → 2x2 grid; 5 frames → 2 latent frames.
        TextToImageRequest req = new()
        {
            Prompt = "(embeddings)", Width = 32, Height = 32, Steps = 2, CfgScale = 5.0f, Seed = 42,
        };
        (byte[][] frames, int w, int h, int seed) = pipeline.GenerateFromEmbeddings(
            qwen, clip, negQwen, negClip, req, numFrames: 5,
            p => _output.WriteLine($"step {p.Step}/{p.TotalSteps}"));

        Assert.Equal(5, frames.Length);   // (T_lat-1)*4+1
        Assert.Equal(32, w);
        Assert.Equal(32, h);
        Assert.Equal(42, seed);
        foreach (byte[] f in frames) Assert.Equal(32 * 32 * 3, f.Length);

        qwen.Dispose(); clip.Dispose(); negQwen.Dispose(); negClip.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_T2V_CfgWithoutNegatives_Throws()
    {
        (Kandinsky5VideoPipeline pipeline, Kandinsky5Config config) = BuildPipeline(withEncoder: false);
        Tensor qwen = Rand([1, 5, config.InTextDim], seed: 1);
        Tensor clip = Rand([1, config.InTextDim2], seed: 2);
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 1, CfgScale = 5.0f, Seed = 1 };

        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromEmbeddings(qwen, clip, null, null, req, numFrames: 5));

        // Frame counts violating (F-1) % 4 == 0 fail fast too.
        Assert.Throws<ArgumentException>(() =>
            pipeline.GenerateFromEmbeddings(qwen, clip, null, null, req with { CfgScale = 1.0f }, numFrames: 4));

        qwen.Dispose(); clip.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_I2V_FirstFrameLatent_ProducesFrames()
    {
        (Kandinsky5VideoPipeline pipeline, Kandinsky5Config config) = BuildPipeline(withEncoder: false);

        Tensor qwen = Rand([1, 5, config.InTextDim], seed: 1);
        Tensor clip = Rand([1, config.InTextDim2], seed: 2);
        Tensor condition = Rand([1, config.InVisualDim, 1, 4, 4], seed: 9);
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 2, CfgScale = 1.0f, Seed = 7 };

        (byte[][] frames, int w, int h, _) = pipeline.GenerateFromEmbeddings(
            qwen, clip, null, null, req, numFrames: 9, onProgress: null, firstFrameLatent: condition);

        Assert.Equal(9, frames.Length);
        Assert.Equal(32, w);
        Assert.Equal(32, h);
        foreach (byte[] f in frames) Assert.Equal(32 * 32 * 3, f.Length);

        // Wrong conditioning shape must fail fast.
        Tensor bad = Rand([1, config.InVisualDim, 1, 8, 8], seed: 10);
        Assert.Throws<ArgumentException>(() => pipeline.GenerateFromEmbeddings(
            qwen, clip, null, null, req, numFrames: 9, onProgress: null, firstFrameLatent: bad));

        condition.Dispose(); bad.Dispose(); qwen.Dispose(); clip.Dispose();
    }

    [Fact]
    public void GenerateFromEmbeddings_I2V_FromRgbBytes_EncodesAndProducesFrames()
    {
        (Kandinsky5VideoPipeline pipeline, Kandinsky5Config config) = BuildPipeline(withEncoder: true);

        byte[] firstFrame = new byte[32 * 32 * 3];
        new Random(5).NextBytes(firstFrame);
        Tensor condition = pipeline.EncodeFirstFrame(firstFrame, width: 32, height: 32);
        Assert.Equal(config.InVisualDim, condition.Shape[1]);
        Assert.Equal(1, condition.Shape[2]);
        Assert.Equal(4, condition.Shape[3]);
        Assert.Equal(4, condition.Shape[4]);

        Tensor qwen = Rand([1, 5, config.InTextDim], seed: 1);
        Tensor clip = Rand([1, config.InTextDim2], seed: 2);
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 2, CfgScale = 1.0f, Seed = 7 };

        (byte[][] frames, int w, int h, _) = pipeline.GenerateFromEmbeddings(
            qwen, clip, null, null, req, numFrames: 5, onProgress: null, firstFrameLatent: condition);
        condition.Dispose();

        Assert.Equal(5, frames.Length);
        Assert.Equal(32, w);
        Assert.Equal(32, h);

        // Without an encoder, the RGB entry fails fast.
        (Kandinsky5VideoPipeline noEncoder, _) = BuildPipeline(withEncoder: false);
        Assert.Throws<InvalidOperationException>(() => noEncoder.EncodeFirstFrame(firstFrame, 32, 32));

        qwen.Dispose(); clip.Dispose();
    }

    [Fact]
    public void GetRopeScaleFactor_FollowsReferenceResolutionRule()
    {
        Assert.Equal((1f, 2f, 2f), Kandinsky5VideoPipeline.GetRopeScaleFactor(512, 768));
        Assert.Equal((1f, 2f, 2f), Kandinsky5VideoPipeline.GetRopeScaleFactor(480, 854));
        Assert.Equal((1f, 3.16f, 3.16f), Kandinsky5VideoPipeline.GetRopeScaleFactor(768, 1024));
        Assert.Equal((1f, 3.16f, 3.16f), Kandinsky5VideoPipeline.GetRopeScaleFactor(32, 32));
    }

    [Fact]
    public void Constructor_NonVisualCondConfig_Throws()
    {
        CpuBackend backend = new();
        Kandinsky5Config imageConfig = Kandinsky5SyntheticWeights.TinyImageConfig;
        using Kandinsky5Transformer transformer = new(imageConfig);
        HunyuanVideoVaeDecoder vae = new(Kandinsky5SyntheticWeights.TinyVaeConfig);

        Assert.Throws<ArgumentException>(() =>
            new Kandinsky5VideoPipeline(backend, transformer, vae, imageConfig));
    }

    private static (Kandinsky5VideoPipeline pipeline, Kandinsky5Config config) BuildPipeline(bool withEncoder)
    {
        CpuBackend backend = new();
        Kandinsky5Config config = Kandinsky5SyntheticWeights.TinyVideoConfig;
        Kandinsky5Transformer transformer = new(config);
        transformer.LoadWeights(Kandinsky5SyntheticWeights.BuildTransformer(config));

        HunyuanVideoVaeConfig vaeConfig = Kandinsky5SyntheticWeights.TinyVaeConfig;
        Dictionary<string, Tensor> vaeWeights = Kandinsky5SyntheticWeights.BuildVae(vaeConfig);
        HunyuanVideoVaeDecoder vae = new(vaeConfig);
        vae.LoadWeights(vaeWeights);

        HunyuanVideoVaeEncoder? encoder = null;
        if (withEncoder)
        {
            encoder = new HunyuanVideoVaeEncoder(vaeConfig);
            encoder.LoadWeights(vaeWeights);
        }

        return (new Kandinsky5VideoPipeline(backend, transformer, vae, config, encoder), config);
    }

    private static Tensor Rand(int[] dims, int seed)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}

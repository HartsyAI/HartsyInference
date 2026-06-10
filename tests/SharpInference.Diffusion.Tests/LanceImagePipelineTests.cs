using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Requests;
using SharpInference.Tests.Common;

namespace SharpInference.Diffusion.Tests;

/// <summary>Full end-to-end structural test for the Lance T2I pipeline: a tiny-config <see cref="LanceTransformer"/> + <see cref="Wan22VaeDecoder"/> driven through <see cref="LanceImagePipeline.GenerateFromTokens"/> on the CPU backend with synthetic weights. Proves the entire backbone → latent handoff → VAE wiring composes and produces a correctly-shaped RGB image. (Numeric correctness vs the real Lance checkpoint is validation-pending — no weights here.)</summary>
public unsafe class LanceImagePipelineTests
{
    private readonly ITestOutputHelper _output;
    public LanceImagePipelineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void GenerateFromTokens_TinyConfig_ProducesCorrectlyShapedRgb()
    {
        CpuBackend backend = new();

        // Tiny backbone: hidden 32, 2 layers, 4 heads : 2 KV (head_dim 8), ffn 64, mrope (2,1,1) sums to 4 = head_dim/2.
        LanceConfig cfg = new()
        {
            HiddenSize = 32, NumLayers = 2, NumHeads = 4, NumKvHeads = 2,
            IntermediateSize = 64, MropeSection = (2, 1, 1), VocabSize = 64, QkNorm = false,
        };
        int vocab = cfg.VocabSize;

        LanceTransformer transformer = new(cfg);
        transformer.LoadWeights(LanceSyntheticWeights.BuildTransformer(cfg));

        int[] dimMult = [1, 2, 4, 4];
        bool[] tUp = [false, true, true];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: tUp);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, tUp));

        LanceImagePipeline pipeline = new(backend, transformer, vae, cfg);

        int[] prompt = [1, 2, 3, 4, 5];
        int[] neg = [1, 2];
        TextToImageRequest req = new()
        {
            Prompt = "x", NegativePrompt = "", Width = 64, Height = 64, Steps = 2, CfgScale = 4.0f, Seed = 42,
        };

        (byte[] rgb, int w, int h, int seed) = pipeline.GenerateFromTokens(prompt, neg, req,
            p => _output.WriteLine($"step {p.Step}/{p.TotalSteps}"));

        Assert.Equal(64, w);
        Assert.Equal(64, h);
        Assert.Equal(64 * 64 * 3, rgb.Length);   // bytes are clamped [0,255] by construction
        _ = vocab;
    }
}

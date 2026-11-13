using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;
using HartsyInference.ThreeD.Tests.MeshPostProcess;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>SyntheticSmoke lane: the whole Paint pipeline on a tiny random UNet, a pooling stand-in for the VAE and a random DINO stand-in. It proves plumbing, shapes,
/// finiteness and bake coverage, not image quality or parity with upstream.</summary>
public sealed unsafe class HunyuanPaintPipelineTests
{
    private static HunyuanPaintPipeline TinyPipeline(CpuBackend backend, bool v21)
    {
        HunyuanPaintConfig cfg = HunyuanPaintTestWeights.TinyConfig(v21) with { CameraClasses = v21 ? 0 : 49 };
        HunyuanPaintUnet unet = new(cfg, generating: true), reference = new(cfg, generating: false);
        Dictionary<string, Tensor> weights = HunyuanPaintTestWeights.Create(cfg, generating: true, seed: 5);
        unet.LoadWeights(weights);
        reference.LoadWeights(HunyuanPaintTestWeights.Create(cfg, generating: false, seed: 6));
        Dictionary<string, Tensor> tokens = [];
        foreach (string name in v21 ? new[] { "learned_text_clip_albedo", "learned_text_clip_mr", "learned_text_clip_ref" } : new[] { "learned_text_clip_gen", "learned_text_clip_ref" })
            tokens[name] = HunyuanPaintUnetTests.Random(name.Length, 1, cfg.TextTokens, cfg.CrossAttentionDim);
        return new(backend, cfg, unet, reference, HunyuanPaintPromptTokens.FromWeights(tokens, cfg), new PoolingCodec(),
            v21 ? new RandomConditioner(cfg.CrossAttentionDim) : null);
    }

    private static readonly HunyuanPaintOptions TinyOptions = new() { ViewResolution = 64, RenderResolution = 128, TextureSize = 32, Steps = 2, MaxViews = 6, Seed = 1 };

    private static byte[] GreyReference()
    {
        byte[] reference64 = new byte[64 * 64 * 3];
        Array.Fill(reference64, (byte)180);
        return reference64;
    }

    [Theory]
    [Trait("Category", "SyntheticSmoke")]
    [InlineData(false)]
    [InlineData(true)]
    public void Generate_OnATinyNetworkProducesAFiniteBakedMaterial(bool v21)
    {
        using CpuBackend backend = new();
        HunyuanPaintPipeline pipeline = TinyPipeline(backend, v21);

        List<int> steps = [];
        PbrPostProcessResult result = pipeline.Generate(SyntheticMeshes.Icosphere(1), GreyReference(), 64, 64, TinyOptions, p => steps.Add(p.Step));

        Assert.Equal([1, 2], steps);
        Assert.Equal(32, result.Material.BaseColor.Width);
        Assert.Equal(3, result.Material.BaseColor.Channels);
        Assert.True(result.CoveredTexels > 0, "no texel was painted");
        Assert.NotEmpty(result.Mesh.Uvs!);
        Assert.Empty(result.Warnings);
    }

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void Generate_OnAMeshTornIntoSeparateTrianglesWarnsAboutTheConfettiAndSaysWhatToDo()
    {
        // 300 triangles that share nothing: the unwrapper can only make one chart per triangle.
        const int pieces = 300;
        float[] vertices = new float[pieces * 9];
        int[] indices = new int[pieces * 3];
        for (int i = 0; i < pieces; i++)
        {
            double angle = i * 0.61803398875 * 2 * Math.PI, z = i / (double)pieces * 1.8 - 0.9, r = Math.Sqrt(Math.Max(0, 1 - z * z));
            float x = (float)(r * Math.Cos(angle)), y = (float)(r * Math.Sin(angle));
            float[] corner = [x, y, (float)z, x + 0.03f, y, (float)z, x, y + 0.03f, (float)z + 0.01f];
            corner.CopyTo(vertices, i * 9);
            for (int k = 0; k < 3; k++)
                indices[i * 3 + k] = i * 3 + k;
        }

        // 256 px, not the 32 of the other tests: 300 charts do not pack into 32 px ("Atlas packing failed").
        using CpuBackend backend = new();
        PbrPostProcessResult result = TinyPipeline(backend, v21: true).Generate(new Mesh { Vertices = vertices, Indices = indices }, GreyReference(), 64, 64, TinyOptions with { TextureSize = 256 });

        string warning = Assert.Single(result.Warnings);
        Assert.Contains("300 charts", warning);
        Assert.Contains("remove-small-components", warning);
        Assert.Contains("decimate", warning);
    }

    [Fact]
    public void CameraClass_FollowsTheUpstreamTableAndGuidanceScaleTheAzimuthMapping()
    {
        Assert.Equal(26, HunyuanPaintPipeline.CameraClass(0, 0));        // ((0 + 9) % 12) / 1 + 12 + 5
        Assert.Equal(5 + 12 + (3 + 9) % 12, HunyuanPaintPipeline.CameraClass(0, 90));
        Assert.Equal(5 + 40 + ((0 + 9) % 12) / 3, HunyuanPaintPipeline.CameraClass(90, 0));
        Assert.Equal(5 + 36 + ((6 + 9) % 12) / 3, HunyuanPaintPipeline.CameraClass(-90, 180));
        Assert.Throws<ArgumentOutOfRangeException>(() => HunyuanPaintPipeline.CameraClass(30, 0));
        Assert.Equal(1f, HunyuanPaintPipeline.ViewScale(0));
        Assert.Equal(1.5f, HunyuanPaintPipeline.ViewScale(45));
        Assert.Equal(2f, HunyuanPaintPipeline.ViewScale(180));
        Assert.Equal(2f, HunyuanPaintPipeline.ViewScale(300));
        Assert.Equal(1f, HunyuanPaintPipeline.ViewScale(360));
    }

    private sealed class PoolingCodec : IPaintLatentCodec
    {
        public Tensor Encode(IBackend backend, Tensor image)
        {
            int n = (int)image.Shape[0], size = (int)image.Shape[2], latent = size / 8;
            Tensor output = new(new TensorShape(n, 4, latent, latent), DType.F32);
            float* src = (float*)image.DataPointer, dst = (float*)output.DataPointer;
            for (int v = 0; v < n; v++)
                for (int y = 0; y < latent; y++)
                    for (int x = 0; x < latent; x++)
                    {
                        float[] mean = new float[3];
                        for (int c = 0; c < 3; c++)
                        {
                            for (int dy = 0; dy < 8; dy++)
                                for (int dx = 0; dx < 8; dx++)
                                    mean[c] += src[((v * 3 + c) * size + y * 8 + dy) * size + x * 8 + dx];
                            mean[c] /= 64;
                            dst[((v * 4 + c) * latent + y) * latent + x] = mean[c];
                        }
                        dst[((v * 4 + 3) * latent + y) * latent + x] = (mean[0] + mean[1] + mean[2]) / 3;
                    }
            return output;
        }

        public Tensor Decode(IBackend backend, Tensor latents)
        {
            int n = (int)latents.Shape[0], latent = (int)latents.Shape[2], size = latent * 8;
            Tensor output = new(new TensorShape(n, 3, size, size), DType.F32);
            float* src = (float*)latents.DataPointer, dst = (float*)output.DataPointer;
            for (int v = 0; v < n; v++)
                for (int c = 0; c < 3; c++)
                    for (int y = 0; y < size; y++)
                        for (int x = 0; x < size; x++)
                            dst[((v * 3 + c) * size + y) * size + x] = Math.Clamp(src[((v * 4 + c) * latent + y / 8) * latent + x / 8], -1f, 1f);
            return output;
        }
    }

    private sealed class RandomConditioner(int dim) : IPaintConditioner
    {
        public Tensor Tokens(IBackend backend, ReadOnlySpan<byte> rgb, int width, int height) => HunyuanPaintUnetTests.Random(77, 1, 3, dim);

        public Tensor UnconditionalTokens(IBackend backend) => new(new TensorShape(1, 3, dim), DType.F32);
    }
}

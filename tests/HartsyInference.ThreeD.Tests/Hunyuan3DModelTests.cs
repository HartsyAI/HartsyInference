using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.Vision.Dinov2;
using Xunit;

namespace HartsyInference.ThreeD.Tests;

/// <summary>CPU structural tests for the Hunyuan3D shape model + pipeline using tiny synthetic weights.
/// Validates shapes, finiteness, and end-to-end flow (image → DiT denoise → VAE occupancy → marching cubes);
/// numeric fidelity vs the real checkpoint is a separate env-gated step.</summary>
public sealed unsafe class Hunyuan3DModelTests
{
    [Fact]
    public void Dit_Forward_FiniteVelocityCorrectShape()
    {
        using IBackend cpu = new CpuBackend();
        Hunyuan3DConfig c = Hunyuan3DSyntheticWeights.TinyConfig;
        Hunyuan3DDit dit = new(c);
        dit.LoadWeights(Hunyuan3DSyntheticWeights.BuildDit(c));

        using Tensor latent = new(new TensorShape(1, c.LatentTokens, c.LatentChannels), DType.F32);
        using Tensor cond = new(new TensorShape(1, 5, c.CondDim), DType.F32);
        Fill(latent, 0.05f); Fill(cond, 0.03f);

        using Tensor v = dit.Forward(cpu, latent, cond, 0.5f);
        Assert.Equal(1, (int)v.Shape[0]);
        Assert.Equal(c.LatentTokens, (int)v.Shape[1]);
        Assert.Equal(c.LatentChannels, (int)v.Shape[2]);
        AssertAllFinite(v);
    }

    [Fact]
    public void ShapeVae_Decode_ProducesFiniteFieldOfCorrectSize()
    {
        using IBackend cpu = new CpuBackend();
        Hunyuan3DConfig c = Hunyuan3DSyntheticWeights.TinyConfig;
        Hunyuan3DShapeVae vae = new(c);
        vae.LoadWeights(Hunyuan3DSyntheticWeights.BuildVae(c));

        using Tensor latent = new(new TensorShape(1, c.LatentTokens, c.LatentChannels), DType.F32);
        Fill(latent, 0.05f);

        ScalarField3D field = vae.Decode(cpu, latent, resolution: 12, bound: 1f, chunkSize: 100);
        Assert.Equal(12 * 12 * 12, field.Values.Length);
        Assert.All(field.Values, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void Pipeline_Generate_RunsEndToEnd()
    {
        using IBackend cpu = new CpuBackend();
        Dinov2Preset dp = Hunyuan3DSyntheticWeights.TinyDino;
        Hunyuan3DConfig c = Hunyuan3DSyntheticWeights.TinyConfig;

        Dinov2VisionEncoder dino = new(dp);
        dino.LoadWeights(Hunyuan3DSyntheticWeights.BuildDino(dp));
        Hunyuan3DDit dit = new(c); dit.LoadWeights(Hunyuan3DSyntheticWeights.BuildDit(c));
        Hunyuan3DShapeVae vae = new(c); vae.LoadWeights(Hunyuan3DSyntheticWeights.BuildVae(c));

        using Hunyuan3DShapePipeline pipeline = new(cpu, dino, dit, vae, c);

        // Small conditioning image (16x16 RGB).
        byte[] img = new byte[16 * 16 * 3];
        for (int i = 0; i < img.Length; i++) img[i] = (byte)(i % 251);
        ImageTo3DRequest req = new()
        {
            ImageRgb = img, Width = 16, Height = 16,
            Steps = 2, CfgScale = 2f, Seed = 7, GridResolution = 16,
        };

        ThreeDResult result = pipeline.Generate(req);
        Assert.Equal(7, result.Seed);
        Assert.NotNull(result.Mesh);

        // The occupancy field from random weights may or may not cross the iso level; if a surface was
        // extracted, it must export to a valid GLB.
        Mesh mesh = result.Mesh!;
        if (mesh.TriangleCount > 0)
        {
            byte[] glb = GlbWriter.Write(mesh);
            Assert.True(glb.Length > 12);
        }
    }

    private static void Fill(Tensor t, float v)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = v;
    }

    private static void AssertAllFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }
}

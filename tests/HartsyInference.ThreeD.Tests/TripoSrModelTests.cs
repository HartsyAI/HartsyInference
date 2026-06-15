using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.TripoSr;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.Vision.Dinov2;
using Xunit;

namespace HartsyInference.ThreeD.Tests;

/// <summary>CPU structural tests for TripoSR (image→triplane transformer + triplane NeRF decoder + pipeline)
/// using tiny synthetic weights. Validates shapes, finiteness, and the deterministic feed-forward flow.</summary>
public sealed unsafe class TripoSrModelTests
{
    private static Dinov2Preset TinyDino => Hunyuan3DSyntheticWeights.TinyDino; // hidden 32, 2 layers, image 28/14

    private static TripoSrConfig TinyConfig => new()
    {
        TriplaneChannels = 8, TriplaneResolution = 8,
        Width = 32, Depth = 2, NumHeads = 4, ImageTokenDim = 32, MlpDim = 64,
        NerfHidden = 16, NerfLayers = 2,
        DensityThreshold = 0f, GridResolution = 12, BoundingBox = 1f,
    };

    [Fact]
    public void Transformer_ProducesTriplaneOfCorrectShape()
    {
        using IBackend cpu = new CpuBackend();
        TripoSrConfig c = TinyConfig;
        TripoSrTransformer tr = new(c);
        tr.LoadWeights(BuildTransformer(c));

        using Tensor tokens = new(new TensorShape(1, 5, c.ImageTokenDim), DType.F32);
        Fill(tokens, 0.03f);
        Triplane plane = tr.Forward(cpu, tokens);
        Assert.Equal(c.TriplaneChannels, plane.Channels);
        Assert.Equal(c.TriplaneResolution, plane.Height);
        Assert.Equal(3 * c.TriplaneChannels * c.TriplaneResolution * c.TriplaneResolution, plane.Features.Length);
        Assert.All(plane.Features, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void NerfDecoder_DensityField_FiniteAndCorrectSize()
    {
        using IBackend cpu = new CpuBackend();
        TripoSrConfig c = TinyConfig;
        TriplaneNerfDecoder dec = new(c);
        dec.LoadWeights(BuildDecoder(c));

        int n = 3 * c.TriplaneChannels * c.TriplaneResolution * c.TriplaneResolution;
        float[] feat = new float[n];
        Random r = new(5);
        for (int i = 0; i < n; i++) feat[i] = (float)(r.NextDouble() * 0.2 - 0.1);
        Triplane tri = new() { Features = feat, Channels = c.TriplaneChannels, Height = c.TriplaneResolution, Width = c.TriplaneResolution };

        ScalarField3D field = dec.DecodeDensityField(cpu, tri, resolution: 10, bound: 1f, chunkSize: 64);
        Assert.Equal(10 * 10 * 10, field.Values.Length);
        Assert.All(field.Values, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void Pipeline_Generate_RunsEndToEnd()
    {
        using IBackend cpu = new CpuBackend();
        Dinov2Preset dp = TinyDino;
        TripoSrConfig c = TinyConfig;

        Dinov2VisionEncoder dino = new(dp); dino.LoadWeights(Hunyuan3DSyntheticWeights.BuildDino(dp));
        TripoSrTransformer tr = new(c); tr.LoadWeights(BuildTransformer(c));
        TriplaneNerfDecoder dec = new(c); dec.LoadWeights(BuildDecoder(c));
        using TripoSrPipeline pipeline = new(cpu, dino, tr, dec, c);

        byte[] img = new byte[16 * 16 * 3];
        for (int i = 0; i < img.Length; i++) img[i] = (byte)(i % 251);
        ImageTo3DRequest req = new() { ImageRgb = img, Width = 16, Height = 16, GridResolution = 12 };

        ThreeDResult result = pipeline.Generate(req);
        Assert.NotNull(result.Mesh);
        if (result.Mesh!.TriangleCount > 0)
            Assert.Equal(result.Mesh.VertexCount * 3, result.Mesh.VertexColors!.Length);
    }

    private static Dictionary<string, Tensor> BuildTransformer(TripoSrConfig c)
    {
        int w = c.Width, t = c.TriplaneTokens;
        Random r = new(44);
        Dictionary<string, Tensor> wts = new()
        {
            ["triplane_pos"] = T(r, 1, t, w),
            ["image_proj.weight"] = T(r, w, c.ImageTokenDim), ["image_proj.bias"] = T(r, w),
            ["out_proj.weight"] = T(r, c.TriplaneChannels, w), ["out_proj.bias"] = T(r, c.TriplaneChannels),
        };
        for (int i = 0; i < c.Depth; i++)
        {
            string p = $"blocks.{i}";
            foreach (string a in new[] { "self_attn", "cross_attn" })
                foreach (string proj in new[] { "q", "k", "v", "o" })
                { wts[$"{p}.{a}.{proj}.weight"] = T(r, w, w); wts[$"{p}.{a}.{proj}.bias"] = T(r, w); }
            wts[$"{p}.mlp.fc1.weight"] = T(r, c.MlpDim, w); wts[$"{p}.mlp.fc1.bias"] = T(r, c.MlpDim);
            wts[$"{p}.mlp.fc2.weight"] = T(r, w, c.MlpDim); wts[$"{p}.mlp.fc2.bias"] = T(r, w);
        }
        return wts;
    }

    private static Dictionary<string, Tensor> BuildDecoder(TripoSrConfig c)
    {
        int feat = 3 * c.TriplaneChannels, hid = c.NerfHidden;
        Random r = new(55);
        Dictionary<string, Tensor> wts = new()
        {
            ["fc_in.weight"] = T(r, hid, feat), ["fc_in.bias"] = T(r, hid),
            ["fc_out.weight"] = T(r, 4, hid), ["fc_out.bias"] = T(r, 4),
        };
        for (int i = 0; i < c.NerfLayers; i++)
        { wts[$"fc_hidden.{i}.weight"] = T(r, hid, hid); wts[$"fc_hidden.{i}.bias"] = T(r, hid); }
        return wts;
    }

    private static Tensor T(Random r, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(r.NextDouble() * 0.2 - 0.1);
        return t;
    }

    private static void Fill(Tensor t, float v)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = v;
    }
}

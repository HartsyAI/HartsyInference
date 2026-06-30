using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.TripoSr;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.Vision.DinoVit;
using Xunit;

namespace HartsyInference.ThreeD.Tests;

/// <summary>CPU structural tests for TripoSR (Transformer1D backbone + triplane NeRF decoder + pipeline)
/// using tiny synthetic weights with the real checkpoint key layout. Validates shapes, finiteness, and the
/// deterministic feed-forward flow without needing the real checkpoint.</summary>
public sealed unsafe class TripoSrModelTests
{
    private static DinoViTConfig TinyDino => new()
    {
        HiddenSize = 32, NumLayers = 2, NumHeads = 4, IntermediateSize = 64,
        ImageSize = 32, NativeImageSize = 16, PatchSize = 16, LayerNormEps = 1e-12f,
    };

    private static TripoSrConfig TinyConfig => new()
    {
        TriplaneChannels = 8, PlaneSize = 4, NumChannels = 8,
        Width = 32, Depth = 2, NumHeads = 4, HeadDim = 8, CrossAttentionDim = 32, NormNumGroups = 4,
        NerfHidden = 16, NerfMidLayers = 2,
        Radius = 0.87f, DensityBias = -1.0f, DensityThreshold = 0f, GridResolution = 12,
    };

    [Fact]
    public void Transformer_ProducesTriplaneOfCorrectShape()
    {
        using IBackend cpu = new CpuBackend();
        TripoSrConfig c = TinyConfig;
        TripoSrTransformer tr = new(c);
        tr.LoadWeights(BuildTransformer(c));

        using Tensor tokens = new(new TensorShape(1, 5, c.CrossAttentionDim), DType.F32);
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

        ScalarField3D field = dec.DecodeDensityField(cpu, tri, resolution: 10, chunkSize: 64);
        Assert.Equal(10 * 10 * 10, field.Values.Length);
        Assert.All(field.Values, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void Pipeline_Generate_RunsEndToEnd()
    {
        using IBackend cpu = new CpuBackend();
        DinoViTConfig dp = TinyDino;
        TripoSrConfig c = TinyConfig;

        DinoViTEncoder dino = new(dp); dino.LoadWeights(BuildDino(dp));
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

    private static Dictionary<string, Tensor> BuildDino(DinoViTConfig c)
    {
        int h = c.HiddenSize, inter = c.IntermediateSize, np = c.NativePatchGrid * c.NativePatchGrid;
        Random r = new(33);
        Dictionary<string, Tensor> w = new()
        {
            ["embeddings.cls_token"] = T(r, 1, 1, h),
            ["embeddings.position_embeddings"] = T(r, 1, 1 + np, h),
            ["embeddings.patch_embeddings.projection.weight"] = T(r, h, 3, c.PatchSize, c.PatchSize),
            ["embeddings.patch_embeddings.projection.bias"] = T(r, h),
            ["layernorm.weight"] = T(r, h), ["layernorm.bias"] = T(r, h),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"encoder.layer.{i}";
            w[$"{p}.layernorm_before.weight"] = T(r, h); w[$"{p}.layernorm_before.bias"] = T(r, h);
            foreach (string proj in new[] { "query", "key", "value" })
            { w[$"{p}.attention.attention.{proj}.weight"] = T(r, h, h); w[$"{p}.attention.attention.{proj}.bias"] = T(r, h); }
            w[$"{p}.attention.output.dense.weight"] = T(r, h, h); w[$"{p}.attention.output.dense.bias"] = T(r, h);
            w[$"{p}.layernorm_after.weight"] = T(r, h); w[$"{p}.layernorm_after.bias"] = T(r, h);
            w[$"{p}.intermediate.dense.weight"] = T(r, inter, h); w[$"{p}.intermediate.dense.bias"] = T(r, inter);
            w[$"{p}.output.dense.weight"] = T(r, h, inter); w[$"{p}.output.dense.bias"] = T(r, h);
        }
        return w;
    }

    private static Dictionary<string, Tensor> BuildTransformer(TripoSrConfig c)
    {
        int w = c.Width, cc = c.NumChannels, ps = c.PlaneSize, inner = 4 * w;
        Random r = new(44);
        Dictionary<string, Tensor> wts = new()
        {
            ["tokenizer.embeddings"] = T(r, 3, cc, ps, ps),
            ["backbone.norm.weight"] = T(r, cc), ["backbone.norm.bias"] = T(r, cc),
            ["backbone.proj_in.weight"] = T(r, w, cc), ["backbone.proj_in.bias"] = T(r, w),
            ["backbone.proj_out.weight"] = T(r, cc, w), ["backbone.proj_out.bias"] = T(r, cc),
            ["post_processor.upsample.weight"] = T(r, cc, c.TriplaneChannels, 2, 2),
            ["post_processor.upsample.bias"] = T(r, c.TriplaneChannels),
        };
        for (int i = 0; i < c.Depth; i++)
        {
            string p = $"backbone.transformer_blocks.{i}";
            wts[$"{p}.norm1.weight"] = T(r, w); wts[$"{p}.norm1.bias"] = T(r, w);
            wts[$"{p}.attn1.to_q.weight"] = T(r, w, w); wts[$"{p}.attn1.to_k.weight"] = T(r, w, w); wts[$"{p}.attn1.to_v.weight"] = T(r, w, w);
            wts[$"{p}.attn1.to_out.0.weight"] = T(r, w, w); wts[$"{p}.attn1.to_out.0.bias"] = T(r, w);
            wts[$"{p}.norm2.weight"] = T(r, w); wts[$"{p}.norm2.bias"] = T(r, w);
            wts[$"{p}.attn2.to_q.weight"] = T(r, w, w); wts[$"{p}.attn2.to_k.weight"] = T(r, w, c.CrossAttentionDim); wts[$"{p}.attn2.to_v.weight"] = T(r, w, c.CrossAttentionDim);
            wts[$"{p}.attn2.to_out.0.weight"] = T(r, w, w); wts[$"{p}.attn2.to_out.0.bias"] = T(r, w);
            wts[$"{p}.norm3.weight"] = T(r, w); wts[$"{p}.norm3.bias"] = T(r, w);
            wts[$"{p}.ff.net.0.proj.weight"] = T(r, 2 * inner, w); wts[$"{p}.ff.net.0.proj.bias"] = T(r, 2 * inner);
            wts[$"{p}.ff.net.2.weight"] = T(r, w, inner); wts[$"{p}.ff.net.2.bias"] = T(r, w);
        }
        return wts;
    }

    private static Dictionary<string, Tensor> BuildDecoder(TripoSrConfig c)
    {
        int feat = 3 * c.TriplaneChannels, hid = c.NerfHidden;
        Random r = new(55);
        Dictionary<string, Tensor> wts = new()
        {
            ["layers.0.weight"] = T(r, hid, feat), ["layers.0.bias"] = T(r, hid),
        };
        for (int i = 0; i < c.NerfMidLayers; i++)
        {
            int idx = 2 * (i + 1);
            wts[$"layers.{idx}.weight"] = T(r, hid, hid); wts[$"layers.{idx}.bias"] = T(r, hid);
        }
        int outIdx = 2 * (c.NerfMidLayers + 1);
        wts[$"layers.{outIdx}.weight"] = T(r, 4, hid); wts[$"layers.{outIdx}.bias"] = T(r, 4);
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

using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.Gguf;
using Xunit;

namespace SharpInference.ModelHandler.Tests;

/// <summary>End-to-end GGUF round-trip tests. Use <see cref="GgufWriter"/> to build a synthetic GGUF file, then load it via <see cref="GgufModelLoader"/> and verify metadata, descriptors, tensor data, key remap, and architecture detection all flow correctly. These exercise every layer of the new GGUF backend (writer + loader + codec registry + key-mapper registry).</summary>
public sealed class GgufRoundTripTests : IDisposable
{
    private readonly string _tempDir;

    public GgufRoundTripTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"sharpinf-gguf-roundtrip-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public unsafe void RoundTrip_SimpleF32_PreservesShapeAndData()
    {
        string path = Path.Combine(_tempDir, "simple.gguf");

        Tensor src = new Tensor(new TensorShape(2, 3), DType.F32);
        try
        {
            float* sp = (float*)src.DataPointer;
            for (int i = 0; i < 6; i++) sp[i] = i * 1.5f;

            using (GgufWriter w = new(path))
            {
                w.SetMetadata("general.architecture", "test");
                w.AddTensor("test.tensor", src);
                w.Flush();
            }

            using GgufModelLoader.LoadedGgufModel loaded = GgufModelLoader.Load(path);
            Assert.Equal("passthrough", loaded.Architecture);
            Assert.Equal("test", loaded.Metadata.GetString("general.architecture"));
            Assert.Single(loaded.Weights);
            Assert.True(loaded.Weights.ContainsKey("test.tensor"));

            Tensor dst = loaded.Weights["test.tensor"];
            Assert.Equal(DType.F32, dst.DType);
            Assert.Equal(2L, dst.Shape[0]);
            Assert.Equal(3L, dst.Shape[1]);

            float* dp = (float*)dst.DataPointer;
            for (int i = 0; i < 6; i++) Assert.Equal(i * 1.5f, dp[i]);
        }
        finally
        {
            src.Dispose();
        }
    }

    [Fact]
    public unsafe void RoundTrip_Q8_0_PreservesQuantizedDataAndDequantizesCorrectly()
    {
        string path = Path.Combine(_tempDir, "q8_0.gguf");

        Tensor q8Src = new Tensor(new TensorShape(32), DType.Q8_0);
        try
        {
            byte* p = (byte*)q8Src.DataPointer;
            *(Half*)p = (Half)0.25f;
            sbyte* qd = (sbyte*)(p + 2);
            for (int i = 0; i < 32; i++) qd[i] = (sbyte)(i - 16);

            using (GgufWriter w = new(path))
            {
                w.SetMetadata("general.architecture", "test");
                w.AddTensor("layer.weight", q8Src);
                w.Flush();
            }

            using GgufModelLoader.LoadedGgufModel loaded = GgufModelLoader.Load(path);
            Tensor dst = loaded.Weights["layer.weight"];
            Assert.Equal(DType.Q8_0, dst.DType);

            using Tensor dequant = GgufDequantizer.Dequantize(dst, DType.F32);
            float* dp = (float*)dequant.DataPointer;
            for (int i = 0; i < 32; i++)
            {
                float expected = 0.25f * (i - 16);
                Assert.True(MathF.Abs(dp[i] - expected) < 1e-3f, $"i={i}: expected {expected}, got {dp[i]}");
            }
        }
        finally
        {
            q8Src.Dispose();
        }
    }

    [Fact]
    public unsafe void RoundTrip_FluxArchitecture_DetectedFromMetadata()
    {
        string path = Path.Combine(_tempDir, "flux.gguf");

        Tensor t = new Tensor(new TensorShape(8), DType.F32);
        try
        {
            using (GgufWriter w = new(path))
            {
                w.SetMetadata("general.architecture", "flux");
                w.SetMetadata("flux.depth", (uint)19);
                w.AddTensor("model.diffusion_model.double_blocks.0.img_attn.qkv.weight", t);
                w.AddTensor("model.diffusion_model.single_blocks.0.linear1.weight", t);
                w.Flush();
            }

            using GgufModelLoader.LoadedGgufModel loaded = GgufModelLoader.Load(path);
            Assert.Equal("flux", loaded.Architecture);
            Assert.Equal((uint)19, loaded.Metadata.GetUInt32("flux.depth"));
            Assert.Equal(2, loaded.Weights.Count);
            Assert.True(loaded.Weights.ContainsKey("model.diffusion_model.double_blocks.0.img_attn.qkv.weight"));
        }
        finally
        {
            t.Dispose();
        }
    }

    [Fact]
    public unsafe void RoundTrip_DetectsArchitectureByKeysWhenMetadataMissing()
    {
        string path = Path.Combine(_tempDir, "auraflow.gguf");

        Tensor t = new Tensor(new TensorShape(8), DType.F32);
        try
        {
            using (GgufWriter w = new(path))
            {
                w.AddTensor("double_layers.0.attn.w2q.weight", t);
                w.AddTensor("modF.1.weight", t);
                w.Flush();
            }

            using GgufModelLoader.LoadedGgufModel loaded = GgufModelLoader.Load(path);
            Assert.Equal("auraflow", loaded.Architecture);
        }
        finally
        {
            t.Dispose();
        }
    }

    [Fact]
    public unsafe void LlamaKeyMapper_RemapsBlockKeys()
    {
        string path = Path.Combine(_tempDir, "llama.gguf");

        Tensor t = new Tensor(new TensorShape(8), DType.F32);
        try
        {
            using (GgufWriter w = new(path))
            {
                w.SetMetadata("general.architecture", "llama");
                w.AddTensor("token_embd.weight", t);
                w.AddTensor("output_norm.weight", t);
                w.AddTensor("blk.0.attn_q.weight", t);
                w.AddTensor("blk.0.attn_k.weight", t);
                w.AddTensor("blk.0.attn_v.weight", t);
                w.AddTensor("blk.0.attn_output.weight", t);
                w.AddTensor("blk.0.ffn_gate.weight", t);
                w.AddTensor("blk.0.ffn_up.weight", t);
                w.AddTensor("blk.0.ffn_down.weight", t);
                w.AddTensor("blk.0.attn_norm.weight", t);
                w.AddTensor("blk.0.ffn_norm.weight", t);
                w.Flush();
            }

            using GgufModelLoader.LoadedGgufModel loaded = GgufModelLoader.Load(path);
            Assert.Equal("llama", loaded.Architecture);
            Assert.True(loaded.Weights.ContainsKey("model.embed_tokens.weight"));
            Assert.True(loaded.Weights.ContainsKey("model.norm.weight"));
            Assert.True(loaded.Weights.ContainsKey("model.layers.0.self_attn.q_proj.weight"));
            Assert.True(loaded.Weights.ContainsKey("model.layers.0.self_attn.k_proj.weight"));
            Assert.True(loaded.Weights.ContainsKey("model.layers.0.self_attn.v_proj.weight"));
            Assert.True(loaded.Weights.ContainsKey("model.layers.0.self_attn.o_proj.weight"));
            Assert.True(loaded.Weights.ContainsKey("model.layers.0.mlp.gate_proj.weight"));
            Assert.True(loaded.Weights.ContainsKey("model.layers.0.mlp.up_proj.weight"));
            Assert.True(loaded.Weights.ContainsKey("model.layers.0.mlp.down_proj.weight"));
            Assert.True(loaded.Weights.ContainsKey("model.layers.0.input_layernorm.weight"));
            Assert.True(loaded.Weights.ContainsKey("model.layers.0.post_attention_layernorm.weight"));
            Assert.False(loaded.Weights.ContainsKey("blk.0.attn_q.weight"));
        }
        finally
        {
            t.Dispose();
        }
    }

    [Fact]
    public unsafe void RoundTrip_MultipleTensorsWithAlignment_RecoversAllData()
    {
        string path = Path.Combine(_tempDir, "multi.gguf");

        Tensor a = new Tensor(new TensorShape(7), DType.F32);
        Tensor b = new Tensor(new TensorShape(13), DType.F32);
        Tensor c = new Tensor(new TensorShape(31), DType.F32);
        try
        {
            float* ap = (float*)a.DataPointer;
            float* bp = (float*)b.DataPointer;
            float* cp = (float*)c.DataPointer;
            for (int i = 0; i < 7; i++) ap[i] = i * 1.0f;
            for (int i = 0; i < 13; i++) bp[i] = i * 2.0f;
            for (int i = 0; i < 31; i++) cp[i] = i * 3.0f;

            using (GgufWriter w = new(path))
            {
                w.SetMetadata("general.architecture", "test");
                w.AddTensor("a", a);
                w.AddTensor("b", b);
                w.AddTensor("c", c);
                w.Flush();
            }

            using GgufModelLoader.LoadedGgufModel loaded = GgufModelLoader.Load(path);
            Assert.Equal(3, loaded.Weights.Count);

            float* dap = (float*)loaded.Weights["a"].DataPointer;
            float* dbp = (float*)loaded.Weights["b"].DataPointer;
            float* dcp = (float*)loaded.Weights["c"].DataPointer;
            for (int i = 0; i < 7; i++) Assert.Equal(i * 1.0f, dap[i]);
            for (int i = 0; i < 13; i++) Assert.Equal(i * 2.0f, dbp[i]);
            for (int i = 0; i < 31; i++) Assert.Equal(i * 3.0f, dcp[i]);
        }
        finally
        {
            a.Dispose();
            b.Dispose();
            c.Dispose();
        }
    }

    [Fact]
    public unsafe void LoadDequantized_ConvertsQuantizedToF16()
    {
        string path = Path.Combine(_tempDir, "dequant.gguf");

        Tensor q = new Tensor(new TensorShape(32), DType.Q8_0);
        try
        {
            byte* p = (byte*)q.DataPointer;
            *(Half*)p = (Half)1.0f;
            sbyte* qd = (sbyte*)(p + 2);
            for (int i = 0; i < 32; i++) qd[i] = (sbyte)i;

            using (GgufWriter w = new(path))
            {
                w.SetMetadata("general.architecture", "test");
                w.AddTensor("layer.weight", q);
                w.Flush();
            }

            (Dictionary<string, Tensor> weights, GgufModelLoader.LoadedGgufModel handle) =
                GgufModelLoader.LoadDequantized(path, DType.F16);
            using (handle)
            {
                Tensor t = weights["layer.weight"];
                Assert.Equal(DType.F16, t.DType);
                Half* hp = (Half*)t.DataPointer;
                for (int i = 0; i < 32; i++)
                {
                    Assert.True(MathF.Abs((float)hp[i] - i) < 0.1f, $"i={i}: expected ~{i}, got {(float)hp[i]}");
                }

                foreach (Tensor t2 in weights.Values) t2.Dispose();
            }
        }
        finally
        {
            q.Dispose();
        }
    }

    [Fact]
    public unsafe void GgufConverterBridge_LoadsThroughTestConverter()
    {
        string path = Path.Combine(_tempDir, "bridge.gguf");

        Tensor t = new Tensor(new TensorShape(8), DType.F32);
        try
        {
            float* tp = (float*)t.DataPointer;
            for (int i = 0; i < 8; i++) tp[i] = i * 0.5f;

            using (GgufWriter w = new(path))
            {
                w.SetMetadata("general.architecture", "test");
                w.AddTensor("foo", t);
                w.AddTensor("bar", t);
                w.Flush();
            }

            (TestConverted converted, GgufModelLoader.LoadedGgufModel handle) =
                GgufConverterBridge.LoadGguf(path, DType.F32, weights => new TestConverted { KeyCount = weights.Count });
            using (handle)
            {
                Assert.Equal(2, converted.KeyCount);
            }
        }
        finally
        {
            t.Dispose();
        }
    }

    private sealed class TestConverted { public int KeyCount { get; init; } }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, true);
        }
        catch { }
    }
}

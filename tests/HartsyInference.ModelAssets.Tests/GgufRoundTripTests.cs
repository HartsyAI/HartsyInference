using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>End-to-end GGUF round-trip tests. Use <see cref="GgufWriter"/> to build a synthetic GGUF file, then load it via <see cref="GgufModelLoader"/> and verify metadata, descriptors, tensor data, key remap, and architecture detection all flow correctly. These exercise every layer of the new GGUF backend (writer + loader + codec registry + key-mapper registry).</summary>
public sealed class GgufRoundTripTests : IDisposable
{
    private readonly string _tempDir;

    public GgufRoundTripTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"sharpinf-gguf-roundtrip-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    /// <summary>The writer emits ggml <c>ne</c> order — fastest axis first — so a rank-2 tensor comes back from the
    /// loader transposed and is put right by <see cref="GgufModelLoader.RelabelRank2ToPyTorchOrder"/>, exactly as a
    /// city96 or llama.cpp file is. Asserting the shape survives the raw load instead would pin the old behaviour,
    /// where our output was self-consistent, unreadable by every other GGUF tool, and transposed by the relabel every
    /// read path in this engine applies.</summary>
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
            // Architecture now reports the real GGUF arch ("test"); the mapper that handled the
            // unknown arch is exposed separately as MapperName ("passthrough").
            Assert.Equal("test", loaded.Architecture);
            Assert.Equal("passthrough", loaded.MapperName);
            Assert.Equal("test", loaded.Metadata.GetString("general.architecture"));
            Assert.Single(loaded.Weights);
            Assert.True(loaded.Weights.ContainsKey("test.tensor"));

            Tensor raw = loaded.Weights["test.tensor"];
            Assert.Equal(DType.F32, raw.DType);
            Assert.Equal(3L, raw.Shape[0]);
            Assert.Equal(2L, raw.Shape[1]);

            Tensor dst = GgufModelLoader.RelabelRank2ToPyTorchOrder(loaded.Weights)["test.tensor"];
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

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, true);
        }
        catch { }
    }
}

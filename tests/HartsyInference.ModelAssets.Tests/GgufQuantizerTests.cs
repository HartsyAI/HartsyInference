using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>End-to-end Phase C tests: build a synthetic in-memory tensor dict, run <see cref="GgufQuantizer"/> with various policies, load back via <see cref="GgufModelLoader"/>, verify round-trip.</summary>
public sealed class GgufQuantizerTests : IDisposable
{
    private readonly string _tempDir;

    public GgufQuantizerTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"sharpinf-quantizer-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public unsafe void Q8_0_Policy_LargeWeightsQuantized_NormsKeptF16()
    {
        string path = Path.Combine(_tempDir, "q8.gguf");
        Dictionary<string, Tensor> tensors = BuildSyntheticDict();
        try
        {
            GgufQuantizationReport report = GgufQuantizer.ConvertDictionaryToGguf(
                tensors, path, GgufQuantPolicy.Q8_0, architecture: "test_arch");

            Assert.True(report.QuantizedCount > 0, "expected at least one quantized tensor");
            Assert.True(report.CastCount > 0, "expected at least one F16-cast tensor (norms)");
            Assert.True(report.OutputBytes > 0);

            using GgufModelLoader.LoadedGgufModel loaded = GgufModelLoader.Load(path);
            Tensor weight = loaded.Weights["layer.0.attn_q.weight"];
            Tensor norm = loaded.Weights["layer.0.input_norm.weight"];
            Tensor bias = loaded.Weights["layer.0.attn_q.bias"];

            Assert.Equal(DType.Q8_0, weight.DType);
            Assert.Equal(DType.F16, norm.DType);
            Assert.Equal(DType.F16, bias.DType);
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    [Fact]
    public unsafe void Q4_K_M_Policy_OutputProjGetsQ6K()
    {
        string path = Path.Combine(_tempDir, "q4km.gguf");
        Dictionary<string, Tensor> tensors = BuildSyntheticDict();
        try
        {
            GgufQuantizer.ConvertDictionaryToGguf(
                tensors, path, GgufQuantPolicy.Q4_K_M, architecture: "test_arch");

            using GgufModelLoader.LoadedGgufModel loaded = GgufModelLoader.Load(path);
            Tensor q = loaded.Weights["layer.0.attn_q.weight"];
            Tensor v = loaded.Weights["layer.0.attn_v.weight"];

            Assert.Equal(DType.Q4_K, q.DType);
            Assert.Equal(DType.Q6_K, v.DType);
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    [Fact]
    public unsafe void EndToEnd_QuantizeLoadDequantize_RecoversApproximateValues()
    {
        string path = Path.Combine(_tempDir, "e2e.gguf");

        Tensor src = new Tensor(new TensorShape(256, 256), DType.F32);
        try
        {
            float* sp = (float*)src.DataPointer;
            Random rng = new Random(42);
            for (int i = 0; i < 256 * 256; i++) sp[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            Dictionary<string, Tensor> tensors = new() { ["layer.0.linear.weight"] = src };
            GgufQuantizer.ConvertDictionaryToGguf(
                tensors, path, GgufQuantPolicy.Q8_0, architecture: "test_arch");

            (Dictionary<string, Tensor> loaded, GgufModelLoader.LoadedGgufModel handle) =
                GgufModelLoader.LoadDequantized(path, DType.F32);
            using (handle)
            {
                Tensor recovered = loaded["layer.0.linear.weight"];
                Assert.Equal(DType.F32, recovered.DType);

                float* rp = (float*)recovered.DataPointer;
                float sumSqErr = 0f;
                for (int i = 0; i < 256 * 256; i++)
                {
                    float err = rp[i] - sp[i];
                    sumSqErr += err * err;
                }
                float rmse = MathF.Sqrt(sumSqErr / (256 * 256));
                Assert.True(rmse < 0.005f, $"Q8_0 round-trip RMSE {rmse:F4} too large");

                foreach (Tensor t in loaded.Values) t.Dispose();
            }
        }
        finally
        {
            src.Dispose();
        }
    }

    [Fact]
    public unsafe void EndToEnd_Q4KM_RecoversApproximateValues()
    {
        string path = Path.Combine(_tempDir, "e2e_q4km.gguf");

        Tensor src = new Tensor(new TensorShape(256, 256), DType.F32);
        try
        {
            float* sp = (float*)src.DataPointer;
            Random rng = new Random(7);
            for (int i = 0; i < 256 * 256; i++) sp[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            Dictionary<string, Tensor> tensors = new() { ["layer.0.linear.weight"] = src };
            GgufQuantizer.ConvertDictionaryToGguf(
                tensors, path, GgufQuantPolicy.Q4_K_M, architecture: "test_arch");

            (Dictionary<string, Tensor> loaded, GgufModelLoader.LoadedGgufModel handle) =
                GgufModelLoader.LoadDequantized(path, DType.F32);
            using (handle)
            {
                float* rp = (float*)loaded["layer.0.linear.weight"].DataPointer;
                float sumSqErr = 0f;
                for (int i = 0; i < 256 * 256; i++)
                {
                    float err = rp[i] - sp[i];
                    sumSqErr += err * err;
                }
                float rmse = MathF.Sqrt(sumSqErr / (256 * 256));
                Assert.True(rmse < 0.05f, $"Q4_K round-trip RMSE {rmse:F4} too large");

                foreach (Tensor t in loaded.Values) t.Dispose();
            }
        }
        finally
        {
            src.Dispose();
        }
    }

    [Fact]
    public unsafe void Q5_K_M_PolicyAppliesAcrossKnownTensorPatterns()
    {
        string path = Path.Combine(_tempDir, "q5km.gguf");
        Dictionary<string, Tensor> tensors = BuildSyntheticDict();
        try
        {
            GgufQuantizer.ConvertDictionaryToGguf(
                tensors, path, GgufQuantPolicy.Q5_K_M, architecture: "test_arch");

            using GgufModelLoader.LoadedGgufModel loaded = GgufModelLoader.Load(path);
            Assert.Equal(DType.Q5_K, loaded.Weights["layer.0.attn_q.weight"].DType);
            Assert.Equal(DType.Q6_K, loaded.Weights["layer.0.attn_v.weight"].DType);
            Assert.Equal(DType.F16, loaded.Weights["layer.0.input_norm.weight"].DType);
        }
        finally
        {
            foreach (Tensor t in tensors.Values) t.Dispose();
        }
    }

    private unsafe Dictionary<string, Tensor> BuildSyntheticDict()
    {
        Dictionary<string, Tensor> tensors = new();
        Random rng = new Random(123);

        Tensor q = new Tensor(new TensorShape(256, 256), DType.F32);
        Tensor k = new Tensor(new TensorShape(256, 256), DType.F32);
        Tensor v = new Tensor(new TensorShape(256, 256), DType.F32);
        Tensor norm = new Tensor(new TensorShape(256), DType.F32);
        Tensor bias = new Tensor(new TensorShape(256), DType.F32);
        Fill(q, rng);
        Fill(k, rng);
        Fill(v, rng);
        Fill(norm, rng);
        Fill(bias, rng);

        tensors["layer.0.attn_q.weight"] = q;
        tensors["layer.0.attn_k.weight"] = k;
        tensors["layer.0.attn_v.weight"] = v;
        tensors["layer.0.input_norm.weight"] = norm;
        tensors["layer.0.attn_q.bias"] = bias;
        return tensors;
    }

    private static unsafe void Fill(Tensor t, Random rng)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
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

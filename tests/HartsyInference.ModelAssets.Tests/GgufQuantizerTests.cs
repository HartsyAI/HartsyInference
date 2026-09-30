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

    /// <summary>A quant cache read back through <see cref="GgufQuantizer.ReadBack"/> carries its source shapes,
    /// whichever axis order the file was written in. The writer emits ggml order, so a raw read hands a
    /// <c>[256, 512]</c> projection over as <c>[512, 256]</c>; a cache written before the writer changed is in the
    /// engine's order already. Both come back as the dictionary they were made from.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public unsafe void ReadBack_RestoresSourceShapes(bool legacyFile)
    {
        string path = Path.Combine(_tempDir, legacyFile ? "legacy.gguf" : "ne.gguf");
        Random rng = new Random(7);
        Dictionary<string, Tensor> source = new()
        {
            ["down_proj.weight"] = new Tensor(new TensorShape(256, 512), DType.F32),
            ["head.weight"] = new Tensor(new TensorShape(512, 256), DType.F32),
            ["square.weight"] = new Tensor(new TensorShape(256, 256), DType.F32),
            ["norm.weight"] = new Tensor(new TensorShape(256), DType.F32),
        };
        foreach (Tensor t in source.Values) Fill(t, rng);
        try
        {
            // The old writer emitted the engine's order verbatim, which is what the current writer produces for a
            // tensor whose shape is already reversed - so a legacy file is written from transposed views.
            Dictionary<string, Tensor> toWrite = new();
            foreach (KeyValuePair<string, Tensor> kv in source)
            {
                toWrite[kv.Key] = legacyFile && kv.Value.Shape.Rank == 2
                    ? kv.Value.Reshape(new TensorShape(kv.Value.Shape[1], kv.Value.Shape[0]))
                    : kv.Value;
            }
            GgufQuantizer.ConvertDictionaryToGguf(toWrite, path, GgufQuantPolicy.Q8_0, architecture: "test_arch");

            using GgufLoader loader = new();
            loader.Load(path);
            Tensor raw = loader.GetTensor("down_proj.weight");
            Assert.Equal(legacyFile ? new long[] { 256, 512 } : new long[] { 512, 256 }, Dims(raw.Shape));

            Dictionary<string, Tensor> weights = GgufQuantizer.ReadBack(loader, source);
            Assert.Equal(new long[] { 256, 512 }, Dims(weights["down_proj.weight"].Shape));
            Assert.Equal(new long[] { 512, 256 }, Dims(weights["head.weight"].Shape));
            Assert.Equal(new long[] { 256, 256 }, Dims(weights["square.weight"].Shape));
            Assert.Equal(new long[] { 256 }, Dims(weights["norm.weight"].Shape));
            Assert.Equal(DType.Q8_0, weights["down_proj.weight"].DType);
            Assert.Equal(DType.F16, weights["norm.weight"].DType);

            // The bytes are row-major in the engine's order either way: the relabelled projection dequantizes to
            // the source, row for row.
            Tensor back = GgufDequantizer.Dequantize(weights["down_proj.weight"], DType.F32);
            try
            {
                float* expected = (float*)source["down_proj.weight"].DataPointer;
                float* actual = (float*)back.DataPointer;
                for (long i = 0; i < back.ElementCount; i += 97)
                {
                    Assert.InRange(actual[i], expected[i] - 0.02f, expected[i] + 0.02f);
                }
            }
            finally
            {
                back.Dispose();
            }

            // A tensor the caller no longer has a source for is assumed to be in the file's ggml order.
            Dictionary<string, Tensor> partial = new() { ["norm.weight"] = source["norm.weight"] };
            Dictionary<string, Tensor> guessed = GgufQuantizer.ReadBack(loader, partial);
            Assert.Equal(legacyFile ? new long[] { 512, 256 } : new long[] { 256, 512 }, Dims(guessed["down_proj.weight"].Shape));
        }
        finally
        {
            foreach (Tensor t in source.Values) t.Dispose();
        }
    }

    [Fact]
    public unsafe void ReadBack_RejectsACacheWrittenFromAnotherDictionary()
    {
        string path = Path.Combine(_tempDir, "other.gguf");
        Dictionary<string, Tensor> written = new() { ["w.weight"] = new Tensor(new TensorShape(256, 256), DType.F32) };
        Dictionary<string, Tensor> other = new() { ["w.weight"] = new Tensor(new TensorShape(256, 512), DType.F32) };
        try
        {
            Fill(written["w.weight"], new Random(1));
            GgufQuantizer.ConvertDictionaryToGguf(written, path, GgufQuantPolicy.Q8_0, architecture: "test_arch");
            using GgufLoader loader = new();
            loader.Load(path);
            Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => GgufQuantizer.ReadBack(loader, other));
        }
        finally
        {
            written["w.weight"].Dispose();
            other["w.weight"].Dispose();
        }
    }

    private static long[] Dims(TensorShape shape)
    {
        long[] dims = new long[shape.Rank];
        for (int i = 0; i < shape.Rank; i++) dims[i] = shape[i];
        return dims;
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

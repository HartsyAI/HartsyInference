using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Csm;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Validates the disk-cached CSM/HeartMuLa quantization pipeline (<see cref="CsmWeightCache.LoadQuantized"/>):
/// convert a small weight dict → quantized GGUF → load back, and check the policy — projection/head matrices become
/// quantized (fused-GEMV bandwidth win), while embed tables + norms stay F16 (host gather / vector ops). Also checks
/// the cache is reused on a second call and shapes/keys are preserved. Tiny tensors → runs anywhere, no OOM.</summary>
public sealed unsafe class CsmWeightCacheTests
{
    private readonly ITestOutputHelper _out;
    public CsmWeightCacheTests(ITestOutputHelper o) => _out = o;

    private static uint _rng = 0x51A7u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.1f; }
    private static Tensor F(params int[] dims)
    {
        long[] longDims = Array.ConvertAll(dims, d => (long)d);
        Tensor t = new(new TensorShape(longDims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = Rand();
        return t;
    }

    [Theory]
    [InlineData("q8_0", "Q8_0")]
    [InlineData("q4_k", "Q4_K")]
    public void LoadQuantized_QuantizesProjectionsAndHeads_KeepsEmbedsAndNorms(string mode, string expectQuant)
    {
        // Small CSM-shaped dict: 2 projection matrices, a head, the backbone→decoder projection, an embed table, a norm.
        Dictionary<string, Tensor> src = new()
        {
            ["backbone.layers.0.self_attn.q_proj.weight"] = F(256, 256),
            ["backbone.layers.0.mlp.down_proj.weight"] = F(256, 512),
            ["codebook0_head.weight"] = F(512, 256),
            ["projection.weight"] = F(256, 256),
            ["text_embeddings.weight"] = F(100, 256),
            ["audio_embeddings.0.weight"] = F(64, 256),
            ["backbone.norm.weight"] = F(256),
            ["muq_linear.weight"] = F(256, 512),
        };

        string dir = Path.Combine(Path.GetTempPath(), "csmqtest_" + mode);
        string gguf = Path.Combine(dir, $"heartmula_{mode}.gguf");
        try
        {
            if (File.Exists(gguf)) File.Delete(gguf);

            (IReadOnlyDictionary<string, Tensor> w, GgufLoader? loader) = CsmWeightCache.LoadQuantized(src, gguf, mode);
            Assert.NotNull(loader);
            Assert.True(File.Exists(gguf), "quantized GGUF cache should have been written");

            // Projections + head → quantized; embeds + norm + muq → kept F16; all keys + shapes preserved.
            void Check(string key, string expectDtype, int[] shape)
            {
                Assert.True(w.ContainsKey(key), $"missing key {key}");
                Tensor t = w[key];
                _out.WriteLine($"{key}: {t.DType.Name}  shape=[{string.Join(",", ToInts(t.Shape))}]");
                Assert.Equal(expectDtype, t.DType.Name);
                Assert.Equal(shape, ToInts(t.Shape));
            }
            Check("backbone.layers.0.self_attn.q_proj.weight", expectQuant, new[] { 256, 256 });
            Check("backbone.layers.0.mlp.down_proj.weight", expectQuant, new[] { 256, 512 });
            Check("codebook0_head.weight", expectQuant, new[] { 512, 256 });
            Check("projection.weight", expectQuant, new[] { 256, 256 });
            Check("text_embeddings.weight", "F16", new[] { 100, 256 });
            Check("audio_embeddings.0.weight", "F16", new[] { 64, 256 });
            Check("backbone.norm.weight", "F16", new[] { 256 });
            Check("muq_linear.weight", "F16", new[] { 256, 512 });

            long firstWrite = new FileInfo(gguf).LastWriteTimeUtc.Ticks;
            loader!.Dispose();

            // Second call must REUSE the cache (no reconvert → same file, same mtime).
            (IReadOnlyDictionary<string, Tensor> w2, GgufLoader? loader2) = CsmWeightCache.LoadQuantized(src, gguf, mode);
            Assert.Equal(firstWrite, new FileInfo(gguf).LastWriteTimeUtc.Ticks);
            Assert.Equal(expectQuant, w2["projection.weight"].DType.Name);
            loader2!.Dispose();

            foreach (Tensor t in src.Values) t.Dispose();
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { /* best-effort */ }
        }
    }

    [Fact]
    public void LoadQuantized_UnsetMode_ReturnsSourceUnchanged()
    {
        Dictionary<string, Tensor> src = new() { ["projection.weight"] = F(256, 256) };
        (IReadOnlyDictionary<string, Tensor> w, GgufLoader? loader) = CsmWeightCache.LoadQuantized(src, "/tmp/should_not_exist.gguf", null);
        Assert.Null(loader);
        Assert.Same(src, w);
        Assert.False(File.Exists("/tmp/should_not_exist.gguf"));
        foreach (Tensor t in src.Values) t.Dispose();
    }

    private static int[] ToInts(TensorShape s)
    {
        int[] d = new int[s.Rank];
        for (int i = 0; i < s.Rank; i++) d[i] = (int)s[i];
        return d;
    }
}

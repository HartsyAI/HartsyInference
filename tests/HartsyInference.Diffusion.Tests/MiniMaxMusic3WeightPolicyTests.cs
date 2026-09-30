using HartsyInference.Audio.Cache;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Locks the shape contract of MiniMax Music 3's disk-cached quantization: what
/// <see cref="MiniMaxMusic3WeightPolicy"/> hands the depth decoder under <c>:q8</c>/<c>:q4</c> must carry the shapes
/// of the weights it was made from. The cache is a GGUF in ggml axis order, and reading it raw handed every projection
/// over transposed — a <c>[out, in]</c> matmul weight became <c>[in, out]</c>, the backend derived <c>M = 0</c>
/// from it and the first CUDA launch failed with an invalid grid. Tiny tensors, one temporary cache file, no card.</summary>
public sealed unsafe class MiniMaxMusic3WeightPolicyTests
{
    [Theory]
    [InlineData("q8")]
    [InlineData("q4")]
    public void PrepareDepthDecoder_KeepsTheSourceShapes(string quant)
    {
        string repo = $"hartsy-tests/minimax-readback-{Guid.NewGuid():N}";
        string cacheDir = Path.Combine(AudioModelCache.CacheRoot, "music", "_gguf-cache");
        Random rng = new Random(11);
        Dictionary<string, Tensor> source = new()
        {
            ["layers.0.attn.to_q.weight"] = new Tensor(new TensorShape(256, 256), DType.F32),
            ["layers.0.attn.to_v.weight"] = new Tensor(new TensorShape(256, 256), DType.F32),
            ["layers.0.mlp.down_proj.weight"] = new Tensor(new TensorShape(256, 512), DType.F32),
            ["layers.0.mlp.up_proj.weight"] = new Tensor(new TensorShape(512, 256), DType.F32),
            ["audio_heads.0.weight"] = new Tensor(new TensorShape(1024, 256), DType.F32),
            ["layers.0.norm.weight"] = new Tensor(new TensorShape(256), DType.F32),
        };
        foreach (Tensor t in source.Values) Fill(t, rng);
        IDisposable? cache = null;
        try
        {
            IReadOnlyDictionary<string, Tensor> prepared = MiniMaxMusic3WeightPolicy.PrepareDepthDecoder(source, repo, quant, out cache);
            Assert.NotNull(cache);
            foreach (KeyValuePair<string, Tensor> kv in source)
            {
                Assert.True(prepared.ContainsKey(kv.Key), $"missing {kv.Key}");
                Assert.Equal(Dims(kv.Value.Shape), Dims(prepared[kv.Key].Shape));
            }
            Assert.NotEqual(DType.F32, prepared["layers.0.mlp.down_proj.weight"].DType);
            Assert.Same(source["layers.0.norm.weight"], prepared["layers.0.norm.weight"]);
        }
        finally
        {
            cache?.Dispose();
            foreach (Tensor t in source.Values) t.Dispose();
            if (Directory.Exists(cacheDir))
            {
                foreach (string file in Directory.GetFiles(cacheDir, $"{repo.Replace('/', '_')}-*"))
                {
                    File.Delete(file);
                }
            }
        }
    }

    private static void Fill(Tensor t, Random rng)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
    }

    private static long[] Dims(TensorShape shape)
    {
        long[] dims = new long[shape.Rank];
        for (int i = 0; i < shape.Rank; i++) dims[i] = shape[i];
        return dims;
    }
}

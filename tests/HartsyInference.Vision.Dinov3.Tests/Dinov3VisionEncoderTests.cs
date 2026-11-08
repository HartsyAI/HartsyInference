using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Dinov3;
using Xunit;

namespace HartsyInference.Vision.Dinov3.Tests;

public sealed class Dinov3VisionEncoderTests
{
    private static readonly IBackend Backend = new CpuBackend();

    [Fact]
    public void LoadWeights_RejectsAShortBlockTensorBeforeAnyKernelReadsIt()
    {
        Dinov2Preset preset = Dinov3Fixture.TinyPreset(imageSize: 64);
        Dictionary<string, Tensor> weights = Dinov3Fixture.Tiny(preset, seed: 5);
        try
        {
            // A one-element norm1.bias would otherwise be read through an unchecked pointer in LayerNorm.
            Tensor full = weights["layer.0.norm1.bias"];
            weights["layer.0.norm1.bias"] = new Tensor(new TensorShape(1), DType.F32);
            full.Dispose();

            Dinov3VisionEncoder encoder = new(preset);
            ArgumentException error = Assert.Throws<ArgumentException>(() => encoder.LoadWeights(weights, ""));
            Assert.Contains("layer.0.norm1.bias", error.Message);
        }
        finally
        {
            Dinov3Fixture.Dispose(weights);
        }
    }

    [Fact]
    public void TokenLayout_IsClsThenStorageThenPatches()
    {
        Dinov2Preset preset = Dinov3Fixture.TinyPreset(imageSize: 64);
        Dictionary<string, Tensor> weights = Dinov3Fixture.Tiny(preset, seed: 1);
        try
        {
            // Zero patch weights make every patch row exactly the projection bias, so the layout is readable directly.
            weights["embeddings.patch_embeddings.weight"].AsSpan<float>().Clear();
            Dinov3VisionEncoder encoder = Load(preset, weights);
            using Tensor pixels = Pixels(1, 64, 64, seed: 2);

            using Tensor embeddings = encoder.EncodeBlocks(Backend, pixels, lastBlock: -1);

            Assert.Equal(new TensorShape(1, 1 + 4 + 16, 128), embeddings.Shape);
            ReadOnlySpan<float> e = embeddings.AsReadOnlySpan<float>();
            ReadOnlySpan<float> cls = weights["embeddings.cls_token"].AsReadOnlySpan<float>();
            ReadOnlySpan<float> reg = weights["embeddings.register_tokens"].AsReadOnlySpan<float>();
            ReadOnlySpan<float> bias = weights["embeddings.patch_embeddings.bias"].AsReadOnlySpan<float>();
            for (int c = 0; c < 128; c++)
            {
                Assert.Equal(cls[c], e[c]);
                for (int r = 0; r < 4; r++) Assert.Equal(reg[r * 128 + c], e[(1 + r) * 128 + c]);
                for (int t = 5; t < 21; t++) Assert.Equal(bias[c], e[t * 128 + c]);
            }
        }
        finally
        {
            Dinov3Fixture.Dispose(weights);
        }
    }

    [Fact]
    public void Encode_NormalizesEveryTokenIncludingStorageAndCls()
    {
        Dinov2Preset preset = Dinov3Fixture.TinyPreset(imageSize: 64);
        Dictionary<string, Tensor> weights = Dinov3Fixture.Tiny(preset, seed: 3);
        try
        {
            Dinov3VisionEncoder encoder = Load(preset, weights);
            using Tensor pixels = Pixels(1, 64, 64, seed: 4);

            using Tensor tokens = encoder.Encode(Backend, pixels);

            Assert.Equal(new TensorShape(1, 21, 128), tokens.Shape);
            ReadOnlySpan<float> t = tokens.AsReadOnlySpan<float>();
            for (int token = 0; token < 21; token++)
            {
                double mean = 0, sq = 0;
                for (int c = 0; c < 128; c++) mean += t[token * 128 + c];
                mean /= 128;
                for (int c = 0; c < 128; c++)
                {
                    double d = t[token * 128 + c] - mean;
                    sq += d * d;
                }
                // Unit weight and zero bias: each row is standardized, so mean 0 and variance eps-shrunk 1.
                Assert.True(Math.Abs(mean) < 1e-4, $"token {token} mean {mean}");
                Assert.True(Math.Abs(sq / 128 - 1) < 1e-3, $"token {token} variance {sq / 128}");
            }
        }
        finally
        {
            Dinov3Fixture.Dispose(weights);
        }
    }

    [Fact]
    public void Encode_BatchMatchesPerImageEncoding()
    {
        Dinov2Preset preset = Dinov3Fixture.TinyPreset(imageSize: 64);
        Dictionary<string, Tensor> weights = Dinov3Fixture.Tiny(preset, seed: 5);
        try
        {
            Dinov3VisionEncoder encoder = Load(preset, weights);
            using Tensor a = Pixels(1, 64, 64, seed: 6);
            using Tensor b = Pixels(1, 64, 64, seed: 7);
            using Tensor both = Pixels(2, 64, 64, seed: 0);
            CopyBatchItem(a, both, 0);
            CopyBatchItem(b, both, 1);

            using Tensor outA = encoder.Encode(Backend, a);
            using Tensor outB = encoder.Encode(Backend, b);
            using Tensor outBoth = encoder.Encode(Backend, both);

            ReadOnlySpan<float> ref1 = outA.AsReadOnlySpan<float>(), ref2 = outB.AsReadOnlySpan<float>(), got = outBoth.AsReadOnlySpan<float>();
            int n = 21 * 128;
            for (int i = 0; i < n; i++)
            {
                Assert.True(Math.Abs(ref1[i] - got[i]) < 1e-5f, $"item 0 element {i}");
                Assert.True(Math.Abs(ref2[i] - got[n + i]) < 1e-5f, $"item 1 element {i}");
            }
        }
        finally
        {
            Dinov3Fixture.Dispose(weights);
        }
    }

    [Fact]
    public void Encode_RejectsSizesOffThePatchGrid()
    {
        Dinov2Preset preset = Dinov3Fixture.TinyPreset(imageSize: 64);
        Dictionary<string, Tensor> weights = Dinov3Fixture.Tiny(preset, seed: 8);
        try
        {
            Dinov3VisionEncoder encoder = Load(preset, weights);
            using Tensor pixels = Pixels(1, 60, 64, seed: 9);

            Assert.Throws<ArgumentException>(() => encoder.Encode(Backend, pixels));
        }
        finally
        {
            Dinov3Fixture.Dispose(weights);
        }
    }

    [Fact]
    public void LoadWeights_RejectsAMisshapenTensor()
    {
        Dinov2Preset preset = Dinov3Fixture.TinyPreset(imageSize: 64);
        Dictionary<string, Tensor> weights = Dinov3Fixture.Tiny(preset, seed: 10);
        try
        {
            weights["embeddings.register_tokens"].Dispose();
            weights["embeddings.register_tokens"] = new Tensor(new TensorShape(1, 3, 128), DType.F32);
            Dinov3VisionEncoder encoder = new(preset);
            Assert.Throws<ArgumentException>(() => encoder.LoadWeights(weights));
        }
        finally
        {
            Dinov3Fixture.Dispose(weights);
        }
    }

    private static Dinov3VisionEncoder Load(Dinov2Preset preset, IReadOnlyDictionary<string, Tensor> weights)
    {
        Dinov3VisionEncoder encoder = new(preset);
        encoder.LoadWeights(weights);
        return encoder;
    }

    private static Tensor Pixels(int batch, int height, int width, int seed)
    {
        Tensor t = new(new TensorShape(batch, 3, height, width), DType.F32);
        Random rng = new(seed);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void CopyBatchItem(Tensor single, Tensor batch, int index)
    {
        long n = single.ElementCount;
        single.AsReadOnlySpan<float>().CopyTo(batch.AsSpan<float>().Slice((int)(index * n), (int)n));
    }
}

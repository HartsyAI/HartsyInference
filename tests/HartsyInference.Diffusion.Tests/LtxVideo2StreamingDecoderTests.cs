using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Streaming;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary><see cref="LtxVideo2StreamingDecoder"/> on a small causal LTX-2 decoder with random weights: decoding a latent fed in
/// pieces, with any number of latent frames per internal pass, joins to the same pixels as one causal decode of the whole
/// latent.</summary>
public sealed class LtxVideo2StreamingDecoderTests
{
    private const int LatentChannels = 4, Mid = 16, OutChannels = 8, Frames = 4, Size = 3;

    [Theory]
    [InlineData(new[] { 1, 3 }, 1)]
    [InlineData(new[] { 1, 3 }, 2)]
    [InlineData(new[] { 1, 3 }, 4)]
    [InlineData(new[] { 4 }, 2)]
    [InlineData(new[] { 2, 2 }, 1)]
    [InlineData(new[] { 1, 1, 1, 1 }, 4)]
    public void StreamedPieces_JoinToTheWholeCausalDecode(int[] feeds, int framesPerPass)
    {
        using CpuBackend cpu = new();
        IBackend backend = cpu;
        Dictionary<string, Tensor> weights = Weights();
        try
        {
            LtxVideo2VaeDecoder decoder = new(latentChannels: LatentChannels, upsampleResidual: [true], patchSize: 1, isCausal: true,
                computeDtype: DType.F32, spatialReflectPad: true);
            decoder.LoadWeights(weights);
            using Tensor latent = Random([1, LatentChannels, Frames, Size, Size], 99, -1f, 1f);
            using Tensor whole = decoder.Decode(backend, latent);

            using LtxVideo2StreamingDecoder streaming = new(backend, decoder, framesPerPass);
            List<Tensor> parts = [];
            try
            {
                int start = 0;
                foreach (int count in feeds)
                {
                    using Tensor piece = StreamingTensors.Slice(backend, latent, 2, start, count);
                    parts.Add(streaming.DecodeNext(piece));
                    start += count;
                }

                using Tensor joined = StreamingTensors.Concat(backend, parts.ToArray(), 2);
                Assert.Equal(whole.Shape, joined.Shape);
                ReadOnlySpan<float> e = whole.AsReadOnlySpan<float>(), a = joined.AsReadOnlySpan<float>();
                float worst = 0f;
                for (int i = 0; i < e.Length; i++)
                    worst = MathF.Max(worst, MathF.Abs(e[i] - a[i]));
                Assert.True(worst <= 1e-4f, $"feeds [{string.Join(",", feeds)}], {framesPerPass} per pass: max difference {worst}");
            }
            finally
            {
                foreach (Tensor part in parts)
                    part.Dispose();
            }
        }
        finally
        {
            foreach (Tensor tensor in weights.Values)
                tensor.Dispose();
        }
    }

    private static Dictionary<string, Tensor> Weights()
    {
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
        int seed = 10;
        void Conv(string name, int cout, int cin)
        {
            weights[$"{name}.conv.weight"] = Random([cout, cin, 3, 3, 3], seed++, -0.1f, 0.1f);
            weights[$"{name}.conv.bias"] = Random([cout], seed++, -0.05f, 0.05f);
        }

        Conv("decoder.conv_in", Mid, LatentChannels);
        Conv("decoder.mid_block.resnets.0.conv1", Mid, Mid);
        Conv("decoder.mid_block.resnets.0.conv2", Mid, Mid);
        Conv("decoder.up_blocks.0.upsamplers.0.conv", OutChannels * 8, Mid);
        Conv("decoder.up_blocks.0.resnets.0.conv1", OutChannels, OutChannels);
        Conv("decoder.up_blocks.0.resnets.0.conv2", OutChannels, OutChannels);
        Conv("decoder.conv_out", 3, OutChannels);
        return weights;
    }

    private static Tensor Random(ReadOnlySpan<long> dims, int seed, float low, float high)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32, DeviceKind.Cpu);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = low + (high - low) * random.NextSingle();
        return tensor;
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed unsafe class VaeTileAccumulatorTests
{
    private const int Scale = 8;

    /// <summary>Tent-blended tiles of a per-tile decode that matches the full decode must reproduce it exactly.</summary>
    [Theory]
    [InlineData(20, 28, 8, 2)]
    [InlineData(17, 9, 6, 3)]
    public void Decode_TiledMatchesFullDecode_WhenTileDecodeIsExact(int latentH, int latentW, int tile, int overlap)
    {
        using IBackend backend = new CpuBackend();
        using Tensor latent = new Tensor(new TensorShape(1, 3, latentH, latentW), DType.F32);
        float* src = (float*)latent.DataPointer;
        for (long i = 0; i < latent.ElementCount; i++) src[i] = (float)Math.Sin(i * 0.37);

        using Tensor expected = NearestUpsample(latent);
        using Tensor actual = VaeTileAccumulator.Decode(backend, latent, tile, overlap, DType.F32, Scale,
            NearestUpsample, (h, w) => 0L);

        Assert.Equal(expected.Shape, actual.Shape);
        float* want = (float*)expected.DataPointer;
        float* got = (float*)actual.DataPointer;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            Assert.True(Math.Abs(want[i] - got[i]) < 1e-5f, $"element {i}: expected {want[i]}, got {got[i]}");
        }
    }

    private static Tensor NearestUpsample(Tensor input)
    {
        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int h = (int)input.Shape[2];
        int w = (int)input.Shape[3];
        Tensor output = new Tensor(new TensorShape(batch, channels, h * Scale, w * Scale), DType.F32);
        float* src = (float*)input.DataPointer;
        float* dst = (float*)output.DataPointer;
        int outW = w * Scale;
        for (int plane = 0; plane < batch * channels; plane++)
        {
            for (int y = 0; y < h * Scale; y++)
            {
                for (int x = 0; x < outW; x++)
                {
                    dst[(long)plane * h * Scale * outW + (long)y * outW + x] = src[(long)plane * h * w + (y / Scale) * w + x / Scale];
                }
            }
        }
        return output;
    }
}

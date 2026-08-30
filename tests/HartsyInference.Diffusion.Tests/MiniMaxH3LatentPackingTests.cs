using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Locks the condition-latent patch boundary where an odd VAE grid would otherwise lose its final row or column.</summary>
public sealed unsafe class MiniMaxH3LatentPackingTests
{
    [Fact]
    public void PackVideoCircularPadsOddSpatialAxes()
    {
        MiniMaxH3Config config = new MiniMaxH3Config
        {
            LatentsDim = 1,
            PatchT = 1,
            PatchH = 2,
            PatchW = 2,
        };
        using Tensor latent = new Tensor(new TensorShape([1L, 1, 1, 3, 5]), DType.F32);
        float* source = (float*)latent.DataPointer;
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 5; x++)
            {
                source[y * 5 + x] = y * 10 + x;
            }
        }

        using Tensor rows = MiniMaxH3Latents.PackVideo(latent, config);

        Assert.Equal(new TensorShape(6, 4), rows.Shape);
        float* packed = (float*)rows.DataPointer;
        long finalRow = 5 * 4L;
        Assert.Equal(24f, packed[finalRow]);
        Assert.Equal(20f, packed[finalRow + 1]);
        Assert.Equal(4f, packed[finalRow + 2]);
        Assert.Equal(0f, packed[finalRow + 3]);
    }

    [Fact]
    public void PackVideoLeavesAlignedAxesByteEquivalent()
    {
        MiniMaxH3Config config = new MiniMaxH3Config
        {
            LatentsDim = 1,
            PatchT = 1,
            PatchH = 2,
            PatchW = 2,
        };
        using Tensor latent = new Tensor(new TensorShape([1L, 1, 1, 2, 2]), DType.F32);
        float[] values = [1f, 2f, 3f, 4f];
        values.CopyTo(latent.AsSpan<float>());
        using Tensor rows = MiniMaxH3Latents.PackVideo(latent, config);
        Assert.Equal(values, rows.AsSpan<float>().ToArray());
    }
}

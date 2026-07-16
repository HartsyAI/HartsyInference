using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Locks the VAE encoder downsample to the diffusers recipe: asymmetric zero-pad (right/bottom by 1,
/// <c>F.pad((0,1,0,1))</c>) followed by Conv2d(kernel=3, stride=2, padding=0). The previously-used symmetric
/// padding=1 conv yields the same output SIZE for even inputs but samples the opposite pixel parity — a one-pixel
/// grid shift per stride-2 stage (~1 latent pixel across the three encoder stages) that put encoded latents
/// off-distribution. Flux Kontext reproduced reference textures as maze/speckle artifacts from that misalignment
/// (encode corr vs ComfyUI: 0.87 misaligned → 0.999993 aligned).</summary>
public sealed unsafe class VaeEncoderDownsamplePaddingTests
{
    [Fact]
    public void PadRightBottom_AppendsZeroRowAndColumn()
    {
        using CpuBackend backend = new();
        using Tensor input = new(new TensorShape(1, 2, 2, 2), DType.F32);
        float* ip = (float*)input.DataPointer;
        for (int i = 0; i < 8; i++) ip[i] = i + 1;

        Tensor padded = VaeEncoder.PadRightBottom(backend, input, batch: 1, channels: 2, h: 2, w: 2, DType.F32);
        try
        {
            Assert.Equal(new TensorShape(1, 2, 3, 3), padded.Shape);
            float* pp = (float*)padded.DataPointer;
            for (int c = 0; c < 2; c++)
            {
                float* cp = pp + c * 9;
                float* ci = ip + c * 4;
                // Original 2x2 content in the top-left corner.
                Assert.Equal(ci[0], cp[0]);
                Assert.Equal(ci[1], cp[1]);
                Assert.Equal(ci[2], cp[3]);
                Assert.Equal(ci[3], cp[4]);
                // Right column and bottom row are zero.
                Assert.Equal(0f, cp[2]);
                Assert.Equal(0f, cp[5]);
                Assert.Equal(0f, cp[6]);
                Assert.Equal(0f, cp[7]);
                Assert.Equal(0f, cp[8]);
            }
        }
        finally
        {
            padded.Dispose();
        }
    }

    [Fact]
    public void AsymmetricDownsample_SamplesOddPixelParity()
    {
        // With a delta kernel at the 3x3 center, pad(0,1,0,1) + stride-2 conv(padding=0) must pick
        // input[2i+1, 2j+1] (the diffusers/BFL parity). Symmetric padding=1 would pick input[2i, 2j].
        const int H = 6, W = 6;
        using CpuBackend backend = new();
        using Tensor input = new(new TensorShape(1, 1, H, W), DType.F32);
        float* ip = (float*)input.DataPointer;
        for (int i = 0; i < H * W; i++) ip[i] = i;

        using Tensor weight = new(new TensorShape(1, 1, 3, 3), DType.F32);
        float* wp = (float*)weight.DataPointer;
        for (int i = 0; i < 9; i++) wp[i] = 0f;
        wp[4] = 1f;

        Tensor padded = VaeEncoder.PadRightBottom(backend, input, batch: 1, channels: 1, h: H, w: W, DType.F32);
        using Tensor output = new(new TensorShape(1, 1, H / 2, W / 2), DType.F32);
        try
        {
            backend.Conv2D(output, padded, weight, null, 2, 2, 0, 0);
            float* op = (float*)output.DataPointer;
            for (int i = 0; i < H / 2; i++)
            {
                for (int j = 0; j < W / 2; j++)
                {
                    Assert.Equal(ip[(2 * i + 1) * W + (2 * j + 1)], op[i * (W / 2) + j]);
                }
            }
        }
        finally
        {
            padded.Dispose();
        }
    }
}

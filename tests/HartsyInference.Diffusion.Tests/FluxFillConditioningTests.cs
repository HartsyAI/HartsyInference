using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Unit tests for the FLUX.1 Fill conditioning math: masked-image formation, the pixel-mask → 64-channel latent expansion, and the packed 256-feature layout — each checked against an independent re-index of the diffusers FluxFillPipeline formulas (mask.view(B,h,8,w,8).permute(0,2,4,1,3) then _pack_latents' c*4 + dy*2 + dx feature order).</summary>
public sealed class FluxFillConditioningTests
{
    [Fact]
    public unsafe void MaskPixelsToNeutral_ZeroesMaskedRegionAcrossChannels()
    {
        const int h = 8;
        const int w = 8;
        Tensor source = new Tensor(new TensorShape(1, 3, h, w), DType.F32);
        Tensor mask = new Tensor(new TensorShape(1, 1, h, w), DType.F32);
        float* sp = (float*)source.DataPointer;
        float* mp = (float*)mask.DataPointer;
        for (int i = 0; i < 3 * h * w; i++) sp[i] = 0.25f + 0.001f * i;
        // Soft mask: left half 1 (inpaint), right half 0.25 (partial).
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                mp[y * w + x] = x < w / 2 ? 1.0f : 0.25f;
            }
        }

        Tensor masked = FluxPipeline.MaskPixelsToNeutral(source, mask);
        float* op = (float*)masked.DataPointer;
        for (int c = 0; c < 3; c++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    long idx = (long)c * h * w + y * w + x;
                    float expected = sp[idx] * (1.0f - mp[y * w + x]);
                    Assert.Equal(expected, op[idx], 6);
                }
            }
        }

        source.Dispose();
        mask.Dispose();
        masked.Dispose();
    }

    [Fact]
    public unsafe void FillMaskToLatentChannels_MatchesDiffusersViewPermute()
    {
        const int latentH = 4;
        const int latentW = 6;
        const int pixH = latentH * 8;
        const int pixW = latentW * 8;
        Tensor maskPixel = new Tensor(new TensorShape(1, 1, pixH, pixW), DType.F32);
        float* mp = (float*)maskPixel.DataPointer;
        // Unique value per pixel so any index mix-up is caught.
        for (int i = 0; i < pixH * pixW; i++) mp[i] = i * (1.0f / (pixH * pixW));

        Tensor mask64 = FluxPipeline.FillMaskToLatentChannels(maskPixel, latentH, latentW);
        Assert.Equal(new long[] { 1, 64, latentH, latentW }, new long[] { mask64.Shape[0], mask64.Shape[1], mask64.Shape[2], mask64.Shape[3] });

        float* op = (float*)mask64.DataPointer;
        for (int sy = 0; sy < 8; sy++)
        {
            for (int sx = 0; sx < 8; sx++)
            {
                int c = sy * 8 + sx;
                for (int i = 0; i < latentH; i++)
                {
                    for (int j = 0; j < latentW; j++)
                    {
                        float expected = mp[(i * 8 + sy) * pixW + (j * 8 + sx)];
                        float actual = op[((long)c * latentH + i) * latentW + j];
                        Assert.Equal(expected, actual, 6);
                    }
                }
            }
        }

        maskPixel.Dispose();
        mask64.Dispose();
    }

    [Fact]
    public unsafe void PackedFillMask_FeatureLayoutMatchesPackLatentsOrder()
    {
        const int latentH = 4;
        const int latentW = 4;
        const int pixH = latentH * 8;
        const int pixW = latentW * 8;
        Tensor maskPixel = new Tensor(new TensorShape(1, 1, pixH, pixW), DType.F32);
        float* mp = (float*)maskPixel.DataPointer;
        for (int i = 0; i < pixH * pixW; i++) mp[i] = i + 1.0f;

        Tensor mask64 = FluxPipeline.FillMaskToLatentChannels(maskPixel, latentH, latentW);
        Tensor packed = FluxPipeline.PackLatent(mask64, latentH, latentW);
        int hPacked = latentH / 2;
        int wPacked = latentW / 2;
        Assert.Equal(new long[] { 1, hPacked * wPacked, 256 }, new long[] { packed.Shape[0], packed.Shape[1], packed.Shape[2] });

        // packed[s, c*4 + dy*2 + dx] must equal maskPixel[( (ph*2+dy)*8 + sy )*pixW + (pw*2+dx)*8 + sx]
        // with c = sy*8+sx and s = ph*wPacked+pw — the composition of the diffusers mask reshape
        // and _pack_latents feature ordering.
        float* pp = (float*)packed.DataPointer;
        for (int ph = 0; ph < hPacked; ph++)
        {
            for (int pw = 0; pw < wPacked; pw++)
            {
                int s = ph * wPacked + pw;
                for (int sy = 0; sy < 8; sy++)
                {
                    for (int sx = 0; sx < 8; sx++)
                    {
                        int c = sy * 8 + sx;
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                float expected = mp[((ph * 2 + dy) * 8 + sy) * pixW + (pw * 2 + dx) * 8 + sx];
                                float actual = pp[(long)s * 256 + c * 4 + dy * 2 + dx];
                                Assert.Equal(expected, actual, 4);
                            }
                        }
                    }
                }
            }
        }

        maskPixel.Dispose();
        mask64.Dispose();
        packed.Dispose();
    }
}

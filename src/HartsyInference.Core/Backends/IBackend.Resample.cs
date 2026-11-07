using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Spatial resampling: nearest and bilinear upsampling and interpolation.
public partial interface IBackend
{
    /// <summary>Nearest-neighbor 2D upsample by the given scale factor.</summary>
    void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW);

    /// <summary>Nearest-neighbour ×2 upsample of NCHW <paramref name="input"/> into <paramref name="output"/>, whose height
    /// and width are each either twice the input's or one less. Output row <c>oh</c> reads input row <c>oh / 2</c>, which
    /// is exactly <c>interpolate(size=…, mode="nearest")</c> for those sizes — what a UNet up path needs when its stride-2
    /// down path rounded an odd size up (diffusers passes the skip's size as <c>upsample_size</c>). Only ×2: at larger
    /// factors a short output no longer reads the same rows as the nearest-neighbour rule. Default: the full upsample into
    /// scratch, then two <see cref="SliceLastDim"/> prefix slices — device-resident on any backend that serves those.</summary>
    unsafe void UpsampleNearest2DToSize(Tensor output, Tensor input)
    {
        (int outH, int outW) = UpsampleNearestExtent.Validate(output, input);
        long n = input.Shape[0], c = input.Shape[1], fullH = 2 * input.Shape[2], fullW = 2 * input.Shape[3];
        if (outH == fullH && outW == fullW)
        {
            UpsampleNearest2D(output, input, 2, 2);
            return;
        }
        using Tensor full = new(new TensorShape(n, c, fullH, fullW), input.DType);
        UpsampleNearest2D(full, input, 2, 2);
        // Keep the leading outW of every row; each plane is then [fullH, outW], so its leading outH rows are one
        // contiguous prefix — a second last-dim slice over [planes, fullH·outW]. Both stay on the device.
        using Tensor narrow = new(new TensorShape(n * c, fullH * outW), input.DType);
        SliceLastDim(narrow, full, 0);
        SliceLastDim(output, narrow, 0);
    }

    /// <summary>Bilinear 2D resize of an NCHW tensor to output's size, matching PyTorch <c>F.interpolate(mode="bilinear")</c> mapping.</summary>
    /// <param name="alignCorners">True: <c>src=dst·(in−1)/(out−1)</c>; false: <c>src=(dst+0.5)·in/out−0.5</c> (DPT fusion: true).</param>
    unsafe void InterpolateBilinear2D(Tensor output, Tensor input, bool alignCorners)
    {
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException($"InterpolateBilinear2D default fallback only supports F32 — got input={input.DType}, output={output.DType}.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4)
            throw new ArgumentException($"InterpolateBilinear2D requires 4D tensors; got input {input.Shape}, output {output.Shape}.");
        if (input.Shape[0] != output.Shape[0] || input.Shape[1] != output.Shape[1])
            throw new ArgumentException($"InterpolateBilinear2D batch/channel dims must match; got input {input.Shape}, output {output.Shape}.");

        int n = (int)input.Shape[0], c = (int)input.Shape[1];
        int inH = (int)input.Shape[2], inW = (int)input.Shape[3];
        int outH = (int)output.Shape[2], outW = (int)output.Shape[3];
        float* src = (float*)input.DataPointer;
        float* dst = (float*)output.DataPointer;

        for (int oy = 0; oy < outH; oy++)
        {
            float sy = alignCorners ? (outH == 1 ? 0f : oy * (float)(inH - 1) / (outH - 1))
                : (oy + 0.5f) * inH / outH - 0.5f;
            int y0 = (int)MathF.Floor(sy);
            float fy = sy - y0;
            int y0c = Math.Clamp(y0, 0, inH - 1), y1c = Math.Clamp(y0 + 1, 0, inH - 1);
            for (int ox = 0; ox < outW; ox++)
            {
                float sx = alignCorners ? (outW == 1 ? 0f : ox * (float)(inW - 1) / (outW - 1))
                    : (ox + 0.5f) * inW / outW - 0.5f;
                int x0 = (int)MathF.Floor(sx);
                float fx = sx - x0;
                int x0c = Math.Clamp(x0, 0, inW - 1), x1c = Math.Clamp(x0 + 1, 0, inW - 1);
                for (long plane = 0; plane < (long)n * c; plane++)
                {
                    float* sp = src + plane * inH * inW;
                    float v0 = sp[y0c * inW + x0c] * (1f - fx) + sp[y0c * inW + x1c] * fx;
                    float v1 = sp[y1c * inW + x0c] * (1f - fx) + sp[y1c * inW + x1c] * fx;
                    dst[plane * outH * outW + (long)oy * outW + ox] = v0 * (1f - fy) + v1 * fy;
                }
            }
        }
    }
}

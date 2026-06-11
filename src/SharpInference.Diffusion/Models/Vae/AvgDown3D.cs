using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>Wan2.2 VAE encoder shortcut pooling (<c>AvgDown3D</c> in <c>vae2_2.py</c>) — the weightless dual of the
/// decoder's <see cref="DupUp3D"/>. Front-pads T to a multiple of <c>factor_t</c> with zeros, stacks each
/// <c>factor_t × factor_s × factor_s</c> cell into channels in <c>(c, t, h, w)</c> factor order, then mean-pools
/// groups of <c>in·factor / out</c> consecutive stacked channels down to <c>out</c> channels.</summary>
public static unsafe class AvgDown3D
{
    /// <summary>Pools <c>[B, inC, T, H, W]</c> → <c>[B, outChannels, ceil(T/ft), H/fs, W/fs]</c>.</summary>
    public static Tensor Forward(Tensor x, int outChannels, int factorT, int factorS)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        if (h % factorS != 0 || w % factorS != 0)
            throw new ArgumentException($"H/W ({h}×{w}) must be divisible by factor_s {factorS}.", nameof(x));
        int factor = factorT * factorS * factorS;
        if ((long)c * factor % outChannels != 0)
            throw new ArgumentException($"in·factor ({c}·{factor}) must be divisible by outChannels {outChannels}.", nameof(outChannels));
        int groupSize = c * factor / outChannels;

        int padT = (factorT - t % factorT) % factorT;
        int tPadded = t + padT;
        int ot = tPadded / factorT, oh = h / factorS, ow = w / factorS;

        Tensor outT = new Tensor(new TensorShape([(long)b, outChannels, ot, oh, ow]), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        long inFrame = (long)h * w;

        for (int bi = 0; bi < b; bi++)
            for (int oc = 0; oc < outChannels; oc++)
                for (int oti = 0; oti < ot; oti++)
                    for (int ohi = 0; ohi < oh; ohi++)
                        for (int owi = 0; owi < ow; owi++)
                        {
                            double sum = 0;
                            for (int g = 0; g < groupSize; g++)
                            {
                                // Stacked-channel index → (c, it, ih, iw) factor coordinates, (c, t, h, w) order.
                                int flat = oc * groupSize + g;
                                int iw = flat % factorS; flat /= factorS;
                                int ih = flat % factorS; flat /= factorS;
                                int it = flat % factorT; flat /= factorT;
                                int ci = flat;
                                int ti = oti * factorT + it - padT;   // front zero-pad
                                if (ti < 0) continue;                  // padded frame contributes 0
                                long src = (((long)bi * c + ci) * t + ti) * inFrame + (long)(ohi * factorS + ih) * w + (owi * factorS + iw);
                                sum += sp[src];
                            }
                            op[(((long)bi * outChannels + oc) * ot + oti) * oh * ow + (long)ohi * ow + owi] = (float)(sum / groupSize);
                        }
        return outT;
    }
}

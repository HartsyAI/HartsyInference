using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>Wan2.2 VAE duplicating up-sampler (<c>DupUp3D</c> in <c>vae2_2.py</c>), reusable by any Wan/LTX-family video decoder. Expands a <c>[B, inC, T, H, W]</c> latent to <c>[B, outC, T·factorT, H·factorS, W·factorS]</c> by channel <c>repeat_interleave</c> + a reshape/permute that scatters the duplicated channels into the temporal & spatial cells (a learnable-free nearest-style expansion).
///
/// <para>Index map (verbatim from the upstream reshape/permute): for output cell <c>(oc, t·factorT+tt, h·factorS+s1, w·factorS+s2)</c>, the source is input channel <c>c' / repeats</c> where <c>c' = ((oc·factorT + tt)·factorS + s1)·factorS + s2</c> and <c>repeats = outC·factor / inC</c>, <c>factor = factorT·factorS²</c>. With <paramref name="firstChunk"/>, the leading <c>factorT−1</c> temporal frames are dropped (so a single latent frame decodes to a single output frame).</para></summary>
public static unsafe class DupUp3D
{
    public static Tensor Forward(Tensor x, int outChannels, int factorT, int factorS, bool firstChunk)
    {
        int b = (int)x.Shape[0], inC = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int factor = factorT * factorS * factorS;
        if (outChannels * factor % inC != 0)
            throw new ArgumentException($"outChannels·factor ({outChannels}·{factor}) not divisible by inC ({inC}).");
        int repeats = outChannels * factor / inC;

        int outT = t * factorT, outH = h * factorS, outW = w * factorS;
        int keepT = firstChunk ? outT - (factorT - 1) : outT;
        int dropT = outT - keepT;

        Tensor outT_ = new Tensor(new TensorShape([(long)b, outChannels, keepT, outH, outW]), DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)outT_.DataPointer;

        for (int bi = 0; bi < b; bi++)
            for (int oc = 0; oc < outChannels; oc++)
                for (int tt = 0; tt < factorT; tt++)
                    for (int s1 = 0; s1 < factorS; s1++)
                        for (int s2 = 0; s2 < factorS; s2++)
                        {
                            int cPrime = ((oc * factorT + tt) * factorS + s1) * factorS + s2;
                            int srcC = cPrime / repeats;
                            for (int ti = 0; ti < t; ti++)
                            {
                                int oTime = ti * factorT + tt;
                                if (oTime < dropT) continue;       // first_chunk drops leading frames
                                int oTimeKept = oTime - dropT;
                                for (int hi = 0; hi < h; hi++)
                                    for (int wi = 0; wi < w; wi++)
                                    {
                                        long srcOff = ((((long)bi * inC + srcC) * t + ti) * h + hi) * w + wi;
                                        long dstOff = ((((long)bi * outChannels + oc) * keepT + oTimeKept) * outH + (hi * factorS + s1)) * outW + (wi * factorS + s2);
                                        dst[dstOff] = src[srcOff];
                                    }
                            }
                        }
        return outT_;
    }
}

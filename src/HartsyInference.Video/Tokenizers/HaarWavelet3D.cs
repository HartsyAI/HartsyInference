using HartsyInference.Core.Tensors;

namespace HartsyInference.Video.Tokenizers;

/// <summary>Separable orthonormal 3D Haar wavelet transform — the front-end of the Cosmos-Tokenize1-DV8x16x16
/// encoder (and the tail of its decoder). This is the one genuinely net-new op in the DV tokenizer port: there is
/// no existing Haar implementation in the engine.
///
/// <para>One level factors each non-overlapping <c>2×2×2</c> spatio-temporal cell into 8 sub-bands — the tensor
/// product of a 1D Haar low/high pass along <c>(T, H, W)</c> — and stacks the sub-bands into the channel dimension
/// (<c>C → C·8</c>, dims halved). A level is <b>orthonormal</b> (each 1D step is <c>lo=(a+b)/√2, hi=(a−b)/√2</c>),
/// so <see cref="Inverse"/> is the exact transpose and <c>Inverse(Forward(x)) == x</c> to float precision — the
/// round-trip invariant the unit test pins. Cosmos runs 2 levels as its "2-level Haar wavelet" front-end.</para>
///
/// <para><b>Integration note.</b> The sub-band channel ordering and the exact per-level axis schedule that
/// reproduce NVIDIA's <c>encoder.jit</c> bit-for-bit are resolved at integration against the real JIT (research-doc
/// Open Q §10); the round-trip property proven here is convention-independent. Sub-bands are laid out channel-major
/// per input channel: output channel <c>c·8 + s</c> where <c>s = st·4 + sh·2 + sw</c>, <c>st/sh/sw ∈ {0=low,1=high}</c>.</para></summary>
public static unsafe class HaarWavelet3D
{
    private static readonly float InvSqrt2 = 1f / MathF.Sqrt(2f);

    /// <summary>Applies <paramref name="levels"/> orthonormal 3D Haar levels to a <c>[B, C, T, H, W]</c> tensor.
    /// Each level requires the current <c>T, H, W</c> to be even. Returns <c>[B, C·8^levels, T/2^levels,
    /// H/2^levels, W/2^levels]</c>.</summary>
    public static Tensor Forward(Tensor x, int levels = 1)
    {
        if (levels < 1) throw new ArgumentOutOfRangeException(nameof(levels), "levels must be >= 1.");
        Tensor current = x;
        bool ownsCurrent = false;
        for (int l = 0; l < levels; l++)
        {
            Tensor next = ForwardLevel(current);
            if (ownsCurrent) current.Dispose();
            current = next;
            ownsCurrent = true;
        }
        return current;
    }

    /// <summary>Inverts <paramref name="levels"/> orthonormal 3D Haar levels: <c>[B, C, T, H, W]</c> →
    /// <c>[B, C/8^levels, T·2^levels, H·2^levels, W·2^levels]</c>. Exact inverse of <see cref="Forward"/>.</summary>
    public static Tensor Inverse(Tensor x, int levels = 1)
    {
        if (levels < 1) throw new ArgumentOutOfRangeException(nameof(levels), "levels must be >= 1.");
        Tensor current = x;
        bool ownsCurrent = false;
        for (int l = 0; l < levels; l++)
        {
            Tensor next = InverseLevel(current);
            if (ownsCurrent) current.Dispose();
            current = next;
            ownsCurrent = true;
        }
        return current;
    }

    private static Tensor ForwardLevel(Tensor x)
    {
        if (x.Shape.Rank != 5) throw new ArgumentException($"Haar input must be [B,C,T,H,W], got {x.Shape}.", nameof(x));
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        if ((t & 1) != 0 || (h & 1) != 0 || (w & 1) != 0)
            throw new ArgumentException($"Haar forward requires even T/H/W, got [{t},{h},{w}].", nameof(x));
        int ot = t / 2, oh = h / 2, ow = w / 2, oc = c * 8;

        Tensor outT = new Tensor(new TensorShape([(long)b, oc, ot, oh, ow]), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        const float norm = 0.35355339059327373f;   // (1/√2)^3

        float* cell = stackalloc float[8];
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                long inBase = ((long)bi * c + ci) * t * h * w;
                for (int oti = 0; oti < ot; oti++)
                    for (int ohi = 0; ohi < oh; ohi++)
                        for (int owi = 0; owi < ow; owi++)
                        {
                            // Load the 2×2×2 cell.
                            for (int it = 0; it < 2; it++)
                                for (int ih = 0; ih < 2; ih++)
                                    for (int iw = 0; iw < 2; iw++)
                                    {
                                        long idx = inBase + (((long)(oti * 2 + it) * h + (ohi * 2 + ih)) * w) + (owi * 2 + iw);
                                        cell[it * 4 + ih * 2 + iw] = sp[idx];
                                    }
                            for (int s = 0; s < 8; s++)
                            {
                                int st = (s >> 2) & 1, sh = (s >> 1) & 1, sw = s & 1;
                                float acc = 0f;
                                for (int it = 0; it < 2; it++)
                                    for (int ih = 0; ih < 2; ih++)
                                        for (int iw = 0; iw < 2; iw++)
                                        {
                                            float sign = Sign(st, it) * Sign(sh, ih) * Sign(sw, iw);
                                            acc += sign * cell[it * 4 + ih * 2 + iw];
                                        }
                                long outIdx = ((((long)bi * oc + (ci * 8 + s)) * ot + oti) * oh + ohi) * ow + owi;
                                op[outIdx] = acc * norm;
                            }
                        }
            }
        return outT;
    }

    private static Tensor InverseLevel(Tensor x)
    {
        if (x.Shape.Rank != 5) throw new ArgumentException($"Haar input must be [B,C,T,H,W], got {x.Shape}.", nameof(x));
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        if (c % 8 != 0) throw new ArgumentException($"Haar inverse requires C divisible by 8, got {c}.", nameof(x));
        int oc = c / 8, ot = t * 2, oh = h * 2, ow = w * 2;

        Tensor outT = new Tensor(new TensorShape([(long)b, oc, ot, oh, ow]), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        const float norm = 0.35355339059327373f;

        float* coeff = stackalloc float[8];
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < oc; ci++)
            {
                long outBase = ((long)bi * oc + ci) * ot * oh * ow;
                for (int ti = 0; ti < t; ti++)
                    for (int hi = 0; hi < h; hi++)
                        for (int wi = 0; wi < w; wi++)
                        {
                            for (int s = 0; s < 8; s++)
                            {
                                long inIdx = ((((long)bi * c + (ci * 8 + s)) * t + ti) * h + hi) * w + wi;
                                coeff[s] = sp[inIdx];
                            }
                            for (int it = 0; it < 2; it++)
                                for (int ih = 0; ih < 2; ih++)
                                    for (int iw = 0; iw < 2; iw++)
                                    {
                                        float acc = 0f;
                                        for (int s = 0; s < 8; s++)
                                        {
                                            int st = (s >> 2) & 1, sh = (s >> 1) & 1, sw = s & 1;
                                            float sign = Sign(st, it) * Sign(sh, ih) * Sign(sw, iw);
                                            acc += sign * coeff[s];
                                        }
                                        long outIdx = outBase + (((long)(ti * 2 + it) * oh + (hi * 2 + ih)) * ow) + (wi * 2 + iw);
                                        op[outIdx] = acc * norm;
                                    }
                        }
            }
        return outT;
    }

    // 1D Haar sign: low pass = [+1, +1]; high pass = [+1, -1].
    private static float Sign(int band, int i) => band == 0 ? 1f : (i == 0 ? 1f : -1f);
}

using System.Runtime.CompilerServices;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>3-axis rotary positional embedding for Anima / Cosmos-Predict2 (<c>CosmosRotaryPosEmbed</c>). Splits the head
/// dimension into <c>(t, h, w)</c> sub-bands using Cosmos's allocation scheme:
/// <code>
///     dim_h = (hidden_size // 6) * 2
///     dim_w = (hidden_size // 6) * 2
///     dim_t = hidden_size - dim_h - dim_w
/// </code>
/// Each axis applies a per-axis NTK frequency scaling (<c>theta * scale^(d/(d-2))</c>) before computing inverse frequencies,
/// then concatenates <c>[emb_t, emb_h, emb_w]</c> twice along the last dim so the final cos/sin tables are <c>head_dim</c>-wide
/// with the second half repeating the first (Cosmos's <c>torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1)</c>).
///
/// <para>The rotation is **GPT-NeoX rotate-half** (<c>use_real=True</c>, <c>use_real_unbind_dim=-2</c>): the head dim is split
/// into two halves of <c>head_dim/2</c> elements; the first half is treated as the "real" component, the second as "imag", and
/// the rotation is <c>(x_real * cos - x_imag * sin, x_real * sin + x_imag * cos)</c>. Because the upstream concat duplicates
/// freqs into both halves, both halves see the same cos/sin pattern and the rotation is well-defined.</para>
///
/// <para>For the t2i variant (<c>T = 1</c>), the temporal axis evaluates to position 0 and contributes <c>cos=1, sin=0</c> for
/// every <c>dim_t/2</c> frequency — i.e. it leaves Q/K untouched in the temporal sub-band. We compute and store these zeros
/// anyway so the layout matches the upstream tables byte-for-byte (useful for diff dumps).</para></summary>
public sealed unsafe class AnimaRope
{
    private readonly int _headDim;
    private readonly int _dimT;
    private readonly int _dimH;
    private readonly int _dimW;
    private readonly float _theta;
    private readonly (float T, float H, float W) _ntkFactor;

    /// <summary>Creates a 3-axis Cosmos RoPE for the given <paramref name="hiddenSize"/> per head and base period.
    /// <paramref name="ropeScale"/> applies the upstream NTK scale: <c>theta_axis = theta * scale^(dim_axis/(dim_axis-2))</c>
    /// for axes with <c>dim_axis > 2</c>.</summary>
    public AnimaRope(int hiddenSize, float theta, (float T, float H, float W) ropeScale)
    {
        _headDim = hiddenSize;
        _dimH = (hiddenSize / 6) * 2;
        _dimW = (hiddenSize / 6) * 2;
        _dimT = hiddenSize - _dimH - _dimW;
        if (_dimT % 2 != 0 || _dimH % 2 != 0 || _dimW % 2 != 0)
            throw new ArgumentException(
                $"AnimaRope: each axis dim must be even. Got dim_t={_dimT}, dim_h={_dimH}, dim_w={_dimW} for hidden={hiddenSize}.",
                nameof(hiddenSize));
        _theta = theta;
        _ntkFactor = (
            T: NtkFactor(ropeScale.T, _dimT),
            H: NtkFactor(ropeScale.H, _dimH),
            W: NtkFactor(ropeScale.W, _dimW));
    }

    /// <summary>Total head dim (= sum of axis sub-dims).</summary>
    public int HeadDim => _headDim;

    /// <summary>Per-axis sub-dim split <c>(dim_t, dim_h, dim_w)</c>.</summary>
    public (int T, int H, int W) AxisDim => (_dimT, _dimH, _dimW);

    /// <summary>Computes one cos/sin table per token for an image grid of shape <c>(tFrames, hPatched, wPatched)</c>. For
    /// t2i, callers pass <c>tFrames=1</c>. Returns <see cref="Tensor"/> arrays sized <c>[seqLen, headDim]</c> each,
    /// row-major over <c>(t, h, w)</c>. Caller takes ownership and disposes.</summary>
    public (Tensor Cos, Tensor Sin) BuildFreqs(int tFrames, int hPatched, int wPatched)
    {
        int seqLen = tFrames * hPatched * wPatched;
        int halfDim = _headDim / 2;

        TensorShape shape = new TensorShape(seqLen, _headDim);
        Tensor cos = new Tensor(shape, DType.F32);
        Tensor sin = new Tensor(shape, DType.F32);
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;

        // Inverse frequencies per axis: 1.0 / (theta_axis ^ (2k / dim_axis)).
        Span<float> invFreqsT = _dimT > 0 ? new float[_dimT / 2] : [];
        Span<float> invFreqsH = _dimH > 0 ? new float[_dimH / 2] : [];
        Span<float> invFreqsW = _dimW > 0 ? new float[_dimW / 2] : [];
        FillInvFreqs(invFreqsT, _dimT, _theta * _ntkFactor.T);
        FillInvFreqs(invFreqsH, _dimH, _theta * _ntkFactor.H);
        FillInvFreqs(invFreqsW, _dimW, _theta * _ntkFactor.W);

        for (int t = 0; t < tFrames; t++)
        {
            for (int h = 0; h < hPatched; h++)
            {
                for (int w = 0; w < wPatched; w++)
                {
                    int seqIdx = (t * hPatched + h) * wPatched + w;
                    long rowOff = (long)seqIdx * _headDim;

                    // Concat order in the upstream: [emb_t, emb_h, emb_w], then repeated to fill head_dim.
                    // Each emb_x contributes dim_x/2 unique cos/sin values.
                    int featOff = 0;
                    WriteAxisFreqs(cosPtr, sinPtr, rowOff, featOff, t, invFreqsT);
                    featOff += _dimT / 2;
                    WriteAxisFreqs(cosPtr, sinPtr, rowOff, featOff, h, invFreqsH);
                    featOff += _dimH / 2;
                    WriteAxisFreqs(cosPtr, sinPtr, rowOff, featOff, w, invFreqsW);
                    featOff += _dimW / 2;

                    // Second half of head_dim is a verbatim copy of the first half. Standard "rotate-half" form
                    // sees the same cos/sin in both halves so the rotation is consistent across the split.
                    long copyBytes = (long)halfDim * sizeof(float);
                    Buffer.MemoryCopy(cosPtr + rowOff, cosPtr + rowOff + halfDim, copyBytes, copyBytes);
                    Buffer.MemoryCopy(sinPtr + rowOff, sinPtr + rowOff + halfDim, copyBytes, copyBytes);
                }
            }
        }
        return (cos, sin);
    }

    /// <summary>Applies in-place rotate-half rotation to Q and K. Both must be <c>[B, numHeads, seqLen, headDim]</c>
    /// laid out contiguously. <paramref name="cosTable"/> and <paramref name="sinTable"/> must be <c>[seqLen, headDim]</c>
    /// from <see cref="BuildFreqs"/>.</summary>
    public void ApplyRotation(Tensor q, Tensor k, Tensor cosTable, Tensor sinTable, int batch, int numHeads, int seqLen)
    {
        int halfDim = _headDim / 2;
        float* qPtr = (float*)q.DataPointer;
        float* kPtr = (float*)k.DataPointer;
        float* cosPtr = (float*)cosTable.DataPointer;
        float* sinPtr = (float*)sinTable.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    long vecOff = (((long)b * numHeads + h) * seqLen + s) * _headDim;
                    long freqOff = (long)s * _headDim;
                    RotateHalf(qPtr + vecOff, cosPtr + freqOff, sinPtr + freqOff, halfDim);
                    RotateHalf(kPtr + vecOff, cosPtr + freqOff, sinPtr + freqOff, halfDim);
                }
            }
        }
    }

    /// <summary>Cosmos NTK factor: <c>scale^(dim/(dim-2))</c>. Returns 1.0 when <paramref name="dim"/> ≤ 2 to avoid div-by-zero.</summary>
    private static float NtkFactor(float scale, int dim)
    {
        if (dim <= 2 || MathF.Abs(scale - 1.0f) < 1e-12f) return 1.0f;
        return MathF.Pow(scale, (float)dim / (dim - 2));
    }

    /// <summary>Fills <paramref name="invFreqs"/> with <c>1 / theta^(2k / dim)</c>. Length must be <paramref name="dim"/>/2.</summary>
    private static void FillInvFreqs(Span<float> invFreqs, int dim, float theta)
    {
        int n = invFreqs.Length;
        for (int k = 0; k < n; k++)
        {
            float exponent = (float)(2 * k) / dim;
            invFreqs[k] = 1.0f / MathF.Pow(theta, exponent);
        }
    }

    /// <summary>Writes <c>cos(pos * invFreq[k])</c> / <c>sin(pos * invFreq[k])</c> into the cos/sin tables for one axis.</summary>
    private static void WriteAxisFreqs(float* cosPtr, float* sinPtr, long rowOff, int featOff, int pos, ReadOnlySpan<float> invFreqs)
    {
        int n = invFreqs.Length;
        for (int k = 0; k < n; k++)
        {
            float angle = pos * invFreqs[k];
            cosPtr[rowOff + featOff + k] = MathF.Cos(angle);
            sinPtr[rowOff + featOff + k] = MathF.Sin(angle);
        }
    }

    /// <summary>GPT-NeoX rotate-half: split the head into two halves [real | imag], rotate by (cos, sin) per element pairwise.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RotateHalf(float* vec, float* cos, float* sin, int halfDim)
    {
        for (int d = 0; d < halfDim; d++)
        {
            float xReal = vec[d];
            float xImag = vec[halfDim + d];
            vec[d] = xReal * cos[d] - xImag * sin[d];
            vec[halfDim + d] = xReal * sin[halfDim + d] + xImag * cos[halfDim + d];
        }
    }
}

using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Codecs;

/// <summary>Finite Scalar Quantization primitives (Mentzer et al. 2023), matching <c>lucidrains/vector-quantize-pytorch::FSQ</c> exactly.</summary>
/// <remarks>
/// Each axis of a continuous latent is bounded by a parity-aware tanh and rounded to one of <c>L_d</c> discrete
/// levels; the per-axis digits pack into a single integer via mixed-radix (base-<c>L</c>) place values. Vocabulary
/// size = <c>product(levels)</c>.
///
/// This is the canonical shared implementation. Cosmos-Tokenize1-DV8x16x16 uses <c>levels = [8,8,8,5,5,5]</c>
/// → 64,000 tokens (mixed-radix basis <c>[1,8,64,512,2560,12800]</c>). Spark-TTS / CosyVoice / other FSQ codecs
/// use their own level configs. Inference-only: no straight-through estimator; <c>round</c> uses banker's rounding
/// (<see cref="MidpointRounding.ToEven"/>), matching PyTorch.
///
/// <b>Dequantize normalization.</b> <see cref="Dequantize"/> maps digit <c>k ∈ [0, L)</c> back to
/// <c>(k − L/2) / (L/2)</c> (integer <c>L/2</c>), the lucidrains / Cosmos <c>indices_to_codes</c> convention. Note
/// this differs from <c>HartsyInference.Audio.Models.Codecs.Fsq</c>'s legacy dequant divisor <c>(L−1)/2</c> for even
/// levels — the two agree only on the (load-bearing) <see cref="Quantize"/> index path, which both share.
/// </remarks>
public static unsafe class Fsq
{
    /// <summary>Vocabulary size for the given per-axis level configuration (= product of levels).</summary>
    public static int VocabSize(ReadOnlySpan<int> levels)
    {
        long product = 1;
        for (int d = 0; d < levels.Length; d++)
        {
            if (levels[d] < 2) throw new ArgumentException($"levels[{d}]={levels[d]} must be >= 2.");
            product *= levels[d];
            if (product > int.MaxValue) throw new ArgumentException("vocab size overflows Int32.");
        }
        return (int)product;
    }

    /// <summary>Mixed-radix place values for packing digits into one integer: <c>basis[d]=basis[d-1]·levels[d-1]</c>, basis[0]=1.</summary>
    public static void Basis(ReadOnlySpan<int> levels, Span<int> basis)
    {
        if (basis.Length != levels.Length) throw new ArgumentException("basis length must match levels length.");
        basis[0] = 1;
        for (int d = 1; d < levels.Length; d++) basis[d] = basis[d - 1] * levels[d - 1];
    }

    /// <summary>Quantizes a continuous latent <c>z [B,T,D]</c> into integer codes <c>[B,T]</c> (lucidrains's <c>round_ste(bound(z))</c>).</summary>
    public static void Quantize(Tensor codes, Tensor z, ReadOnlySpan<int> levels)
    {
        if (z.Shape.Rank != 3) throw new ArgumentException($"z must be rank-3 [B, T, D], got {z.Shape}.");
        if (codes.Shape.Rank != 2) throw new ArgumentException($"codes must be rank-2 [B, T], got {codes.Shape}.");
        int b = (int)z.Shape[0];
        int t = (int)z.Shape[1];
        int d = (int)z.Shape[2];
        if (levels.Length != d) throw new ArgumentException($"levels.Length ({levels.Length}) must match D ({d}).");
        if ((int)codes.Shape[0] != b || (int)codes.Shape[1] != t)
            throw new ArgumentException($"codes shape mismatch: expected [{b}, {t}], got {codes.Shape}.");
        if (codes.DType != DType.I32) throw new ArgumentException($"codes must be Int32, got {codes.DType}.");

        Span<float> halfL = stackalloc float[d];
        Span<float> shift = stackalloc float[d];
        Span<int> halfLevel = stackalloc int[d];
        const float eps = 1e-6f;
        for (int dd = 0; dd < d; dd++)
        {
            int L = levels[dd];
            halfL[dd] = (L - 1) * (1f + eps) / 2f;
            float offset = (L % 2 == 0) ? 0.5f : 0f;
            shift[dd] = MathF.Atan(offset / halfL[dd]);
            halfLevel[dd] = L / 2;
        }

        Span<int> placeValue = stackalloc int[d];
        placeValue[0] = 1;
        for (int dd = 1; dd < d; dd++) placeValue[dd] = placeValue[dd - 1] * levels[dd - 1];

        float* zp = (float*)z.DataPointer;
        int* cp = (int*)codes.DataPointer;

        for (int bi = 0; bi < b; bi++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                int code = 0;
                int rowBase = (bi * t + ti) * d;
                for (int dd = 0; dd < d; dd++)
                {
                    int L = levels[dd];
                    int half = halfLevel[dd];
                    float offset = (L % 2 == 0) ? 0.5f : 0f;
                    float bounded = MathF.Tanh(zp[rowBase + dd] + shift[dd]) * halfL[dd] - offset;
                    int rounded = (int)MathF.Round(bounded, MidpointRounding.ToEven);
                    int shifted = rounded + half;
                    if (shifted < 0) shifted = 0;
                    else if (shifted >= L) shifted = L - 1;
                    code += shifted * placeValue[dd];
                }
                cp[bi * t + ti] = code;
            }
        }
    }

    /// <summary>Converts grid-quantized codes to indices; inverse of <see cref="Dequantize"/> (skips <see cref="Quantize"/>'s tanh bound).</summary>
    public static void CodesToIndices(Tensor codes, Tensor zHat, ReadOnlySpan<int> levels)
    {
        if (zHat.Shape.Rank != 3) throw new ArgumentException($"zHat must be rank-3 [B, T, D], got {zHat.Shape}.");
        if (codes.Shape.Rank != 2) throw new ArgumentException($"codes must be rank-2 [B, T], got {codes.Shape}.");
        int b = (int)zHat.Shape[0];
        int t = (int)zHat.Shape[1];
        int d = (int)zHat.Shape[2];
        if (levels.Length != d) throw new ArgumentException($"levels.Length ({levels.Length}) must match D ({d}).");
        if ((int)codes.Shape[0] != b || (int)codes.Shape[1] != t)
            throw new ArgumentException($"codes shape mismatch: expected [{b}, {t}], got {codes.Shape}.");
        if (codes.DType != DType.I32) throw new ArgumentException($"codes must be Int32, got {codes.DType}.");

        Span<int> halfLevel = stackalloc int[d];
        Span<int> placeValue = stackalloc int[d];
        placeValue[0] = 1;
        for (int dd = 0; dd < d; dd++) halfLevel[dd] = levels[dd] / 2;
        for (int dd = 1; dd < d; dd++) placeValue[dd] = placeValue[dd - 1] * levels[dd - 1];

        float* zp = (float*)zHat.DataPointer;
        int* cp = (int*)codes.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ti = 0; ti < t; ti++)
            {
                int code = 0;
                int rowBase = (bi * t + ti) * d;
                for (int dd = 0; dd < d; dd++)
                {
                    int L = levels[dd], half = halfLevel[dd];
                    int digit = (int)MathF.Round(zp[rowBase + dd] * half + half, MidpointRounding.ToEven);
                    if (digit < 0) digit = 0; else if (digit >= L) digit = L - 1;
                    code += digit * placeValue[dd];
                }
                cp[bi * t + ti] = code;
            }
    }

    /// <summary>Inverse of <see cref="Quantize"/>: codes <c>[B,T]</c> → vectors <c>[B,T,D]</c>, each axis as <c>(k−L/2)/(L/2)</c>.</summary>
    public static void Dequantize(Tensor zHat, Tensor codes, ReadOnlySpan<int> levels)
    {
        if (zHat.Shape.Rank != 3) throw new ArgumentException($"zHat must be rank-3 [B, T, D], got {zHat.Shape}.");
        if (codes.Shape.Rank != 2) throw new ArgumentException($"codes must be rank-2 [B, T], got {codes.Shape}.");
        int b = (int)zHat.Shape[0];
        int t = (int)zHat.Shape[1];
        int d = (int)zHat.Shape[2];
        if (levels.Length != d) throw new ArgumentException($"levels.Length ({levels.Length}) must match D ({d}).");
        if ((int)codes.Shape[0] != b || (int)codes.Shape[1] != t)
            throw new ArgumentException($"codes shape mismatch: expected [{b}, {t}], got {codes.Shape}.");
        if (codes.DType != DType.I32) throw new ArgumentException($"codes must be Int32, got {codes.DType}.");

        Span<float> invHalf = stackalloc float[d];
        Span<int> halfLevel = stackalloc int[d];
        for (int dd = 0; dd < d; dd++)
        {
            int L = levels[dd];
            int half = L / 2;
            invHalf[dd] = half > 0 ? 1f / half : 1f;
            halfLevel[dd] = half;
        }

        Span<int> placeValue = stackalloc int[d];
        placeValue[0] = 1;
        for (int dd = 1; dd < d; dd++) placeValue[dd] = placeValue[dd - 1] * levels[dd - 1];

        int* cp = (int*)codes.DataPointer;
        float* zp = (float*)zHat.DataPointer;

        for (int bi = 0; bi < b; bi++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                int code = cp[bi * t + ti];
                int rowBase = (bi * t + ti) * d;
                for (int dd = 0; dd < d; dd++)
                {
                    int digit = (code / placeValue[dd]) % levels[dd];
                    zp[rowBase + dd] = (digit - halfLevel[dd]) * invHalf[dd];
                }
            }
        }
    }
}

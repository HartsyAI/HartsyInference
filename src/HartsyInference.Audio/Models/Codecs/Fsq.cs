using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs;

/// <summary>Finite Scalar Quantization primitives — a deterministic round-to-grid alternative to VQ-VAE's learned codebooks.</summary>
/// <remarks>
/// Each axis of the continuous latent is bounded by tanh and rounded to one of
/// <c>L_d</c> discrete values, then the per-axis "digits" pack into a single integer
/// code via positional base-L encoding.
///
/// <para>Vocabulary size = <c>product(levels)</c>. Typical configurations:</para>
/// <list type="bullet">
///   <item>Spark-TTS BiCodec global FSQ: <c>levels = [8, 8, 8, 5, 5]</c> → 12800 tokens</item>
///   <item>Mimi acoustic codec: per-codebook configurations vary</item>
///   <item>CosyVoice S3Tokenizer: <c>levels = [4, 4, 4, 4, 4]</c> → 1024 tokens</item>
/// </list>
///
/// <para>Formulation follows
/// <c>lucidrains/vector-quantize-pytorch::FSQ</c> (Mentzer et al. 2023, "Finite Scalar
/// Quantization: VQ-VAE Made Simple"). Both odd and even <c>L</c> are supported via the
/// parity-aware <c>Bound</c> step that offsets the tanh so the rounded grid contains
/// exactly <c>L</c> points centered around zero.</para>
///
/// <para>Inference-only: no straight-through estimator (no gradient), and the
/// <c>round</c> uses C#'s default banker's rounding which matches PyTorch.</para>
/// </remarks>
public static unsafe class Fsq
{
    /// <summary>Vocabulary size for the given per-axis level configuration.</summary>
    public static int VocabSize(ReadOnlySpan<int> levels) => Core.Codecs.Fsq.VocabSize(levels);

    /// <summary>Quantizes a continuous latent into integer codes. Input <paramref name="z"/> is channels-last <c>[B, T, D]</c>; output <paramref name="codes"/> is <c>[B, T]</c> Int32; <paramref name="levels"/> length must equal <c>D</c>. Delegates to the canonical <see cref="Core.Codecs.Fsq.Quantize"/> — the tanh-bound + mixed-radix index packing is byte-identical to the shared implementation.</summary>
    public static void Quantize(Tensor codes, Tensor z, ReadOnlySpan<int> levels)
        => Core.Codecs.Fsq.Quantize(codes, z, levels);

    /// <summary>Inverse — turns integer codes back into the continuous quantized vector <paramref name="zHat"/>, normalized to <c>[-1, 1]</c> per axis: integer digit <c>k ∈ [0, L)</c> maps to <c>(k - L/2) / halfL</c>. Useful when the decoder consumes the continuous quantized vector rather than the integer code itself.</summary>
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

        Span<float> invHalfL = stackalloc float[d];
        Span<int> halfLevel = stackalloc int[d];
        for (int dd = 0; dd < d; dd++)
        {
            int L = levels[dd];
            float halfL = (L - 1) / 2f;
            invHalfL[dd] = halfL > 0 ? 1f / halfL : 1f;
            halfLevel[dd] = L / 2;
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
                    int shifted = (code / placeValue[dd]) % levels[dd];
                    int zInt = shifted - halfLevel[dd];
                    zp[rowBase + dd] = zInt * invHalfL[dd];
                }
            }
        }
    }
}

using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Dia;

/// <summary>Adapts the published Dia-1.6B checkpoint (nari-labs native <c>DenseGeneral</c> layout) to the
/// 2D <c>[out, in]</c> weights the C# attention/MLP consume via <see cref="WhisperOps.ProjectLinear"/>.
/// The native projections are stored "in-major": <c>q/k/v_proj</c> as <c>[in, heads, head_dim]</c>,
/// <c>o_proj</c> as <c>[heads, head_dim, out]</c>, the fused MLP <c>wi_fused</c> as <c>[in, 2, ffn]</c>, and
/// <c>wo</c> as <c>[ffn, out]</c>. Each is flattened to 2D and transposed to <c>[out, in]</c>; the MLP keys
/// are renamed (<c>wi_fused</c>→<c>gate_up_proj</c>, <c>wo</c>→<c>down_proj</c>). Norms / embeddings pass
/// through. Also provides split-half (NeoX) RoPE, which Dia uses (<c>chunk(x, 2)</c>).</summary>
public static unsafe class DiaWeights
{
    /// <summary>Returns a new dict with every Dia projection transposed to <c>[out, in]</c> and the MLP keys
    /// renamed, leaving everything else untouched.</summary>
    public static Dictionary<string, Tensor> Adapt(IReadOnlyDictionary<string, Tensor> w)
    {
        Dictionary<string, Tensor> o = new(w.Count);
        foreach ((string k, Tensor v) in w)
        {
            if (k.EndsWith(".q_proj.weight", StringComparison.Ordinal) ||
                k.EndsWith(".k_proj.weight", StringComparison.Ordinal) ||
                k.EndsWith(".v_proj.weight", StringComparison.Ordinal))
                o[k] = TransposeFlat(v, leadingDims: 1);          // [in, h, k] → [h*k, in]
            else if (k.EndsWith(".o_proj.weight", StringComparison.Ordinal))
                o[k] = TransposeFlat(v, leadingDims: v.Shape.Rank - 1); // [h, k, out] → [out, h*k]
            else if (k.EndsWith(".mlp.wi_fused.weight", StringComparison.Ordinal))
                o[k[..^"wi_fused.weight".Length] + "gate_up_proj.weight"] = TransposeFlat(v, leadingDims: 1); // [in,2,ffn]→[2ffn,in]
            else if (k.EndsWith(".mlp.wo.weight", StringComparison.Ordinal))
                o[k[..^"wo.weight".Length] + "down_proj.weight"] = TransposeFlat(v, leadingDims: 1);          // [ffn,out]→[out,ffn]
            else if (k.EndsWith("logits_dense.weight", StringComparison.Ordinal))
                o[k] = TransposeFlat(v, leadingDims: 1);          // [in, channels, vocab] → [channels*vocab, in]
            else
                o[k] = v;
        }
        return o;
    }

    // Flattens the tensor to 2D [A, B] (A = product of the first leadingDims dims, B = the rest) then
    // transposes to [B, A].
    private static Tensor TransposeFlat(Tensor src, int leadingDims)
    {
        Tensor f = WhisperOps.EnsureF32(src);
        long a = 1; for (int i = 0; i < leadingDims; i++) a *= f.Shape[i];
        long b = f.ElementCount / a;
        Tensor o = new(new TensorShape(b, a), DType.F32);
        float* sp = (float*)f.DataPointer, op = (float*)o.DataPointer;
        for (long i = 0; i < a; i++)
            for (long j = 0; j < b; j++)
                op[j * a + i] = sp[i * b + j];
        return o;
    }

    /// <summary>Split-half (NeoX) RoPE in place on <c>[1, H, T, D]</c>: dim <c>d</c> pairs with <c>d + D/2</c>,
    /// using cos/sin tables of shape <c>[maxPos, D/2]</c>. Dia uses this (not the GPT-J interleaved form).</summary>
    public static void RopeSplitHalfInPlace(Tensor t, int heads, int seqLen, int headDim, int posStart,
        float[] cos, float[] sin)
    {
        int half = headDim / 2;
        float* p = (float*)t.DataPointer;
        for (int h = 0; h < heads; h++)
            for (int s = 0; s < seqLen; s++)
            {
                long baseOff = ((long)h * seqLen + s) * headDim;
                int row = (posStart + s) * half;
                for (int d = 0; d < half; d++)
                {
                    float c = cos[row + d], sn = sin[row + d];
                    float x1 = p[baseOff + d], x2 = p[baseOff + d + half];
                    p[baseOff + d] = x1 * c - x2 * sn;
                    p[baseOff + d + half] = x2 * c + x1 * sn;
                }
            }
    }

    /// <summary>GPT-J interleaved RoPE: rotates adjacent element pairs <c>(2d, 2d+1)</c> using the same
    /// <paramref name="cos"/>/<paramref name="sin"/> tables (identical inv-freqs, only the pairing differs from
    /// <see cref="RopeSplitHalfInPlace"/>). Used by Zonos (<c>rotary_emb_interleaved=true</c>).</summary>
    public static void RopeInterleavedInPlace(Tensor t, int heads, int seqLen, int headDim, int posStart,
        float[] cos, float[] sin)
    {
        int half = headDim / 2;
        float* p = (float*)t.DataPointer;
        for (int h = 0; h < heads; h++)
            for (int s = 0; s < seqLen; s++)
            {
                long baseOff = ((long)h * seqLen + s) * headDim;
                int row = (posStart + s) * half;
                for (int d = 0; d < half; d++)
                {
                    float c = cos[row + d], sn = sin[row + d];
                    float x1 = p[baseOff + 2 * d], x2 = p[baseOff + 2 * d + 1];
                    p[baseOff + 2 * d] = x1 * c - x2 * sn;
                    p[baseOff + 2 * d + 1] = x2 * c + x1 * sn;
                }
            }
    }
}

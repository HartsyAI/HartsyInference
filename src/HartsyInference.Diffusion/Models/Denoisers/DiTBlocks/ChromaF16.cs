using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>F16-activation support for the Chroma block loop (HARTSY_DIT_F16 opt-in).
///
/// <para>Chroma is Flux-derived, and Flux's known F16 failure is <b>residual-stream growth</b>: the token
/// stream accumulated across 19 double + 38 single blocks exceeds F16's 65504 in late blocks. Unlike
/// Z-Image there is no sandwich norm to damp against — instead the ENTIRE residual stream is scaled by
/// <see cref="ResidualDamp"/>, which is exact (not approximate) because every read of the stream goes
/// through a scale-invariant no-affine LayerNorm:</para>
///
/// <para>Damp the two token embedders (<c>x_embedder</c>, <c>context_embedder</c>) and every branch-output
/// projection (<c>attn.to_out.0</c>, <c>attn.to_add_out</c>, <c>ff*.net.2</c>, single-block <c>proj_out</c>)
/// by <c>c</c>. Then the residual recurrence <c>x' = x + gate·f(LN(x))</c> maps to <c>c·x' = c·x +
/// gate·(c·f(LN(c·x)))</c> — LN(c·x) ≡ LN(x), so every branch INPUT (Q/K/V/MLP projections, attention,
/// GELU) sees bit-comparable baseline-scale values while the stream itself rides at <c>c</c> scale. The
/// final <c>ChromaAdaLayerNormContinuousPruned</c> starts with the same no-affine LayerNorm, so <c>c</c>
/// cancels exactly before <c>proj_out</c>: the velocity is unchanged. <c>c</c> is a power of two, so no
/// relative precision is lost anywhere. Weight damping folds into the GEMM alpha via
/// <see cref="Tensor.Fp8ScaleFactor"/> (any weight dtype, zero extra kernels); biases are value-scaled
/// copies (the GEMM epilogue adds bias after alpha).</para></summary>
public static unsafe class ChromaF16
{
    /// <summary>Residual-stream scale for the F16 path. 1/32 puts a worst-case ~65k late-block residual at
    /// ~2k, far inside F16 range, while text/image token values (O(1-100) post-embed) keep full relative
    /// precision (power-of-two exponent shift).</summary>
    public const float ResidualDamp = 1.0f / 32.0f;

    /// <summary>HARTSY_CHROMA_F16TRACE=1: logs min/max/nan/inf of key block intermediates for the first few
    /// forwards to locate F16 overflow sites. Each probe D2H-drains the tensor (very slow); debug only.</summary>
    private static readonly bool TraceEnabled = EngineKnobs.ChromaF16trace.Value;
    private static int _traceCallsLeft = TraceEnabled ? 2000 : 0;

    /// <summary>Returns a value-scaled F32 copy of a bias tensor (weight damping rides the GEMM alpha, but the
    /// epilogue adds bias AFTER alpha, so the bias must carry the damp factor in its values).</summary>
    public static Tensor DampBias(Tensor bias, float damp)
    {
        Tensor f32 = bias.DType == DType.F32 ? CopyF32(bias) : bias.CastTo(DType.F32);
        float* p = (float*)f32.DataPointer;
        long n = f32.Shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] *= damp;
        return f32;
    }

    /// <summary>Debug probe (see <see cref="TraceEnabled"/>): min/max/nan/inf of a tensor, F16 or F32.</summary>
    public static void Probe(string name, Tensor t)
    {
        if (_traceCallsLeft <= 0) return;
        _traceCallsLeft--;
        float min = float.MaxValue, max = float.MinValue;
        long nan = 0, inf = 0;
        long n = t.Shape.ElementCount;
        if (t.DType == DType.F16)
        {
            Half* p = (Half*)t.DataPointer;
            for (long i = 0; i < n; i++)
            {
                float v = (float)p[i];
                if (float.IsNaN(v)) { nan++; continue; }
                if (float.IsInfinity(v)) { inf++; continue; }
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        else
        {
            float* p = (float*)t.DataPointer;
            for (long i = 0; i < n; i++)
            {
                float v = p[i];
                if (float.IsNaN(v)) { nan++; continue; }
                if (float.IsInfinity(v)) { inf++; continue; }
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        Logs.Info($"[chroma-f16trace] {name} [{t.DType.Name}] min={min:F3} max={max:F3} nan={nan} inf={inf} n={n}");
    }

    private static Tensor CopyF32(Tensor src)
    {
        Tensor copy = new Tensor(src.Shape, DType.F32);
        long bytes = src.Shape.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)copy.DataPointer, bytes, bytes);
        return copy;
    }
}

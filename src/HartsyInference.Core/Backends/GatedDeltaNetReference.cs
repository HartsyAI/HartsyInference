using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Host reference for the Gated-DeltaNet prefill ops (<see cref="IBackend.SsmConvSplitPrefill"/>,
/// <see cref="IBackend.SsmDeltaPrefill"/>): Qwen3.5's linear-attention layer as llama.cpp's
/// <c>build_delta_net_autoregressive</c> runs it, one token at a time. Heads are independent, so the recurrence fans out
/// over value heads; within a head every sum keeps the sequential order of the single-threaded loop it replaced.</summary>
public static class GatedDeltaNetReference
{
    /// <summary>Above this, softplus is the identity in F32.</summary>
    private const float SoftplusLinearAbove = 20f;

    /// <summary>Validates <see cref="IBackend.SsmConvSplitPrefill"/>'s operands and returns the token count.</summary>
    public static int ValidateConvSplit(Tensor q, Tensor k, Tensor v, Tensor qkvMixed, Tensor history, Tensor convW,
        int kernel, int kHeads, int skDim, int vHeads, int svDim)
    {
        RequireF32(nameof(IBackend.SsmConvSplitPrefill), q, k, v, qkvMixed, history, convW);
        if (kernel < 1 || kHeads < 1 || skDim < 1 || vHeads < 1 || svDim < 1)
            throw new ArgumentException($"SsmConvSplitPrefill needs positive dims, got kernel={kernel}, heads={kHeads}/{vHeads}, dims={skDim}/{svDim}.");
        long keyDim = (long)kHeads * skDim, valueDim = (long)vHeads * svDim, convDim = 2 * keyDim + valueDim;
        if (qkvMixed.ElementCount % convDim != 0)
            throw new ArgumentException($"SsmConvSplitPrefill qkvMixed {qkvMixed.Shape} is not a whole number of {convDim}-wide rows.", nameof(qkvMixed));
        long seq = qkvMixed.ElementCount / convDim;
        if (q.ElementCount != seq * keyDim || k.ElementCount != seq * keyDim || v.ElementCount != seq * valueDim)
            throw new ArgumentException($"SsmConvSplitPrefill outputs q {q.Shape}, k {k.Shape}, v {v.Shape} do not hold {seq} tokens.", nameof(q));
        if (history.ElementCount != (kernel - 1) * convDim || convW.ElementCount != convDim * kernel)
            throw new ArgumentException($"SsmConvSplitPrefill history {history.Shape} / taps {convW.Shape} do not match kernel {kernel} over {convDim} channels.", nameof(history));
        return checked((int)seq);
    }

    /// <summary>Validates <see cref="IBackend.SsmDeltaPrefill"/>'s operands and returns the token count.</summary>
    public static int ValidateDeltaScan(Tensor output, Tensor state, Tensor q, Tensor k, Tensor v, Tensor z, Tensor alphaRaw,
        Tensor betaRaw, Tensor dtBias, Tensor ssmA, Tensor normW, int hv, int sv, int sk, int repeat)
    {
        RequireF32(nameof(IBackend.SsmDeltaPrefill), output, state, q, k, v, z, alphaRaw, betaRaw, dtBias, ssmA, normW);
        if (hv < 1 || sv < 1 || sk < 1 || repeat < 1 || hv % repeat != 0)
            throw new ArgumentException($"SsmDeltaPrefill needs positive dims with hv divisible by repeat, got hv={hv}, sv={sv}, sk={sk}, repeat={repeat}.");
        long valueDim = (long)hv * sv, keyDim = (long)(hv / repeat) * sk;
        if (alphaRaw.ElementCount % hv != 0)
            throw new ArgumentException($"SsmDeltaPrefill alphaRaw {alphaRaw.Shape} is not a whole number of {hv}-wide rows.", nameof(alphaRaw));
        long seq = alphaRaw.ElementCount / hv;
        if (betaRaw.ElementCount != seq * hv || output.ElementCount != seq * valueDim || v.ElementCount != seq * valueDim
            || z.ElementCount != seq * valueDim || q.ElementCount != seq * keyDim || k.ElementCount != seq * keyDim)
        {
            throw new ArgumentException($"SsmDeltaPrefill operands do not all hold {seq} tokens of {hv}×{sv} values over {keyDim} keys.", nameof(output));
        }
        if (state.ElementCount != valueDim * sk || dtBias.ElementCount != hv || ssmA.ElementCount != hv || normW.ElementCount != sv)
            throw new ArgumentException($"SsmDeltaPrefill state {state.Shape}, dtBias, ssmA or normW does not match hv={hv}, sv={sv}, sk={sk}.", nameof(state));
        return checked((int)seq);
    }

    /// <summary>Host <see cref="IBackend.SsmConvSplitPrefill"/>.</summary>
    public static unsafe void ConvSplit(Tensor q, Tensor k, Tensor v, Tensor qkvMixed, Tensor history, Tensor convW,
        int kernel, int kHeads, int skDim, int vHeads, int svDim, float eps, float qScale)
    {
        int seq = ValidateConvSplit(q, k, v, qkvMixed, history, convW, kernel, kHeads, skDim, vHeads, svDim);
        int keyDim = kHeads * skDim, valueDim = vHeads * svDim, convDim = 2 * keyDim + valueDim;
        float* mixed = (float*)qkvMixed.DataPointer;
        float* hist = (float*)history.DataPointer;
        float* taps = (float*)convW.DataPointer;
        float* qOut = (float*)q.DataPointer;
        float* kOut = (float*)k.DataPointer;
        float* vOut = (float*)v.DataPointer;
        Parallel.For(0, seq, s =>
        {
            for (int c = 0; c < convDim; c++)
            {
                float acc = 0f;
                for (int j = 0; j < kernel; j++)
                {
                    int combined = s + j;
                    float src = combined < kernel - 1 ? hist[(long)combined * convDim + c] : mixed[(long)(combined - (kernel - 1)) * convDim + c];
                    acc += taps[(long)c * kernel + j] * src;
                }
                float act = acc / (1f + MathF.Exp(-acc));
                if (c < keyDim) qOut[(long)s * keyDim + c] = act;
                else if (c < 2 * keyDim) kOut[(long)s * keyDim + c - keyDim] = act;
                else vOut[(long)s * valueDim + c - 2 * keyDim] = act;
            }
            for (int h = 0; h < kHeads; h++)
            {
                L2Normalize(qOut + (long)s * keyDim + (long)h * skDim, skDim, eps, qScale);
                L2Normalize(kOut + (long)s * keyDim + (long)h * skDim, skDim, eps, 1f);
            }
        });
        AdvanceHistory(hist, mixed, seq, kernel, convDim);
    }

    /// <summary>Host <see cref="IBackend.SsmDeltaPrefill"/>.</summary>
    public static unsafe void DeltaScan(Tensor output, Tensor state, Tensor q, Tensor k, Tensor v, Tensor z, Tensor alphaRaw,
        Tensor betaRaw, Tensor dtBias, Tensor ssmA, Tensor normW, int hv, int sv, int sk, int repeat, float eps)
    {
        int seq = ValidateDeltaScan(output, state, q, k, v, z, alphaRaw, betaRaw, dtBias, ssmA, normW, hv, sv, sk, repeat);
        int keyDim = hv / repeat * sk, valueDim = hv * sv;
        float* st = (float*)state.DataPointer;
        float* qp = (float*)q.DataPointer;
        float* kp = (float*)k.DataPointer;
        float* vp = (float*)v.DataPointer;
        float* zp = (float*)z.DataPointer;
        float* alpha = (float*)alphaRaw.DataPointer;
        float* beta = (float*)betaRaw.DataPointer;
        float* bias = (float*)dtBias.DataPointer;
        float* decay = (float*)ssmA.DataPointer;
        float* nw = (float*)normW.DataPointer;
        float* o = (float*)output.DataPointer;
        Parallel.For(0, hv, h =>
        {
            int kh = h / repeat;
            float* hs = st + (long)h * sv * sk;
            float* rowDot = stackalloc float[sv];
            for (int s = 0; s < seq; s++)
            {
                float av = alpha[(long)s * hv + h] + bias[h];
                float softplus = av > SoftplusLinearAbove ? av : MathF.Log(1f + MathF.Exp(av));
                float gExp = MathF.Exp(softplus * decay[h]);
                float b = 1f / (1f + MathF.Exp(-beta[(long)s * hv + h]));
                float* kv = kp + (long)s * keyDim + (long)kh * sk;
                float* qv = qp + (long)s * keyDim + (long)kh * sk;
                float* vv = vp + (long)s * valueDim + (long)h * sv;
                float* ov = o + (long)s * valueDim + (long)h * sv;
                for (long i = 0; i < (long)sv * sk; i++) hs[i] *= gExp;
                for (int row = 0; row < sv; row++)
                {
                    float* rb = hs + (long)row * sk;
                    float acc = 0f;
                    for (int col = 0; col < sk; col++) acc += rb[col] * kv[col];
                    rowDot[row] = acc;
                }
                for (int row = 0; row < sv; row++)
                {
                    float* rb = hs + (long)row * sk;
                    float delta = (vv[row] - rowDot[row]) * b;
                    for (int col = 0; col < sk; col++) rb[col] += kv[col] * delta;
                }
                for (int row = 0; row < sv; row++)
                {
                    float* rb = hs + (long)row * sk;
                    float acc = 0f;
                    for (int col = 0; col < sk; col++) acc += rb[col] * qv[col];
                    ov[row] = acc;
                }
                GatedNorm(ov, zp + (long)s * valueDim + (long)h * sv, nw, sv, eps);
            }
        });
    }

    /// <summary>Shifts the conv carry left by this call's rows and appends its last pre-conv rows.</summary>
    private static unsafe void AdvanceHistory(float* hist, float* mixed, int seq, int kernel, int convDim)
    {
        if (kernel <= 1)
            return;
        int carry = Math.Min(kernel - 1, seq);
        for (int r = 0; r < kernel - 1 - carry; r++)
            Buffer.MemoryCopy(hist + (long)(r + carry) * convDim, hist + (long)r * convDim, convDim * 4L, convDim * 4L);
        for (int r = 0; r < carry; r++)
        {
            Buffer.MemoryCopy(mixed + (long)(seq - carry + r) * convDim, hist + (long)(kernel - 1 - carry + r) * convDim,
                convDim * 4L, convDim * 4L);
        }
    }

    /// <summary><c>x /= sqrt(Σx² + eps)</c>, then <c>x ·= postScale</c> (no learned weight).</summary>
    private static unsafe void L2Normalize(float* x, int dim, float eps, float postScale)
    {
        float sumSq = 0f;
        for (int c = 0; c < dim; c++) sumSq += x[c] * x[c];
        float inv = 1f / MathF.Sqrt(sumSq + eps);
        for (int c = 0; c < dim; c++) x[c] = x[c] * inv * postScale;
    }

    /// <summary>Qwen3.5's gated norm, norm first then gate: <c>o = rmsnorm(o)·normW·silu(z)</c>.</summary>
    private static unsafe void GatedNorm(float* o, float* z, float* normW, int dim, float eps)
    {
        float var = 0f;
        for (int c = 0; c < dim; c++) var += o[c] * o[c];
        float inv = 1f / MathF.Sqrt(var / dim + eps);
        for (int c = 0; c < dim; c++)
        {
            float zv = z[c];
            o[c] = o[c] * inv * normW[c] * (zv / (1f + MathF.Exp(-zv)));
        }
    }

    private static void RequireF32(string op, params Tensor[] tensors)
    {
        foreach (Tensor t in tensors)
        {
            ArgumentNullException.ThrowIfNull(t);
            if (t.DType != DType.F32)
                throw new NotSupportedException($"{op} takes F32 tensors only, got {t.DType} {t.Shape}.");
        }
    }
}

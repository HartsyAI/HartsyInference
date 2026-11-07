using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Activations, gated linear units and the fused linear-plus-activation epilogues.
public partial interface IBackend
{
    /// <summary>DIAMOND pixel quantize to 256 levels: <c>floor((clamp(v,−1,1)+1)·127.5)/127.5 − 1</c>.</summary>
    unsafe void PixelQuantize(Tensor output, Tensor input)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32) throw new NotSupportedException("PixelQuantize default fallback only supports F32.");
        long n = input.ElementCount;
        float* ip = (float*)input.DataPointer, op = (float*)output.DataPointer;
        for (long i = 0; i < n; i++)
        {
            float v = ip[i];
            if (v > 1f) v = 1f; else if (v < -1f) v = -1f;
            int b = (int)((v + 1f) * 0.5f * 255f);
            op[i] = b / 255f * 2f - 1f;
        }
    }

    /// <summary>GEGLU with exact (erf) GELU gate: <c>output[rows,inner] = proj[:,:inner] · gelu_erf(proj[:,inner:])</c>, fused split+gate.</summary>
    unsafe void GegluErf(Tensor output, Tensor proj, long rows, int inner)
        => GegluErfReference(output, proj, rows, inner);

    /// <summary>The managed <see cref="GegluErf"/> body, callable directly. A backend override must
    /// call THIS to fall back — <c>((IBackend)this).GegluErf(...)</c> re-enters the override and
    /// recurses until the stack overflows.</summary>
    static unsafe void GegluErfReference(Tensor output, Tensor proj, long rows, int inner)
    {
        if (output.DType != DType.F32 || proj.DType != DType.F32)
            throw new NotSupportedException("GegluErf default fallback only supports F32.");
        float* pp = (float*)proj.DataPointer, gp = (float*)output.DataPointer;
        for (long row = 0; row < rows; row++)
        {
            float* hin = pp + row * 2 * inner, gin = hin + inner, gout = gp + row * inner;
            for (int i = 0; i < inner; i++)
            {
                float g = gin[i];
                gout[i] = hin[i] * (0.5f * g * (1f + Erf(g * 0.70710678118654752440f)));
            }
        }
    }

    /// <summary>Exact (erf) GELU: <c>out = 0.5·x·(1+erf(x/√2))</c>, PyTorch's <c>nn.GELU()</c>; <see cref="Gelu"/> is the tanh approx.</summary>
    unsafe void GeluErf(Tensor output, Tensor input)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException($"GeluErf default fallback only supports F32 — got input={input.DType}, output={output.DType}.");
        float* pIn = (float*)input.DataPointer;
        float* pOut = (float*)output.DataPointer;
        long count = output.ElementCount;
        for (long i = 0; i < count; i++)
        {
            float x = pIn[i];
            pOut[i] = 0.5f * x * (1f + Erf(x * 0.70710678118654752440f));
        }
    }

    /// <summary>Abramowitz-Stegun 7.1.26 rational polynomial approximation of the Gauss error function (max error ~1.5e-7).</summary>
    private static float Erf(float x)
    {
        int sign = x < 0 ? -1 : 1;
        x = MathF.Abs(x);
        float t = 1f / (1f + 0.3275911f * x);
        float y = 1f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * MathF.Exp(-x * x);
        return sign * y;
    }

    /// <summary>MoE expert combine, in-place: <c>inout += gate[expertIndex] * value</c>, gate from <see cref="MoeTopKGate"/>'s dense table.</summary>
    unsafe void RowGatedAccumulate(Tensor inout, Tensor value, Tensor gate, int numExperts, int expertIndex)
    {
        if (inout.DType != DType.F32 || value.DType != DType.F32 || gate.DType != DType.F32)
            throw new NotSupportedException("RowGatedAccumulate default fallback only supports F32.");
        int dim = (int)inout.Shape[inout.Shape.Rank - 1];
        long rows = inout.ElementCount / dim;
        float* op = (float*)inout.DataPointer;
        float* vp = (float*)value.DataPointer;
        float* gp = (float*)gate.DataPointer;
        for (long r = 0; r < rows; r++)
        {
            float g = gp[r * numExperts + expertIndex];
            if (g == 0f) continue;
            long off = r * dim;
            for (int d = 0; d < dim; d++)
                op[off + d] += g * vp[off + d];
        }
    }

    /// <summary>Gated-FFN epilogue that may emit a Q8_1 sidecar (<see cref="RmsNormEmitQ8"/>); semantically = <see cref="GluActivate"/>.</summary>
    void GluActivateEmitQ8(Tensor output, Tensor gateUp, int ff, bool gelu)
        => GluActivate(output, gateUp, ff, gelu);

    /// <summary>Fused gated-FFN epilogue: <c>output[r,i] = act(gateUp[r,i])·gateUp[r,ff+i]</c> over the concatenated [gate|up] projection.</summary>
    /// <param name="gelu">false = SiLU (SwiGLU), true = GELU-tanh (GeGLU).</param>
    void GluActivate(Tensor output, Tensor gateUp, int ff, bool gelu)
    {
        long rows = gateUp.ElementCount / (2L * ff);
        TensorShape half = new(1, (int)rows, ff);
        Tensor gate = new(half, DType.F32);
        Tensor up = new(half, DType.F32);
        try
        {
            SliceLastDim(gate, gateUp, 0);
            SliceLastDim(up, gateUp, ff);
            if (gelu) Gelu(gate, gate);
            else Silu(gate, gate);
            Mul(output, gate, up);
        }
        finally
        {
            gate.Dispose();
            up.Dispose();
        }
    }

    /// <summary>Linear immediately followed by GELU. Backends may fold the activation into the GEMM epilogue;
    /// the default runs the two ops in sequence, which is what every caller did before.</summary>
    unsafe void LinearGelu(Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        Linear(output, input, weight, bias);
        Gelu(output, output);
    }

    /// <summary>LTX-2's per-head output gate followed by a projection. A backend whose Linear already walks the
    /// activation (to quantize it) can apply the gate in that pass instead of in one of its own; the default is
    /// the two ops in sequence, which is what the caller did before and which mutates <paramref name="input"/>
    /// in place.</summary>
    unsafe void LinearHeadGated(Tensor output, Tensor input, Tensor weight, Tensor? bias,
        Tensor gateLogits, int heads, int headDim)
    {
        Ltx2HeadGate(input, gateLogits, (int)input.Shape[0], heads, headDim);
        Linear(output, input, weight, bias);
    }

    /// <summary>LTX-2 per-head output gate, IN PLACE: <c>x[s, h*headDim+d] *= 2*sigmoid(logits[s,h])</c>.
    /// Replaces expanding the per-(token,head) scalar to a full <c>[seq, heads*headDim]</c> tensor through a
    /// constant 0/1 GEMM and then multiplying.</summary>
    unsafe void Ltx2HeadGate(Tensor x, Tensor logits, int seq, int heads, int headDim)
    {
        if (x.DType != DType.F32 || logits.DType != DType.F32)
            throw new NotSupportedException($"Ltx2HeadGate default fallback only supports F32, got x={x.DType}, logits={logits.DType}.");
        float* p = (float*)x.DataPointer;
        float* g = (float*)logits.DataPointer;
        int w = heads * headDim;
        for (int s = 0; s < seq; s++)
            for (int h = 0; h < heads; h++)
            {
                float lg = g[(long)s * heads + h];
                float sig = lg >= 0f ? 1f / (1f + MathF.Exp(-lg)) : MathF.Exp(lg) / (1f + MathF.Exp(lg));
                long baseIdx = (long)s * w + (long)h * headDim;
                for (int d = 0; d < headDim; d++) p[baseIdx + d] *= 2f * sig;
            }
    }

    /// <summary>GELU activation using the standard tanh approximation; use <see cref="GeluErf"/> for the exact erf form.</summary>
    void Gelu(Tensor output, Tensor input);

    /// <summary>SiLU activation (x * sigmoid(x)).</summary>
    void Silu(Tensor output, Tensor input);

    /// <summary>Mish activation (x * tanh(softplus(x))) — the Matcha/CosyVoice CFM decoder's block activation.</summary>
    unsafe void Mish(Tensor output, Tensor input)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("Mish default fallback only supports F32.");
        long count = output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (long i = 0; i < count; i++)
        {
            float x = pIn[i];
            float sp = x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
            pOut[i] = x * MathF.Tanh(sp);
        }
    }

    /// <summary>Sigmoid activation (1 / (1 + exp(-x))). Used by LSTM gating.</summary>
    void Sigmoid(Tensor output, Tensor input);

    /// <summary>Hyperbolic tangent activation. Used by LSTM cell update/output and several vocoders (Mish, snake-bias).</summary>
    void Tanh(Tensor output, Tensor input);

    /// <summary>ELU: <c>x if x &gt;= 0 else alpha * (exp(x) - 1)</c>. Used by SEANet residual blocks in EnCodec/DAC (alpha typically 1.0).</summary>
    void Elu(Tensor output, Tensor input, float alpha);

    /// <summary>Leaky ReLU: <c>x if x &gt;= 0 else slope * x</c>. Kokoro/StyleTTS 2 and HiFi-GAN MRF blocks use slope=0.2.</summary>
    void LeakyRelu(Tensor output, Tensor input, float slope);

    /// <summary>Snake: <c>x + (sin(alpha*x))^2 / divisor</c>; divisor=alpha (Oobleck VAE) or beta+1e-8 when supplied (BigVGAN snake-beta).</summary>
    void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta);

    /// <summary>Parametric ReLU: <c>x if x&gt;=0 else alpha*x</c>, alpha per-channel (<c>ElementCount==C</c>) or a single shared value.</summary>
    unsafe void Prelu(Tensor output, Tensor input, Tensor alpha)
        => PreluReference(output, input, alpha);

    /// <summary>The managed <see cref="Prelu"/> body, callable directly. A backend override must
    /// call THIS to fall back — <c>((IBackend)this).Prelu(...)</c> re-enters the override and
    /// recurses until the stack overflows.</summary>
    static unsafe void PreluReference(Tensor output, Tensor input, Tensor alpha)
    {
        int batch = (int)input.Shape[0], channels = (int)input.Shape[1], timeDim = (int)input.Shape[2];
        float* op = (float*)output.DataPointer, ip = (float*)input.DataPointer, ap = (float*)alpha.DataPointer;
        bool perCh = alpha.ElementCount > 1;
        for (int bi = 0; bi < batch; bi++)
            for (int ci = 0; ci < channels; ci++)
            {
                float a = perCh ? ap[ci] : ap[0];
                long off = ((long)bi * channels + ci) * timeDim;
                for (int ti = 0; ti < timeDim; ti++) { float v = ip[off + ti]; op[off + ti] = v >= 0f ? v : a * v; }
            }
    }

    /// <summary>GEGLU activation: splits input in half along last dim, applies GELU gate. Output has half the elements of input.</summary>
    void GeGlu(Tensor output, Tensor input);
}

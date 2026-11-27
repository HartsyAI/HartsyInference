using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Activations, elementwise exp and reciprocal, the uniform fake quantize, the gated linear unit, the group gate and the fused
// linear prologue and epilogue.
public partial interface IBackend
{
    /// <summary>Elementwise <c>out = e^x</c>, computed in F32; input and output each F32 or BF16, of equal element count.
    /// Contract: <see cref="Contracts.ElementwiseMathContract"/>; reference: <see cref="ElementwiseMathReference"/>.</summary>
    void Exp(Tensor output, Tensor input) => ElementwiseMathReference.Exp(output, input);

    /// <summary>Elementwise <c>out = 1 / x</c> with IEEE division (<c>±0</c> gives <c>±∞</c>); dtypes as <see cref="Exp"/>.</summary>
    void Reciprocal(Tensor output, Tensor input) => ElementwiseMathReference.Reciprocal(output, input);

    /// <summary>Uniform fake quantization: every value clamped to <c>[low, high]</c>, truncated to one of <paramref name="levels"/>
    /// evenly spaced steps and mapped back into the range (DIAMOND's 256-level pixel quantization over <c>[−1, 1]</c>). F32.
    /// Contract: <see cref="Contracts.FakeQuantizeContract"/>; reference: <see cref="FakeQuantizeReference"/>.</summary>
    void FakeQuantizeUniform(Tensor output, Tensor input, float low, float high, int levels)
        => FakeQuantizeReference.Apply(output, input, low, high, levels);

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

    /// <summary>A gated linear unit over the two halves of the last axis: <c>out[r, i] = act(gate[r, i]) · value[r, i]</c>, the gate
    /// and value halves in the order the <see cref="GluSpec"/> says (SwiGLU and GeGLU over <c>[gate | up]</c>, GEGLU and SiTU-GLU
    /// over <c>[value | gate]</c>). Input and output are F32, F16 or BF16 alike; the output's last extent is half the input's.
    /// Contract in <c>docs/BACKEND_OPS.md</c>; <see cref="GluReference.Apply"/> is its executable form.</summary>
    void Glu(Tensor output, Tensor input, in GluSpec spec) => GluReference.Apply(output, input, spec);

    /// <summary><see cref="Glu"/> that may also emit a Q8_1 sidecar of its output, so the dp4a GEMV that follows skips its quantize
    /// (see <see cref="RmsNormEmitQ8"/>); semantically <see cref="Glu"/>.</summary>
    void GluEmitQ8(Tensor output, Tensor input, in GluSpec spec) => Glu(output, input, spec);

    /// <summary>Scales every group of a row by its own gate: <c>out[r, group] = f(gates[offset + r · stride + group]) · in[r, group]</c>,
    /// or adds it to the output, as the <see cref="GroupScaleSpec"/> says: token masking by a row vector, a per-head gate, one
    /// expert of a dense routing table accumulated into the combine. The output may be the input (in place). Contract in
    /// <c>docs/BACKEND_OPS.md</c>; <see cref="GroupScaleReference.Apply"/> is its executable form.</summary>
    void GroupScale(Tensor output, Tensor input, Tensor gates, in GroupScaleSpec spec) => GroupScaleReference.Apply(output, input, gates, spec);

    /// <summary>A projection with what the <see cref="LinearFusion"/> folds around it: a <see cref="GroupScale"/> of the input before
    /// the GEMM and an activation of the output after it. A backend whose GEMM already walks the input or the output applies them
    /// in that pass; the default runs the separate ops, scaling a copy of the input so the caller's tensor is untouched.</summary>
    void LinearFused(Tensor output, Tensor input, Tensor weight, Tensor? bias, in LinearFusion fusion)
        => ComposeLinearFused(this, output, input, weight, bias, fusion);

    /// <summary>The unfused form of <see cref="LinearFused"/>: the prologue's group scale, the GEMM, the epilogue's activation. Static so a
    /// backend whose fused path does not serve a call can still run it.</summary>
    static void ComposeLinearFused(IBackend b, Tensor output, Tensor input, Tensor weight, Tensor? bias, in LinearFusion fusion)
    {
        Tensor? gated = null;
        try
        {
            Tensor source = input;
            if (fusion.PrologueGates is not null)
            {
                gated = new Tensor(input.Shape, input.DType);
                b.GroupScale(gated, input, fusion.PrologueGates, fusion.Prologue);
                source = gated;
            }
            b.Linear(output, source, weight, bias);
            if (fusion.Epilogue == LinearActivation.GeluTanh) b.Gelu(output, output);
            else if (fusion.Epilogue != LinearActivation.None) throw new ArgumentOutOfRangeException(nameof(fusion), fusion.Epilogue, "Unknown linear activation.");
        }
        finally
        {
            gated?.Dispose();
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
}

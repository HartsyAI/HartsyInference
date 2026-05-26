using SharpInference.Core.Tensors;

namespace SharpInference.Cpu.Kernels;

/// <summary>Provides activation function CPU compute kernels for F32 tensors. Uses scalar loops since GELU and SiLU involve transcendental functions that do not benefit significantly from SIMD vectorization.</summary>
public static class ActivationKernels
{
    private const float Sqrt2OverPi = 0.7978845608028654f; // sqrt(2/pi)
    private const float GeluCoeff = 0.044715f;

    /// <summary>Applies the Gaussian Error Linear Unit (GELU) activation function. Exact formulation: output = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Gelu(Tensor output, Tensor input)
    {
        int count = (int)output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;

        for (int i = 0; i < count; i++)
        {
            float x = pIn[i];
            float x3 = x * x * x;
            float inner = Sqrt2OverPi * (x + GeluCoeff * x3);
            float tanhVal = MathF.Tanh(inner);
            pOut[i] = x * 0.5f * (1.0f + tanhVal);
        }
    }

    /// <summary>Applies the Sigmoid Linear Unit (SiLU) activation function, also known as Swish. Formulation: output = x * sigmoid(x) = x / (1 + exp(-x)).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Silu(Tensor output, Tensor input)
    {
        int count = (int)output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;

        for (int i = 0; i < count; i++)
        {
            float x = pIn[i];
            float sigmoid = 1.0f / (1.0f + MathF.Exp(-x));
            pOut[i] = x * sigmoid;
        }
    }

    /// <summary>Sigmoid: output = 1 / (1 + exp(-x)). Used by LSTM input/forget/output gates and by the binary EOS classifier in streaming TTS. Sign-aware formulation avoids overflow in exp.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Sigmoid(Tensor output, Tensor input)
    {
        int count = (int)output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;

        for (int i = 0; i < count; i++)
        {
            float x = pIn[i];
            if (x >= 0f)
            {
                float ex = MathF.Exp(-x);
                pOut[i] = 1.0f / (1.0f + ex);
            }
            else
            {
                float ex = MathF.Exp(x);
                pOut[i] = ex / (1.0f + ex);
            }
        }
    }

    /// <summary>Hyperbolic tangent: output = tanh(x). Used by LSTM cell-state updates / outputs and by other vocoders' modulation paths.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Tanh(Tensor output, Tensor input)
    {
        int count = (int)output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;

        for (int i = 0; i < count; i++) pOut[i] = MathF.Tanh(pIn[i]);
    }

    /// <summary>Leaky ReLU: <c>x if x &gt;= 0 else slope * x</c>. Used by Kokoro /
    /// StyleTTS 2's text encoder (slope=0.2) and several other audio nets where the
    /// non-zero negative gradient prevents dead-neuron failure modes in narrow channel
    /// counts.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void LeakyRelu(Tensor output, Tensor input, float slope)
    {
        int count = (int)output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (int i = 0; i < count; i++)
        {
            float x = pIn[i];
            pOut[i] = x >= 0f ? x : slope * x;
        }
    }

    /// <summary>ELU activation: <c>x if x &gt;= 0 else alpha * (exp(x) - 1)</c>. Standard SEANet activation for EnCodec / DAC residual blocks.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Elu(Tensor output, Tensor input, float alpha)
    {
        int count = (int)output.ElementCount;
        float* pOut = (float*)output.DataPointer;
        float* pIn = (float*)input.DataPointer;
        for (int i = 0; i < count; i++)
        {
            float x = pIn[i];
            pOut[i] = x >= 0f ? x : alpha * (MathF.Exp(x) - 1f);
        }
    }

    /// <summary>Snake activation, with optional snake-beta variant. Inputs and outputs
    /// are channels-first <c>[B, C, T]</c>; <paramref name="alpha"/> is shape <c>[C]</c>
    /// (the leading and trailing singleton dims are squeezed). When
    /// <paramref name="beta"/> is null the divisor is <paramref name="alpha"/>[c] (vanilla
    /// snake from BigVGAN / Stable Audio Oobleck VAE); when supplied the divisor is
    /// <paramref name="beta"/>[c] + 1e-8 (snake-beta from BigVGAN-v2).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta)
    {
        if (input.Shape.Rank != 3) throw new ArgumentException($"Snake expects rank-3 input, got {input.Shape}.");
        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int t = (int)input.Shape[2];
        if (alpha.ElementCount != channels)
            throw new ArgumentException($"Snake alpha must have {channels} elements, got {alpha.ElementCount}.");
        if (beta is not null && beta.ElementCount != channels)
            throw new ArgumentException($"Snake beta must have {channels} elements, got {beta.ElementCount}.");

        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        float* ap = (float*)alpha.DataPointer;
        float* bp = beta is null ? null : (float*)beta.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float aV = ap[c];
                float divisor = bp is null ? aV : bp[c] + 1e-8f;
                float invDiv = 1f / divisor;
                int row = (b * channels + c) * t;
                for (int j = 0; j < t; j++)
                {
                    float x = ip[row + j];
                    float s = MathF.Sin(aV * x);
                    op[row + j] = x + s * s * invDiv;
                }
            }
        }
    }
}

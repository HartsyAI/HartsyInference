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
}

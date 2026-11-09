using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Quantized GEMM, activation quantization, and the Hadamard rotation applied before it.
public partial interface IBackend
{
    /// <summary>Walsh-Hadamard transform of each row over the last dimension (a power of two):
    /// <c>output = scale * (input @ H)</c> with the Sylvester (natural order) <c>H</c>; see <see cref="Host.HadamardReference"/>.
    /// F32; in place is allowed. <c>scale = 1/sqrt(width)</c> makes it orthonormal.</summary>
    void Hadamard(Tensor output, Tensor input, float scale) =>
        throw NotSupportedPrimitive(nameof(Hadamard));

    /// <summary>Fused quantized matmul: <c>output = input @ quantWeight^T (+ bias)</c>, dequantized in-kernel (Q8_0/Q4_K/Q5_K/Q6_K).</summary>
    void QuantizedMatMul(Tensor output, Tensor input, Tensor quantWeight, Tensor? bias)
        => throw new NotSupportedException(
            "QuantizedMatMul is implemented on the CUDA backend only; dequantize the weight to F16/F32 for CPU/Vulkan.");

    /// <summary>Encodes F32 activations as the uint8 codes <see cref="LinearI8U8"/> reads, element by element, with
    /// <see cref="Numerics.Int8Tiles.QuantizeActivation"/>. <paramref name="output"/> is U8 and holds as many elements
    /// as <paramref name="input"/>.</summary>
    void QuantizeActivationsU8(Tensor output, Tensor input)
        => throw new NotSupportedException("QuantizeActivationsU8 is implemented on the CPU backend only.");

    /// <summary>int8-weight, uint8-activation product with a float epilogue, the int8 linear layer of Opus's DNN code
    /// and of RNNoise:
    /// <c>output[m, n] = scale[n]·Σₖ weight[n, k]·input[m, k] + bias[n] + diag[n]·diagInput[m, n mod K]</c>.</summary>
    /// <remarks><para><paramref name="input"/> is U8 <c>[M, K]</c>, codes from <see cref="QuantizeActivationsU8"/>.
    /// <paramref name="weight"/> is I8 <c>[N/8, K/4, 8, 4]</c> in <see cref="Numerics.Int8Tiles"/> order;
    /// <paramref name="scale"/>, <paramref name="bias"/> and <paramref name="diag"/> hold N floats; the output is F32
    /// <c>[M, N]</c>. K is at most 16384.</para>
    /// <para>The sum over k is exact in int32: no pair is saturated at int16, as AVX2's <c>maddubs</c> would. The
    /// epilogue rounds each step on its own and never fuses a multiply into an add: <c>(float)sum·scale</c>, then
    /// <c>+ bias</c>, then <c>+ diag·diagInput</c>.</para>
    /// <para><paramref name="diag"/> carries the diagonal a GRU's recurrent weight keeps in float while the rest is
    /// int8, one per gate: N must be a multiple of K, and K a multiple of 8. <paramref name="diagInput"/> is the F32
    /// input <c>[M, K]</c> the codes were made from. Pass both or neither.</para></remarks>
    void LinearI8U8(Tensor output, Tensor input, Tensor weight, Tensor scale, Tensor? bias, Tensor? diag = null,
        Tensor? diagInput = null)
        => throw new NotSupportedException("LinearI8U8 is implemented on the CPU backend only.");
}

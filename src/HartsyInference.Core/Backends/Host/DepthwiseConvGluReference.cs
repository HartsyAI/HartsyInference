using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>Host reference for <see cref="IBackend.DepthwiseConvGlu"/>: the (optionally SiLU'd) input read in F32, the depthwise
/// taps summed in row-major tap order, the bias added, then <c>value · SiLU(gate)</c>, rounded once to the output dtype. Frames
/// fan out.</summary>
public static class DepthwiseConvGluReference
{
    /// <summary>Validates the operands and returns the hidden width <c>H</c> (half the input channels).</summary>
    public static int Validate(Tensor output, Tensor input, Tensor taps, Tensor? bias, int frames, int height, int width, int kernel)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(taps);
        if (frames < 1 || height < 1 || width < 1 || kernel < 1 || kernel % 2 == 0)
            throw new ArgumentException($"DepthwiseConvGlu needs positive extents and an odd kernel, got {frames}x{height}x{width}, k={kernel}.");
        if (input.DType != DType.F32 && input.DType != DType.BF16)
            throw new ArgumentException($"DepthwiseConvGlu input must be F32 or BF16, got {input.DType.Name}.", nameof(input));
        if (output.DType != input.DType)
            throw new ArgumentException($"DepthwiseConvGlu output dtype {output.DType.Name} must match the input ({input.DType.Name}).", nameof(output));
        if (taps.DType != DType.F32)
            throw new ArgumentException($"DepthwiseConvGlu taps must be F32, got {taps.DType.Name}.", nameof(taps));
        if (bias is not null && bias.DType != DType.F32)
            throw new ArgumentException($"DepthwiseConvGlu bias must be F32, got {bias.DType.Name}.", nameof(bias));
        long pixels = (long)frames * height * width;
        if (input.ElementCount % pixels != 0 || (input.ElementCount / pixels) % 2 != 0)
            throw new ArgumentException($"DepthwiseConvGlu input {input.Shape} is not {pixels} pixels of an even channel count.", nameof(input));
        long channels = input.ElementCount / pixels;
        long hidden = channels / 2;
        if (output.ElementCount != pixels * hidden)
            throw new ArgumentException($"DepthwiseConvGlu output {output.Shape} must hold {pixels} pixels of {hidden} channels.", nameof(output));
        if (taps.ElementCount != channels * kernel * kernel)
            throw new ArgumentException($"DepthwiseConvGlu taps {taps.Shape} must be [{kernel}, {kernel}, {channels}].", nameof(taps));
        if (bias is not null && bias.ElementCount != channels)
            throw new ArgumentException($"DepthwiseConvGlu bias {bias.Shape} must hold {channels} values.", nameof(bias));
        return checked((int)hidden);
    }

    /// <summary>Host <see cref="IBackend.DepthwiseConvGlu"/>.</summary>
    public static unsafe void Run(Tensor output, Tensor input, Tensor taps, Tensor? bias, int frames, int height, int width, int kernel,
        bool inputSilu)
    {
        int hidden = Validate(output, input, taps, bias, frames, height, width, kernel);
        float[] weights = ToF32(taps);
        float[]? biases = bias is null ? null : ToF32(bias);
        bool bf16 = input.DType == DType.BF16;
        nint inPtr = (nint)input.DataPointer, outPtr = (nint)output.DataPointer;
        long work = (long)frames * height * width * hidden * kernel * kernel * 2;
        CpuParallel.For(frames, work, (inPtr, outPtr), (f, state) =>
        {
            int channels = 2 * hidden, pad = kernel / 2;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    long pixel = ((long)f * height + y) * width + x;
                    for (int c = 0; c < hidden; c++)
                    {
                        float value = biases?[c] ?? 0f;
                        float gate = biases?[hidden + c] ?? 0f;
                        for (int ky = 0; ky < kernel; ky++)
                        {
                            int sy = y + ky - pad;
                            if (sy < 0 || sy >= height) continue;
                            for (int kx = 0; kx < kernel; kx++)
                            {
                                int sx = x + kx - pad;
                                if (sx < 0 || sx >= width) continue;
                                long source = (((long)f * height + sy) * width + sx) * channels;
                                long tap = (long)(ky * kernel + kx) * channels;
                                float v = Load(state.inPtr, source + c, bf16), g = Load(state.inPtr, source + hidden + c, bf16);
                                if (inputSilu)
                                {
                                    v = Silu(v);
                                    g = Silu(g);
                                }

                                value += weights[tap + c] * v;
                                gate += weights[tap + hidden + c] * g;
                            }
                        }
                        float product = value * Silu(gate);
                        Store(state.outPtr, pixel * hidden + c, product, bf16);
                    }
                }
            }
        });
    }

    private static float Silu(float v) => v / (1f + MathF.Exp(-v));

    private static unsafe float Load(nint basePtr, long index, bool bf16) =>
        bf16 ? BFloat16Bits.ToFloat(((ushort*)basePtr)[index]) : ((float*)basePtr)[index];

    private static unsafe void Store(nint basePtr, long index, float value, bool bf16)
    {
        if (bf16) ((ushort*)basePtr)[index] = BFloat16Bits.FromFloat(value);
        else ((float*)basePtr)[index] = value;
    }

    private static unsafe float[] ToF32(Tensor tensor)
    {
        float[] values = new float[tensor.ElementCount];
        if (tensor.DType == DType.F32)
        {
            tensor.AsReadOnlySpan<float>().CopyTo(values);
            return values;
        }

        ReadOnlySpan<ushort> bits = tensor.AsReadOnlySpan<ushort>();
        for (int i = 0; i < values.Length; i++)
            values[i] = BFloat16Bits.ToFloat(bits[i]);
        return values;
    }
}

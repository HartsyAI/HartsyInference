using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Duo;

/// <summary>Exact precompute of the AdaLN modulation for a fixed list of times. A block's <c>adaln_proj</c> maps <c>silu(time_embedding(t))</c> (2688) to <c>3 modalities * 6 * hidden</c> values; with the time list fixed, the 13.0B parameters of the 50 <c>adaln_proj</c> layers
/// collapse to <c>times * 3 * 6 * hidden</c> numbers per block. A table is <c>[times * 3, 6 * hidden]</c>, row <c>time_index * 3 + modality_tag</c> (the same index the DiT's AdaLN rows use), columns in the chunk order shift, scale, gate of the attention then of the MLP.</summary>
public static class DuoAdalnTables
{
    /// <summary>Sinusoid (cos before sin, <c>exp(-ln(10000) * i / half)</c> frequencies), <c>proj_in</c>, SiLU, <c>proj_out</c>, in the time embedder's F32. Returns <c>[times, outDim]</c> row-major.</summary>
    public static float[] TimeEmbeddings(ReadOnlySpan<float> times, Tensor projInWeight, Tensor? projInBias, Tensor projOutWeight, Tensor? projOutBias)
    {
        ArgumentNullException.ThrowIfNull(projInWeight);
        ArgumentNullException.ThrowIfNull(projOutWeight);
        if (projInWeight.Shape.Rank != 2 || projOutWeight.Shape.Rank != 2 || projInWeight.Shape[0] != projOutWeight.Shape[1])
            throw new ArgumentException($"Time embedder weights {projInWeight.Shape} and {projOutWeight.Shape} do not chain.");
        int freqDim = (int)projInWeight.Shape[1], hidden = (int)projInWeight.Shape[0], outDim = (int)projOutWeight.Shape[0];
        if (freqDim % 2 != 0) throw new ArgumentException("The sinusoid width must be even.", nameof(projInWeight));

        float[] w1 = ToF32(projInWeight), w2 = ToF32(projOutWeight);
        float[]? b1 = projInBias is null ? null : ToF32(projInBias), b2 = projOutBias is null ? null : ToF32(projOutBias);
        int half = freqDim / 2;
        float[] result = new float[times.Length * outDim];
        double[] sinusoid = new double[freqDim], act = new double[hidden];
        for (int t = 0; t < times.Length; t++)
        {
            for (int i = 0; i < half; i++)
            {
                double arg = times[t] * Math.Exp(-Math.Log(10000.0) * i / half);
                sinusoid[i] = (float)Math.Cos(arg);
                sinusoid[half + i] = (float)Math.Sin(arg);
            }
            for (int h = 0; h < hidden; h++)
            {
                double sum = b1?[h] ?? 0f;
                for (int k = 0; k < freqDim; k++) sum += sinusoid[k] * w1[(long)h * freqDim + k];
                act[h] = (float)sum / (1f + MathF.Exp(-(float)sum));
            }
            for (int o = 0; o < outDim; o++)
            {
                double sum = b2?[o] ?? 0f;
                for (int h = 0; h < hidden; h++) sum += act[h] * w2[(long)o * hidden + h];
                result[(long)t * outDim + o] = (float)sum;
            }
        }
        return result;
    }

    /// <summary>One block's table: <c>silu(embedding) @ weight.T + bias</c> for every time, from the BF16 or F32 <c>adaln_proj.linear</c> weight <c>[N, embedDim]</c>. Returns an owned F32 <c>[times, N]</c>, or <paramref name="outputShape"/> over the same values
    /// (<c>[times * 3, 6 * hidden]</c> when <c>N = 18 * hidden</c>).</summary>
    public static unsafe Tensor Project(float[] embeddings, int embedDim, Tensor weight, Tensor? bias, DuoAdalnPrecision precision,
        TensorShape? outputShape = null)
    {
        ArgumentNullException.ThrowIfNull(embeddings);
        ArgumentNullException.ThrowIfNull(weight);
        if (embedDim <= 0 || embeddings.Length % embedDim != 0)
            throw new ArgumentException($"{embeddings.Length} embedding values are not whole rows of {embedDim}.", nameof(embeddings));
        if (weight.Shape.Rank != 2 || weight.Shape[1] != embedDim)
            throw new ArgumentException($"adaln_proj weight {weight.Shape} does not take {embedDim} inputs.", nameof(weight));
        if (weight.DType != DType.BF16 && weight.DType != DType.F32)
            throw new NotSupportedException($"adaln_proj weight dtype {weight.DType} is not supported.");
        if (bias is not null && (bias.Shape.Rank != 1 || bias.Shape[0] != weight.Shape[0]))
            throw new ArgumentException($"adaln_proj bias {bias.Shape} does not match weight {weight.Shape}.", nameof(bias));

        int times = embeddings.Length / embedDim;
        long outDim = weight.Shape[0];
        bool roundBf16 = precision == DuoAdalnPrecision.Bf16Reference;
        float[] activation = new float[embeddings.Length];
        for (int i = 0; i < activation.Length; i++)
        {
            float x = embeddings[i];
            float silu = x / (1f + MathF.Exp(-x));
            activation[i] = roundBf16 ? BF16ToF32(TensorCasts.F32ToBf16BitsRoundNearestEven(silu)) : silu;
        }

        float[]? biasValues = bias is null ? null : ToF32(bias);
        TensorShape shape = outputShape ?? new TensorShape(times, outDim);
        if (shape.ElementCount != times * outDim)
            throw new ArgumentException($"Output shape {shape} does not hold {times} x {outDim} values.", nameof(outputShape));
        Tensor result = new(shape, DType.F32);
        nint weightAddress = (nint)weight.DataPointer, resultAddress = (nint)result.DataPointer;
        bool weightBf16 = weight.DType == DType.BF16;
        Parallel.For(0, outDim, () => new float[embedDim], (n, _, row) =>
        {
            if (weightBf16)
            {
                ushort* src = (ushort*)weightAddress + n * embedDim;
                for (int k = 0; k < embedDim; k++) row[k] = BF16ToF32(src[k]);
            }
            else
            {
                new ReadOnlySpan<float>((float*)weightAddress + n * embedDim, embedDim).CopyTo(row);
            }
            float* output = (float*)resultAddress;
            for (int t = 0; t < times; t++)
            {
                double sum = biasValues?[n] ?? 0f;
                int offset = t * embedDim;
                for (int k = 0; k < embedDim; k++) sum += (double)activation[offset + k] * row[k];
                float value = (float)sum;
                output[t * outDim + n] = roundBf16 ? BF16ToF32(TensorCasts.F32ToBf16BitsRoundNearestEven(value)) : value;
            }
            return row;
        }, _ => { });
        return result;
    }

    private static float BF16ToF32(ushort bits) => BitConverter.UInt32BitsToSingle((uint)bits << 16);

    private static unsafe float[] ToF32(Tensor tensor)
    {
        float[] values = new float[tensor.ElementCount];
        if (tensor.DType == DType.F32)
        {
            tensor.AsReadOnlySpan<float>().CopyTo(values);
        }
        else if (tensor.DType == DType.BF16)
        {
            ReadOnlySpan<ushort> bits = tensor.AsReadOnlySpan<ushort>();
            for (int i = 0; i < values.Length; i++) values[i] = BF16ToF32(bits[i]);
        }
        else
        {
            throw new NotSupportedException($"Time embedder dtype {tensor.DType} is not supported.");
        }
        return values;
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Part;

/// <summary>A P3-SAM MLP: <c>Linear, GELU, Linear, GELU, Linear</c> (state-dict keys <c>{prefix}.0</c>, <c>.2</c>, <c>.4</c>), held as host arrays, with the matrix
/// products going through the backend.</summary>
internal sealed class P3SamMlp : IDisposable
{
    public readonly float[] W0, B0;
    public readonly Tensor T0, TB0, T1, TB1, T2, TB2;
    public readonly int In, Hidden, Out;

    public P3SamMlp(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        Tensor w0 = Get(weights, prefix, 0, "weight"), w1 = Get(weights, prefix, 2, "weight"), w2 = Get(weights, prefix, 4, "weight");
        Hidden = (int)w0.Shape[0];
        In = (int)w0.Shape[1];
        Out = (int)w2.Shape[0];
        if (w1.Shape[0] != Hidden || w1.Shape[1] != Hidden || w2.Shape[1] != Hidden)
            throw new HartsyInferenceException($"P3-SAM MLP '{prefix}' has inconsistent layer shapes.");
        // The first layer stays as host arrays too (the heads split its columns); every layer lives as a tensor made once, so a forward pass does not copy
        // the weights into a fresh tensor on every call.
        W0 = ToArray(w0);
        B0 = ToArray(Get(weights, prefix, 0, "bias"));
        (T0, TB0) = (Own(W0, Hidden, In), Own(B0, Hidden));
        (T1, TB1) = (Own(ToArray(w1), Hidden, Hidden), Own(ToArray(Get(weights, prefix, 2, "bias")), Hidden));
        (T2, TB2) = (Own(ToArray(w2), Out, Hidden), Own(ToArray(Get(weights, prefix, 4, "bias")), Out));
    }

    /// <summary>Releases the layer tensors.</summary>
    public void Dispose()
    {
        foreach (Tensor tensor in new[] { T0, TB0, T1, TB1, T2, TB2 })
            tensor.Dispose();
    }

    // A weight tensor made once. ForceF32Gemm keeps its matrix products in F32 on a GPU backend (the default path casts both operands to F16).
    internal static Tensor Own(float[] values, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32) { ForceF32Gemm = true };
        Copy(values, tensor);
        return tensor;
    }

    /// <summary>The per-column maximum over the rows of <paramref name="source"/> <c>[rows, width]</c> (reads it back to the host), split over cores.</summary>
    internal static unsafe float[] MaxRows(Tensor source, int rows, int width)
    {
        float* data = (float*)source.DataPointer;
        int chunks = Math.Max(1, Math.Min(Environment.ProcessorCount, rows / 1024));
        float[][] partial = new float[chunks][];
        Parallel.For(0, chunks, chunk =>
        {
            float[] max = new float[width];
            Array.Fill(max, float.NegativeInfinity);
            int from = (int)((long)rows * chunk / chunks), to = (int)((long)rows * (chunk + 1) / chunks);
            for (int i = from; i < to; i++)
            {
                float* row = data + (long)i * width;
                for (int c = 0; c < width; c++)
                    if (row[c] > max[c])
                        max[c] = row[c];
            }

            partial[chunk] = max;
        });
        float[] result = partial[0];
        for (int k = 1; k < chunks; k++)
            for (int c = 0; c < width; c++)
                result[c] = Math.Max(result[c], partial[k][c]);
        return result;
    }

    /// <summary>All rows through the whole MLP.</summary>
    public float[] Forward(IBackend backend, float[] rows, int count)
    {
        float[] h = Matmul(backend, rows, count, In, T0, Hidden, TB0);
        h = Gelu(backend, h);
        h = Matmul(backend, h, count, Hidden, T1, Hidden, TB1);
        h = Gelu(backend, h);
        return Matmul(backend, h, count, Hidden, T2, Out, TB2);
    }

    /// <summary>One row through the whole MLP.</summary>
    public float[] ForwardRow(IBackend backend, float[] row) => Forward(backend, row, 1);

    // nn.GELU (exact, erf based) through the backend's own kernel.
    internal static float[] Gelu(IBackend backend, float[] x)
    {
        using Tensor input = new(new TensorShape(x.Length), DType.F32);
        using Tensor output = new(new TensorShape(x.Length), DType.F32);
        Copy(x, input);
        backend.GeluErf(output, input);
        float[] y = new float[x.Length];
        Read(output, y);
        return y;
    }

    internal static float[] Matmul(IBackend backend, float[] input, int rows, int inDim, Tensor weight, int outDim, Tensor? bias)
    {
        using Tensor x = new(new TensorShape(rows, inDim), DType.F32);
        using Tensor y = new(new TensorShape(rows, outDim), DType.F32);
        Copy(input, x);
        backend.Linear(y, x, weight, bias);
        float[] result = new float[(long)rows * outDim];
        Read(y, result);
        return result;
    }

    internal static unsafe void Copy(float[] source, Tensor destination) =>
        new ReadOnlySpan<float>(source).CopyTo(new Span<float>((float*)destination.DataPointer, source.Length));

    /// <summary>Writes rows of <paramref name="left"/> (<paramref name="leftWidth"/> values each) followed by rows of <paramref name="right"/> into <paramref name="destination"/>.</summary>
    internal static unsafe void CopyRows(float[] left, int leftWidth, float[] right, int rightWidth, int rows, Tensor destination)
    {
        float* target = (float*)destination.DataPointer;
        int width = leftWidth + rightWidth;
        for (int i = 0; i < rows; i++)
        {
            new ReadOnlySpan<float>(left, i * leftWidth, leftWidth).CopyTo(new Span<float>(target + (long)i * width, leftWidth));
            new ReadOnlySpan<float>(right, i * rightWidth, rightWidth).CopyTo(new Span<float>(target + (long)i * width + leftWidth, rightWidth));
        }
    }

    internal static unsafe void Fill(Tensor destination, float value) =>
        new Span<float>((float*)destination.DataPointer, (int)destination.ElementCount).Fill(value);

    internal static unsafe void Read(Tensor source, float[] destination) =>
        new ReadOnlySpan<float>((float*)source.DataPointer, destination.Length).CopyTo(destination);


    private static Tensor Get(IReadOnlyDictionary<string, Tensor> weights, string prefix, int layer, string kind)
    {
        string key = $"{prefix}.{layer}.{kind}";
        if (!weights.TryGetValue(key, out Tensor? tensor))
            throw new HartsyInferenceException($"P3-SAM weight '{key}' is missing.");
        if (tensor.DType != DType.F32)
            throw new HartsyInferenceException($"P3-SAM weight '{key}' must be F32, got {tensor.DType.Name}.");
        return tensor;
    }

    private static unsafe float[] ToArray(Tensor t)
    {
        float[] values = new float[t.ElementCount];
        new ReadOnlySpan<float>((float*)t.DataPointer, values.Length).CopyTo(values);
        return values;
    }
}

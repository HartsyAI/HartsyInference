using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Part;

/// <summary>A P3-SAM MLP: <c>Linear, GELU, Linear, GELU, Linear</c> (state-dict keys <c>{prefix}.0</c>, <c>.2</c>, <c>.4</c>), held as host arrays, with the matrix
/// products going through the backend.</summary>
internal sealed class P3SamMlp
{
    public readonly float[] W0, B0, W1, B1, W2, B2;
    public readonly int In, Hidden, Out;

    public P3SamMlp(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        Tensor w0 = Get(weights, prefix, 0, "weight"), w1 = Get(weights, prefix, 2, "weight"), w2 = Get(weights, prefix, 4, "weight");
        Hidden = (int)w0.Shape[0];
        In = (int)w0.Shape[1];
        Out = (int)w2.Shape[0];
        if (w1.Shape[0] != Hidden || w1.Shape[1] != Hidden || w2.Shape[1] != Hidden)
            throw new HartsyInferenceException($"P3-SAM MLP '{prefix}' has inconsistent layer shapes.");
        W0 = ToArray(w0);
        B0 = ToArray(Get(weights, prefix, 0, "bias"));
        W1 = ToArray(w1);
        B1 = ToArray(Get(weights, prefix, 2, "bias"));
        W2 = ToArray(w2);
        B2 = ToArray(Get(weights, prefix, 4, "bias"));
    }

    /// <summary>All rows through the whole MLP.</summary>
    public float[] Forward(IBackend backend, float[] rows, int count)
    {
        float[] h = Matmul(backend, rows, count, In, W0, Hidden, B0);
        h = Gelu(backend, h);
        h = Matmul(backend, h, count, Hidden, W1, Hidden, B1);
        h = Gelu(backend, h);
        return Matmul(backend, h, count, Hidden, W2, Out, B2);
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

    internal static float[] Matmul(IBackend backend, float[] input, int rows, int inDim, float[] weight, int outDim, float[]? bias)
    {
        using Tensor x = new(new TensorShape(rows, inDim), DType.F32);
        using Tensor w = new(new TensorShape(outDim, inDim), DType.F32);
        using Tensor y = new(new TensorShape(rows, outDim), DType.F32);
        Tensor? b = bias is null ? null : new Tensor(new TensorShape(outDim), DType.F32);
        try
        {
            Copy(input, x);
            Copy(weight, w);
            if (b is not null)
                Copy(bias!, b);
            backend.Linear(y, x, w, b);
            float[] result = new float[(long)rows * outDim];
            Read(y, result);
            return result;
        }
        finally
        {
            b?.Dispose();
        }
    }

    internal static unsafe void Copy(float[] source, Tensor destination) =>
        new ReadOnlySpan<float>(source).CopyTo(new Span<float>((float*)destination.DataPointer, source.Length));

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

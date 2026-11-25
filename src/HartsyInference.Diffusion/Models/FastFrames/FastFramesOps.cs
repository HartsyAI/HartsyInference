using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.FastFrames;

/// <summary>The small F32 building blocks the FastFrames refinement network and pixel decoder share: row-major <c>[S, C]</c>
/// token tensors, weights in the engine's <c>[out, in]</c> layout, every result a fresh tensor the caller disposes.</summary>
internal static class FastFramesOps
{
    /// <summary><c>x @ w^T + bias</c> for <c>x [S, in]</c> and <c>w [out, in]</c>.</summary>
    public static Tensor Linear(IBackend backend, Tensor x, Tensor weight, Tensor? bias)
    {
        Tensor output = new(new TensorShape(x.Shape[0], weight.Shape[0]), DType.F32);
        backend.Linear(output, x, weight, bias);
        return output;
    }

    /// <summary>Layer norm over the last axis without a learned affine.</summary>
    public static Tensor LayerNormPlain(IBackend backend, Tensor x, float eps)
    {
        Tensor output = new(x.Shape, DType.F32);
        backend.LayerNormNoAffine(output, x, eps);
        return output;
    }

    /// <summary>Layer norm over the last axis with a learned scale and shift.</summary>
    public static Tensor LayerNormAffine(IBackend backend, Tensor x, Tensor weight, Tensor bias, float eps)
    {
        Tensor output = new(x.Shape, DType.F32);
        backend.LayerNorm(output, x, weight, bias, eps);
        return output;
    }

    /// <summary><c>norm(x) · (1 + scale) + shift</c> with a plain layer norm; <paramref name="scalePlusOne"/> already holds
    /// <c>1 + scale</c>.</summary>
    public static Tensor Modulate(IBackend backend, Tensor x, Tensor shift, Tensor scalePlusOne, float eps)
    {
        using Tensor normed = LayerNormPlain(backend, x, eps);
        Tensor output = new(x.Shape, DType.F32);
        backend.AffineBroadcastLastDim(output, normed, scalePlusOne, shift);
        return output;
    }

    /// <summary><c>x + gate · value</c> with <paramref name="gate"/> <c>[C]</c> broadcast over the rows.</summary>
    public static Tensor GatedAdd(IBackend backend, Tensor x, Tensor value, Tensor gate)
    {
        Tensor output = new(x.Shape, DType.F32);
        backend.GatedResidualLastDim(output, x, value, gate);
        return output;
    }

    /// <summary><c>a + b</c>.</summary>
    public static Tensor Add(IBackend backend, Tensor a, Tensor b)
    {
        Tensor output = new(a.Shape, DType.F32);
        backend.Add(output, a, b);
        return output;
    }

    /// <summary>GELU with the tanh approximation (<c>nn.GELU(approximate="tanh")</c>).</summary>
    public static Tensor GeluTanh(IBackend backend, Tensor x)
    {
        Tensor output = new(x.Shape, DType.F32);
        backend.Gelu(output, x);
        return output;
    }

    /// <summary>Exact GELU (<c>nn.GELU()</c>).</summary>
    public static Tensor GeluErf(IBackend backend, Tensor x)
    {
        Tensor output = new(x.Shape, DType.F32);
        backend.GeluErf(output, x);
        return output;
    }

    /// <summary>SiLU.</summary>
    public static Tensor Silu(IBackend backend, Tensor x)
    {
        Tensor output = new(x.Shape, DType.F32);
        backend.Silu(output, x);
        return output;
    }

    /// <summary>A vector <c>[count]</c> from a host array.</summary>
    public static Tensor Vector(ReadOnlySpan<float> values)
    {
        Tensor vector = new(new TensorShape(values.Length), DType.F32);
        values.CopyTo(vector.AsSpan<float>());
        return vector;
    }

    /// <summary><c>1 + values</c> as a vector.</summary>
    public static Tensor OnePlus(ReadOnlySpan<float> values)
    {
        Tensor vector = new(new TensorShape(values.Length), DType.F32);
        Span<float> span = vector.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            span[i] = 1f + values[i];
        return vector;
    }

    /// <summary>The sinusoidal timestep embedding of <c>fastframes/models/kv_refinement.py</c> and Wan: <c>[cos, sin]</c> of
    /// <c>t · 10000^(-i/half)</c>, computed in double and stored as float32, as a <c>[1, dim]</c> row.</summary>
    public static Tensor SinusoidalEmbedding(float timestep, int dim)
    {
        // The shared helper in double precision: the same frequencies and phases this file used to compute inline.
        Tensor row = new(new TensorShape(1, dim), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(row, timestep, 1, dim, 10000f, doublePrecision: true);
        return row;
    }

    /// <summary>A fresh F32 tensor of <paramref name="shape"/> holding <paramref name="source"/>'s values, cast when the source is not
    /// F32. It never aliases the source: the caller owns the result, and may dispose the source straight away.</summary>
    /// <exception cref="ArgumentException">The element counts differ.</exception>
    public static Tensor CopyF32(Tensor source, TensorShape shape)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.ElementCount != shape.ElementCount)
            throw new ArgumentException($"A weight of {source.Shape} cannot be read as {shape}: the element counts differ.", nameof(shape));
        Tensor target = new(shape, DType.F32);
        if (source.DType == DType.F32)
        {
            source.AsReadOnlySpan<float>().CopyTo(target.AsSpan<float>());
            return target;
        }
        using Tensor cast = source.CastTo(DType.F32);
        cast.AsReadOnlySpan<float>().CopyTo(target.AsSpan<float>());
        return target;
    }

    /// <summary>The transpose of layer <paramref name="layer"/> of a stacked <c>[layers, in, out]</c> weight as an F32
    /// <c>[out, in]</c> tensor (the engine's Linear layout), converting from BF16 or F16 on the way.</summary>
    public static Tensor TransposedLayer(Tensor stacked, int layer)
    {
        int rows = (int)stacked.Shape[1], cols = (int)stacked.Shape[2];
        using Tensor view = stacked.SliceRows(layer, 1);
        using Tensor f32 = view.DType == DType.F32 ? view.Reshape(new TensorShape(rows, cols)) : view.CastTo(DType.F32);
        ReadOnlySpan<float> source = f32.AsReadOnlySpan<float>();
        Tensor output = new(new TensorShape(cols, rows), DType.F32);
        Span<float> target = output.AsSpan<float>();
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                target[j * rows + i] = source[i * cols + j];
        return output;
    }

    /// <summary>Row <paramref name="layer"/> of a stacked <c>[layers, 1, width]</c> bias or scale as an F32 <c>[width]</c> vector.</summary>
    public static Tensor LayerVector(Tensor stacked, int layer)
    {
        int width = (int)stacked.Shape[2];
        using Tensor view = stacked.SliceRows(layer, 1);
        using Tensor f32 = view.DType == DType.F32 ? view.Reshape(new TensorShape(width)) : view.CastTo(DType.F32);
        return Vector(f32.AsReadOnlySpan<float>());
    }
}

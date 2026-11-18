using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Idol.Budget;

/// <summary>Host-side helpers the Budget3D adapters share: tensors from arrays and back, and the F32 linear layer through the backend. The adapters are small next to Idol's trunk (a few thousand rows of width 128 to 1024),
/// so their matrix products go to the backend and the elementwise and tiny-sequence parts stay on the host.</summary>
internal static class BudgetHost
{
    /// <summary>A tensor of <paramref name="shape"/> holding <paramref name="data"/>.</summary>
    internal static Tensor Make(ReadOnlySpan<float> data, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        if (t.Shape.ElementCount != data.Length)
        {
            t.Dispose();
            throw new ArgumentException($"{data.Length} values do not fill a {new TensorShape(shape)} tensor.", nameof(data));
        }

        data.CopyTo(t.AsSpan<float>());
        return t;
    }

    /// <summary><c>x · Wᵀ + b</c> for <paramref name="input"/> <c>[1, rows, in]</c>, as <c>[1, rows, out]</c>; the caller owns it.</summary>
    internal static Tensor Linear(IBackend backend, Tensor input, Tensor weight, Tensor? bias)
    {
        Tensor output = new(new TensorShape(1, input.Shape[input.Shape.Rank - 2], weight.Shape[0]), DType.F32);
        try
        {
            backend.Linear(output, input, weight, bias);
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    /// <summary>Applies an elementwise backend activation to a copy of <paramref name="input"/>.</summary>
    internal static Tensor Activate(IBackend backend, Tensor input, Action<IBackend, Tensor, Tensor> activation)
    {
        Tensor output = new(input.Shape, DType.F32);
        try
        {
            activation(backend, output, input);
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }
}

/// <summary>Budget3D's scale mixer (upstream <c>ScaleMixer3D</c>): <c>n</c> child token features become one parent token feature, the plain mean plus a learned correction
/// <c>up(gelu(einsum(gelu(down(x + pos)), spatial_w) + spatial_b))</c> (exact GELU). <c>n = 2</c> pools two Sobol points into a 16³ cell and <c>n = 8</c> pools eight cells into an 8³ cube.</summary>
internal sealed class BudgetScaleMixer
{
    private readonly BudgetAdapterWeights _w;
    private readonly string _prefix;
    private readonly int _n;

    internal BudgetScaleMixer(BudgetAdapterWeights weights, int children)
    {
        _w = weights;
        _n = children;
        _prefix = $"bp_mixers.{(children == 2 ? 2 : 4)}";
    }

    /// <summary>The tensors a forward pass hands to the backend.</summary>
    internal IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _w[$"{_prefix}.down.weight"];
        yield return _w[$"{_prefix}.down.bias"];
        yield return _w[$"{_prefix}.up.weight"];
        yield return _w[$"{_prefix}.up.bias"];
    }

    /// <summary>The parents of <paramref name="children"/> (<c>parents × n</c> rows of width <c>d</c>, the n children of a parent adjacent): <c>[parents × d]</c> floats.</summary>
    internal float[] Apply(IBackend backend, ReadOnlySpan<float> children, int parents, int d)
    {
        int rows = parents * _n, bott = BudgetAdapterWeights.MixerBottleneck;
        ReadOnlySpan<float> pos = _w[$"{_prefix}.pos"].AsReadOnlySpan<float>();
        float[] shifted = new float[rows * d];
        for (int r = 0; r < rows; r++)
        {
            int slot = r % _n;
            for (int c = 0; c < d; c++)
                shifted[r * d + c] = children[r * d + c] + pos[slot * d + c];
        }

        float[] activated;
        using (Tensor input = BudgetHost.Make(shifted, 1, rows, d))
        using (Tensor down = BudgetHost.Linear(backend, input, _w[$"{_prefix}.down.weight"], _w[$"{_prefix}.down.bias"]))
        using (Tensor gelu = BudgetHost.Activate(backend, down, static (b, o, i) => b.GeluErf(o, i)))
            activated = gelu.AsReadOnlySpan<float>().ToArray();

        // einsum("...sc,cts->...tc", h, spatial_w) + spatial_b: for every parent and channel a small [n × n] product over the children.
        ReadOnlySpan<float> w = _w[$"{_prefix}.spatial_w"].AsReadOnlySpan<float>(), b = _w[$"{_prefix}.spatial_b"].AsReadOnlySpan<float>();
        float[] mixed = new float[parents * _n * bott];
        for (int p = 0; p < parents; p++)
            for (int t = 0; t < _n; t++)
                for (int c = 0; c < bott; c++)
                {
                    float acc = 0f;
                    for (int s = 0; s < _n; s++)
                        acc += activated[(p * _n + s) * bott + c] * w[(c * _n + t) * _n + s];
                    mixed[(p * _n + t) * bott + c] = acc + b[t * bott + c];
                }

        float[] correction;
        using (Tensor mixedTensor = BudgetHost.Make(mixed, 1, parents * _n, bott))
        using (Tensor gelu = BudgetHost.Activate(backend, mixedTensor, static (be, o, i) => be.GeluErf(o, i)))
        using (Tensor flat = gelu.Reshape(new TensorShape(1, parents, (long)_n * bott)))
        using (Tensor up = BudgetHost.Linear(backend, flat, _w[$"{_prefix}.up.weight"], _w[$"{_prefix}.up.bias"]))
            correction = up.AsReadOnlySpan<float>().ToArray();

        float[] result = new float[parents * d];
        for (int p = 0; p < parents; p++)
            for (int c = 0; c < d; c++)
            {
                float mean = 0f;
                for (int s = 0; s < _n; s++)
                    mean += children[(p * _n + s) * d + c];
                result[p * d + c] = mean / _n + correction[p * d + c];
            }

        return result;
    }
}

/// <summary>Budget3D's integrated positional encoding of the cube a merged token covers (upstream <c>MipNeRFIPE3D</c>): the sinusoids of the cube's centre at 8 octaves, damped by the cube's variance, then a small MLP.
/// The second pair of feature groups uses the excess variance over the s = 2 cell, so an s = 2 token contributes nothing there.</summary>
internal sealed class BudgetIpe
{
    private readonly BudgetAdapterWeights _w;

    internal BudgetIpe(BudgetAdapterWeights weights) => _w = weights;

    /// <summary>The tensors a forward pass hands to the backend.</summary>
    internal IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _w["bp_ipe.mlp.0.weight"];
        yield return _w["bp_ipe.mlp.0.bias"];
        yield return _w["bp_ipe.mlp.2.weight"];
        yield return _w["bp_ipe.mlp.2.bias"];
    }

    /// <summary>The encoding of <paramref name="count"/> cubes (<paramref name="centres"/>, <c>count × 3</c> in [0, 1]³) of scale <paramref name="scale"/> (2 or 4), <c>[count × d]</c> floats.</summary>
    internal float[] Encode(IBackend backend, ReadOnlySpan<float> centres, int count, int scale, int d)
    {
        int octaves = BudgetAdapterWeights.IpeOctaves;
        double varS = (2 * Math.PI * scale / 32) * (2 * Math.PI * scale / 32) / 12;
        double baseVar = (2 * Math.PI / 32 * 2) * (2 * Math.PI / 32 * 2) / 12;
        float[] damp = new float[octaves], excess = new float[octaves], omega = new float[octaves];
        for (int k = 0; k < octaves; k++)
        {
            omega[k] = MathF.Pow(2f, k);
            float w2 = -0.5f * omega[k] * omega[k];
            damp[k] = MathF.Exp(w2 * (float)varS);
            excess[k] = (float)(Math.Exp(w2 * (float)Math.Max(varS - baseVar, 0.0)) - 1.0);   // expm1; exactly 0 where the variance equals the cell's
        }

        float[] features = new float[count * 12 * octaves];
        float pi = (float)Math.PI;
        for (int i = 0; i < count; i++)
            for (int axis = 0; axis < 3; axis++)
            {
                float mu = (2f * centres[i * 3 + axis] - 1f) * pi;
                for (int k = 0; k < octaves; k++)
                {
                    float angle = mu * omega[k], sin = MathF.Sin(angle), cos = MathF.Cos(angle);
                    int o = i * 12 * octaves + axis * 4 * octaves;
                    features[o + k] = sin * damp[k];
                    features[o + octaves + k] = cos * damp[k];
                    features[o + 2 * octaves + k] = sin * excess[k];
                    features[o + 3 * octaves + k] = cos * excess[k];
                }
            }

        using Tensor input = BudgetHost.Make(features, 1, count, 12 * octaves);
        using Tensor hidden = BudgetHost.Linear(backend, input, _w["bp_ipe.mlp.0.weight"], _w["bp_ipe.mlp.0.bias"]);
        using Tensor act = BudgetHost.Activate(backend, hidden, static (b, o, i) => b.Silu(o, i));
        using Tensor output = BudgetHost.Linear(backend, act, _w["bp_ipe.mlp.2.weight"], _w["bp_ipe.mlp.2.bias"]);
        return output.AsReadOnlySpan<float>().ToArray();
    }
}

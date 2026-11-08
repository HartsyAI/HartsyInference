using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Linear stack with ReLU between layers and none after the last.</summary>
public sealed class Sam3Mlp
{
    private Tensor[] _weights = [];
    private Tensor[] _biases = [];

    /// <summary>Input width of the first layer.</summary>
    public int InputDim { get; private set; }

    /// <summary>Output width of the last layer.</summary>
    public int OutputDim { get; private set; }

    /// <summary>Loads <c>{prefix}layers.{i}.weight/bias</c>; <paramref name="layerDims"/> lists input, hidden widths and output.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix, params int[] layerDims)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(prefix);
        ArgumentNullException.ThrowIfNull(layerDims);
        if (layerDims.Length < 2)
            throw new ArgumentException("An MLP needs at least one layer.", nameof(layerDims));

        int layers = layerDims.Length - 1;
        Tensor[] w = new Tensor[layers];
        Tensor[] b = new Tensor[layers];
        for (int i = 0; i < layers; i++)
        {
            w[i] = Sam3WeightAccess.Require(weights, $"{prefix}layers.{i}.weight", layerDims[i + 1], layerDims[i]);
            b[i] = Sam3WeightAccess.Require(weights, $"{prefix}layers.{i}.bias", layerDims[i + 1]);
        }
        Bind(w, b);
    }

    /// <summary>Binds explicit layer weights <c>[out, in]</c> and biases <c>[out]</c>, in order, checking that widths chain.</summary>
    public void Bind(IReadOnlyList<Tensor> weights, IReadOnlyList<Tensor> biases)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(biases);
        if (weights.Count == 0 || weights.Count != biases.Count)
            throw new ArgumentException("An MLP needs one bias per weight and at least one layer.");

        for (int i = 0; i < weights.Count; i++)
        {
            Tensor w = weights[i];
            Tensor b = biases[i];
            if (w.Shape.Rank != 2 || b.Shape.Rank != 1 || b.Shape[0] != w.Shape[0])
                throw new ArgumentException($"MLP layer {i} must be a [out, in] weight with a [out] bias; found {w.Shape} and {b.Shape}.");
            if (i > 0 && w.Shape[1] != weights[i - 1].Shape[0])
                throw new ArgumentException($"MLP layer {i} input width {w.Shape[1]} does not match layer {i - 1} output {weights[i - 1].Shape[0]}.");
        }

        _weights = [.. weights];
        _biases = [.. biases];
        InputDim = (int)_weights[0].Shape[1];
        OutputDim = (int)_weights[^1].Shape[0];
    }

    /// <summary>Applies the stack to <paramref name="input"/> <c>[1, rows, InputDim]</c>; returns a new
    /// <c>[1, rows, OutputDim]</c> tensor the caller owns. The input is not disposed.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(input);
        if (_weights.Length == 0)
            throw new InvalidOperationException("SAM 3 MLP weights are not loaded.");
        Sam3Ops.RequireSequence(input, "MLP input", InputDim);

        Tensor current = input;
        bool owned = false;
        for (int i = 0; i < _weights.Length; i++)
        {
            Tensor next = Sam3Ops.Linear(backend, current, _weights[i], _biases[i]);
            if (owned)
                current.Dispose();
            if (i < _weights.Length - 1)
                Sam3Ops.Relu(backend, next);
            current = next;
            owned = true;
        }
        return current;
    }
}

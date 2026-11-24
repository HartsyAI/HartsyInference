using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Glm5Next.SciPerception;

/// <summary>Sage-SciPerception's <c>AtomProjector</c> (<c>modeling_projector.py</c>, SciPerception revision 7c228fc0): prepared per-row scientific features
/// become rows of the language model's embedding width. <c>LayerNorm([feature | position | discrete embeddings]) → Linear → GELU → Linear</c>, plus an
/// optional side-measurement MLP and a frozen per-row classifier whose probability a learned bridge maps into the same space. Host maths in F32.</summary>
public sealed class SciProjector
{
    private const float LayerNormEps = 1e-5f;
    private const int ReadoutHidden = 256, BridgeHidden = 128;

    private readonly IBackend _backend;
    private readonly SciProjectorSpec _s;
    private readonly float[] _atomIndex, _normW, _normB, _w1, _b1, _w2, _b2;
    private readonly float[][] _discrete;
    private readonly float[]? _sideW1, _sideW2;
    private readonly float[]? _mean, _std, _clsNormW, _clsNormB, _clsW1, _clsB1, _clsW2, _clsB2, _brW1, _brB1, _brW2, _brB2;

    /// <summary>Builds the projector from <paramref name="weights"/> (the projector's safetensors, keys as exported: <c>atom_index.weight</c>,
    /// <c>layers.{0,1,3}.*</c>, <c>discrete.N.weight</c>, <c>side.{0,2}.weight</c>, <c>readout.*</c>). Unused keys (<c>task_head.*</c>) are ignored.</summary>
    /// <exception cref="InvalidDataException">A needed tensor is missing or its shape does not match <paramref name="spec"/>.</exception>
    public SciProjector(IBackend backend, SciProjectorSpec spec, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(weights);
        _backend = backend;
        _s = spec;
        int width = spec.InputWidth;
        _atomIndex = Take(weights, "atom_index.weight", spec.MaxAtoms, spec.IndexDim);
        _discrete = new float[spec.DiscreteSizes.Count][];
        for (int i = 0; i < _discrete.Length; i++) _discrete[i] = Take(weights, $"discrete.{i}.weight", spec.DiscreteSizes[i], spec.DiscreteDim);
        _normW = Take(weights, "layers.0.weight", width);
        _normB = Take(weights, "layers.0.bias", width);
        _w1 = Take(weights, "layers.1.weight", spec.HiddenDim, width);
        _b1 = Take(weights, "layers.1.bias", spec.HiddenDim);
        _w2 = Take(weights, "layers.3.weight", spec.OutputDim, spec.HiddenDim);
        _b2 = Take(weights, "layers.3.bias", spec.OutputDim);
        if (spec.SideDim > 0)
        {
            _sideW1 = Take(weights, "side.0.weight", spec.SideHiddenDim, spec.SideDim);
            _sideW2 = Take(weights, "side.2.weight", spec.OutputDim, spec.SideHiddenDim);
        }
        if (spec.Readout)
        {
            int f = spec.FeatureDim;
            _mean = Take(weights, "readout.feature_mean", f);
            _std = Take(weights, "readout.feature_std", f);
            _clsNormW = Take(weights, "readout.classifier.0.weight", f);
            _clsNormB = Take(weights, "readout.classifier.0.bias", f);
            _clsW1 = Take(weights, "readout.classifier.1.weight", ReadoutHidden, f);
            _clsB1 = Take(weights, "readout.classifier.1.bias", ReadoutHidden);
            _clsW2 = Take(weights, "readout.classifier.3.weight", 1, ReadoutHidden);
            _clsB2 = Take(weights, "readout.classifier.3.bias", 1);
            _brW1 = Take(weights, "readout.bridge.0.weight", BridgeHidden, 1);
            _brB1 = Take(weights, "readout.bridge.0.bias", BridgeHidden);
            _brW2 = Take(weights, "readout.bridge.2.weight", spec.OutputDim, BridgeHidden);
            _brB2 = Take(weights, "readout.bridge.2.bias", spec.OutputDim);
        }
    }

    /// <summary>The projector's shape.</summary>
    public SciProjectorSpec Spec => _s;

    /// <summary>Projects <paramref name="rows"/> feature rows into <c>[rows, OutputDim]</c>.</summary>
    /// <param name="features"><c>[rows, FeatureDim]</c>.</param>
    /// <param name="positions">Each row's index into the position embedding (0-based, below <see cref="SciProjectorSpec.MaxAtoms"/>); the objects' own row numbers.</param>
    /// <param name="discrete"><c>[rows, DiscreteSizes.Count]</c> category ids, or empty when the projector has none.</param>
    /// <param name="side"><c>[rows, SideDim]</c>, or empty when the projector has none.</param>
    public float[] Forward(ReadOnlySpan<float> features, int rows, ReadOnlySpan<int> positions, ReadOnlySpan<int> discrete, ReadOnlySpan<float> side)
    {
        SciProjectorSpec s = _s;
        int f = s.FeatureDim, idx = s.IndexDim, dd = s.DiscreteDim, nd = s.DiscreteSizes.Count, width = s.InputWidth;
        if (features.Length != (long)rows * f) throw new ArgumentException($"features must hold rows x {f} values.", nameof(features));
        if (positions.Length != rows) throw new ArgumentException("positions must hold one entry per row.", nameof(positions));
        if (discrete.Length != rows * nd) throw new ArgumentException($"discrete must hold rows x {nd} ids.", nameof(discrete));
        if (side.Length != rows * s.SideDim) throw new ArgumentException($"side must hold rows x {s.SideDim} values.", nameof(side));

        float[] x = new float[rows * width];
        for (int r = 0; r < rows; r++)
        {
            int p = positions[r];
            if ((uint)p >= (uint)s.MaxAtoms) throw new ArgumentOutOfRangeException(nameof(positions), $"Row position {p} is outside the {s.MaxAtoms} the projector learned.");
            Span<float> row = x.AsSpan(r * width, width);
            features.Slice(r * f, f).CopyTo(row);
            _atomIndex.AsSpan(p * idx, idx).CopyTo(row[f..]);
            for (int c = 0; c < nd; c++)
            {
                int id = discrete[r * nd + c];
                if ((uint)id >= (uint)s.DiscreteSizes[c]) throw new ArgumentOutOfRangeException(nameof(discrete), $"Discrete column {c} id {id} is outside {s.DiscreteSizes[c]} categories.");
                _discrete[c].AsSpan(id * dd, dd).CopyTo(row[(f + idx + c * dd)..]);
            }
        }
        float[] normed = new float[x.Length];
        for (int r = 0; r < rows; r++) LayerNorm(x.AsSpan(r * width, width), _normW, _normB, normed.AsSpan(r * width, width));
        float[] hidden = HostMath.Linear(normed, _w1, rows, width, s.HiddenDim);
        AddBias(hidden, _b1, rows);
        Gelu(hidden);
        float[] output = HostMath.Linear(hidden, _w2, rows, s.HiddenDim, s.OutputDim);
        AddBias(output, _b2, rows);

        if (_sideW1 is not null)
        {
            float[] sh = HostMath.Linear(side, _sideW1, rows, s.SideDim, s.SideHiddenDim);
            Gelu(sh);
            Add(output, HostMath.Linear(sh, _sideW2!, rows, s.SideHiddenDim, s.OutputDim));
        }
        if (_mean is not null) Add(output, Readout(features, rows));
        return output;
    }

    // The frozen classifier's probability per row, mapped by the learned bridge: (values - mean) / std, LayerNorm, MLP, sigmoid, then 1 -> 128 -> out.
    private float[] Readout(ReadOnlySpan<float> features, int rows)
    {
        int f = _s.FeatureDim;
        float[] normalized = new float[rows * f], ln = new float[rows * f];
        for (int i = 0; i < normalized.Length; i++) normalized[i] = (features[i] - _mean![i % f]) / _std![i % f];
        for (int r = 0; r < rows; r++) LayerNorm(normalized.AsSpan(r * f, f), _clsNormW!, _clsNormB!, ln.AsSpan(r * f, f));
        float[] h = HostMath.Linear(ln, _clsW1!, rows, f, ReadoutHidden);
        AddBias(h, _clsB1!, rows);
        Gelu(h);
        float[] logit = HostMath.Linear(h, _clsW2!, rows, ReadoutHidden, 1);
        float[] prob = new float[rows];
        for (int r = 0; r < rows; r++) prob[r] = 1f / (1f + MathF.Exp(-(logit[r] + _clsB2![0])));
        float[] b = HostMath.Linear(prob, _brW1!, rows, 1, BridgeHidden);
        AddBias(b, _brB1!, rows);
        Gelu(b);
        float[] o = HostMath.Linear(b, _brW2!, rows, BridgeHidden, _s.OutputDim);
        AddBias(o, _brB2!, rows);
        return o;
    }

    private static void LayerNorm(ReadOnlySpan<float> x, float[] w, float[] b, Span<float> y)
    {
        float mean = 0f;
        foreach (float v in x) mean += v;
        mean /= x.Length;
        float var = 0f;
        foreach (float v in x) var += (v - mean) * (v - mean);
        float inv = 1f / MathF.Sqrt(var / x.Length + LayerNormEps);
        for (int i = 0; i < x.Length; i++) y[i] = (x[i] - mean) * inv * w[i] + b[i];
    }

    // nn.GELU() default: the exact erf form.
    private void Gelu(float[] data)
    {
        using Tensor input = HostMath.Tensor(data, data.Length);
        using Tensor output = new(new TensorShape(data.Length), DType.F32);
        _backend.GeluErf(output, input);
        output.AsReadOnlySpan<float>().CopyTo(data);
    }

    private static void AddBias(float[] y, float[] bias, int rows)
    {
        int n = bias.Length;
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < n; j++) y[r * n + j] += bias[j];
    }

    private static void Add(float[] y, float[] x)
    {
        for (int i = 0; i < y.Length; i++) y[i] += x[i];
    }

    private static float[] Take(IReadOnlyDictionary<string, Tensor> weights, string name, params int[] shape)
    {
        if (!weights.TryGetValue(name, out Tensor? t)) throw new InvalidDataException($"The projector has no tensor '{name}'.");
        long expected = 1;
        foreach (int d in shape) expected *= d;
        if (t.ElementCount != expected) throw new InvalidDataException($"'{name}' holds {t.ElementCount} values; the projector config needs {expected}.");
        Tensor f32 = TensorCasts.EnsureF32(t);
        try
        {
            return f32.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            if (!ReferenceEquals(f32, t)) f32.Dispose();
        }
    }
}

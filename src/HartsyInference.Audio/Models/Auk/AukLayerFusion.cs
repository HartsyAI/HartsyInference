using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>AuK thinker-layer fusion: <c>out = layer_scale · Σ_i softmax(layer_weights)_i · LayerNorm(h_i)</c> (no affine, eps 1e-5), accumulated layer by layer so only one output and one scratch tensor are live.</summary>
/// <remarks>Hidden states follow HF <c>output_hidden_states[1:]</c>: layers 0..N-2 are raw layer outputs (feed them through <see cref="OnLayer"/> as the <c>layerTap</c>), and layer N-1 is the FINAL-NORMED output, passed to <see cref="Complete"/> (the tap's raw value for the last layer is ignored).</remarks>
public sealed class AukLayerFusion : IDisposable
{
    public const string LayerWeightsKey = "layer_weights";
    public const string LayerScaleKey = "layer_scale";

    private readonly IBackend _backend;
    private readonly double[] _coeffs;
    private readonly int _hidden;
    private readonly float _eps;
    private Tensor? _acc;
    private Tensor? _scratch;
    private Tensor? _scaled;
    private int _tokens;
    private int _seen;

    /// <summary>Number of fused layers.</summary>
    public int Layers => _coeffs.Length;

    /// <summary>Per-layer coefficient <c>softmax(layer_weights)_i · layer_scale</c> (softmax in double).</summary>
    public ReadOnlySpan<double> Coefficients => _coeffs;

    public AukLayerFusion(IBackend backend, ReadOnlySpan<float> layerWeights, float layerScale, int hidden = 2_048, float eps = 1e-5f)
    {
        if (layerWeights.Length < 1) throw new ArgumentException("layer_weights must not be empty.", nameof(layerWeights));
        if (hidden < 1) throw new ArgumentOutOfRangeException(nameof(hidden));
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _hidden = hidden;
        _eps = eps;
        _coeffs = Softmax(layerWeights, layerScale);
    }

    /// <summary>Builds from the checkpoint's top-level <c>layer_weights</c> [N] and <c>layer_scale</c> [1] tensors.</summary>
    public static AukLayerFusion FromCheckpoint(IBackend backend, IReadOnlyDictionary<string, Tensor> weights, int hidden = 2_048,
        float eps = 1e-5f)
    {
        if (!weights.TryGetValue(LayerWeightsKey, out Tensor? lw)) throw new KeyNotFoundException($"Missing checkpoint tensor '{LayerWeightsKey}'.");
        if (!weights.TryGetValue(LayerScaleKey, out Tensor? ls)) throw new KeyNotFoundException($"Missing checkpoint tensor '{LayerScaleKey}'.");
        if (lw.Shape.Rank != 1) throw new ArgumentException($"{LayerWeightsKey} must be rank 1, got rank {lw.Shape.Rank}.");
        if (ls.Shape.ElementCount != 1) throw new ArgumentException($"{LayerScaleKey} must have one element.");
        Tensor lwF32 = TensorCasts.EnsureF32(lw);
        Tensor lsF32 = TensorCasts.EnsureF32(ls);
        try
        {
            return new AukLayerFusion(backend, lwF32.AsSpan<float>(), lsF32.AsSpan<float>()[0], hidden, eps);
        }
        finally
        {
            if (!ReferenceEquals(lwF32, lw)) lwF32.Dispose();
            if (!ReferenceEquals(lsF32, ls)) lsF32.Dispose();
        }
    }

    /// <summary>Starts a fusion over <paramref name="tokens"/> rows, discarding any unfinished one.</summary>
    public void Begin(int tokens)
    {
        if (tokens < 1) throw new ArgumentOutOfRangeException(nameof(tokens));
        ReleaseBuffers();
        _tokens = tokens;
        _seen = 0;
        TensorShape shape = new(1, tokens, _hidden);
        _acc = new Tensor(shape, DType.F32);
        _scratch = new Tensor(shape, DType.F32);
        _scaled = new Tensor(shape, DType.F32);
        _acc.AsSpan<float>().Clear();
    }

    /// <summary>Layer-tap callback for <c>GenericTransformer.ForwardEmbeds</c>: accumulates layers 0..N-2 and ignores the raw last layer, which <see cref="Complete"/> takes post-final-norm.</summary>
    public void OnLayer(int layer, Tensor hidden)
    {
        if (layer == Layers - 1) return;
        Accumulate(layer, hidden);
    }

    /// <summary>Adds layer <paramref name="layer"/>; layers must arrive in order 0..N-1.</summary>
    public void Accumulate(int layer, Tensor hidden)
    {
        if (_acc is null) throw new InvalidOperationException("Call Begin before Accumulate.");
        if (layer != _seen || layer >= Layers) throw new ArgumentException($"Expected layer {_seen}, got {layer}.", nameof(layer));
        if (hidden.Shape.ElementCount != (long)_tokens * _hidden)
            throw new ArgumentException($"Layer {layer} hidden has {hidden.Shape.ElementCount} elements; expected {_tokens}x{_hidden}.", nameof(hidden));
        _backend.LayerNormNoAffine(_scratch!, hidden, _eps);
        _backend.Scale(_scaled!, _scratch!, (float)_coeffs[layer]);
        _backend.Add(_acc, _acc, _scaled!);
        _seen++;
    }

    /// <summary>Adds the final-normed last layer and returns the fused <c>[1, tokens, hidden]</c> F32 tensor, owned by the caller.</summary>
    public Tensor Complete(Tensor finalNormed)
    {
        Accumulate(Layers - 1, finalNormed);
        Tensor result = _acc!;
        _acc = null;
        ReleaseBuffers();
        return result;
    }

    public void Dispose() => ReleaseBuffers();

    private void ReleaseBuffers()
    {
        _acc?.Dispose();
        _scratch?.Dispose();
        _scaled?.Dispose();
        _acc = _scratch = _scaled = null;
    }

    private static double[] Softmax(ReadOnlySpan<float> weights, float scale)
    {
        double max = double.NegativeInfinity;
        foreach (float w in weights) max = Math.Max(max, w);
        double[] result = new double[weights.Length];
        double sum = 0;
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = Math.Exp(weights[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < result.Length; i++) result[i] = result[i] / sum * scale;
        return result;
    }
}

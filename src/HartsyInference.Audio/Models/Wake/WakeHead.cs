using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Wake;

/// <summary>One wake word's classifier: 16 stacked <see cref="SpeechEmbeddingModel"/> embeddings (1.28 s of
/// context) flattened to 1536, through two hidden layers, to a single probability.
///
/// <para>The shipped openWakeWord heads are <b>not</b> one architecture — <c>alexa</c> has no LayerNorm,
/// <c>hey_mycroft</c> and <c>hey_jarvis</c> do, and <c>hey_jarvis</c> prefixes its weights with <c>model.</c>
/// while also bundling a second <c>verifier_model.*</c> network in the same file. So the shape is discovered
/// from the weight names rather than assumed: entries are <c>nn.Sequential</c> indices, a rank-2 weight is a
/// Linear and a rank-1 weight is a LayerNorm, and each hidden Linear is followed by its optional LayerNorm and
/// then ReLU. Heads trained in-engine are written with the same naming, so they load through this one path.</para></summary>
public sealed unsafe class WakeHead : IDisposable
{
    /// <summary>Embedding frames the head consumes.</summary>
    public const int ContextFrames = 16;
    /// <summary>Flattened input width.</summary>
    public const int InputDim = ContextFrames * SpeechEmbeddingModel.EmbeddingDim;

    private const float LayerNormEps = 1e-5f;

    private readonly List<Tensor> _owned = [];
    private Block[] _hidden = [];
    private Tensor? _outW;
    private Tensor? _outB;
    private int _disposed;

    /// <summary>The wake word this head detects, used in detection events and config.</summary>
    public string Name { get; }

    public WakeHead(string name)
    {
        if (string.IsNullOrWhiteSpace(name)) throw new ArgumentException("Wake head needs a name.", nameof(name));
        Name = name;
    }

    /// <summary>Loads a head, auto-detecting the <c>model.</c> prefix and the presence of LayerNorm.
    /// <c>verifier_model.*</c> tensors, which ship alongside the main network in some files, are ignored.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        string prefix = weights.Keys.Any(static k => k.StartsWith("model.", StringComparison.Ordinal)) ? "model." : "";

        List<(int Index, Tensor Weight, Tensor? Bias, bool IsLinear)> entries = [];
        for (int i = 0; i <= 16; i++)
        {
            if (!weights.TryGetValue($"{prefix}{i}.weight", out Tensor? w)) continue;
            weights.TryGetValue($"{prefix}{i}.bias", out Tensor? b);
            entries.Add((i, Track(WhisperOps.EnsureF32(w)), b is null ? null : Track(WhisperOps.EnsureF32(b)), w.Shape.Rank == 2));
        }

        int linearCount = entries.Count(static e => e.IsLinear);
        if (linearCount < 2)
            throw new HartsyInferenceException($"Wake head '{Name}' needs at least two Linear layers, found {linearCount}. Is this an openWakeWord-family head?");

        List<Block> hidden = [];
        for (int i = 0; i < entries.Count; i++)
        {
            if (!entries[i].IsLinear) continue;
            bool isOutput = entries.Skip(i + 1).All(static e => !e.IsLinear);
            if (isOutput)
            {
                _outW = entries[i].Weight;
                _outB = entries[i].Bias;
                break;
            }
            bool hasNorm = i + 1 < entries.Count && !entries[i + 1].IsLinear;
            hidden.Add(new Block(entries[i].Weight, entries[i].Bias,
                hasNorm ? entries[i + 1].Weight : null, hasNorm ? entries[i + 1].Bias : null));
        }
        _hidden = [.. hidden];

        if (_outW is null) throw new HartsyInferenceException($"Wake head '{Name}' has no output layer.");
        if (_hidden.Length == 0 || _hidden[0].Weight.Shape[1] != InputDim)
            throw new HartsyInferenceException($"Wake head '{Name}' expects a {InputDim}-wide input, got {(_hidden.Length == 0 ? 0 : _hidden[0].Weight.Shape[1])}.");
        if (_outW.Shape[0] != 1)
            throw new HartsyInferenceException($"Wake head '{Name}' must emit one logit, got {_outW.Shape[0]}.");
    }

    private Tensor Track(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    /// <summary>Scores <paramref name="features"/> (<c>16 × 96</c> row-major, oldest frame first), returning the
    /// wake probability in <c>[0, 1]</c>.</summary>
    public float Score(IBackend backend, ReadOnlySpan<float> features)
    {
        if (_outW is null) throw new InvalidOperationException($"Wake head '{Name}' weights not loaded.");
        if (features.Length != InputDim)
            throw new ArgumentException($"Wake head input must be {InputDim} floats, got {features.Length}.", nameof(features));

        Tensor x = new(new TensorShape(1, InputDim), DType.F32);
        features.CopyTo(x.AsSpan<float>());

        foreach (Block block in _hidden)
        {
            int outDim = (int)block.Weight.Shape[0];
            Tensor h = new(new TensorShape(1, outDim), DType.F32);
            backend.Linear(h, x, block.Weight, block.Bias);
            x.Dispose();

            if (block.NormWeight is not null && block.NormBias is not null)
            {
                Tensor n = new(new TensorShape(1, outDim), DType.F32);
                backend.LayerNorm(n, h, block.NormWeight, block.NormBias, LayerNormEps);
                h.Dispose();
                h = n;
            }

            float* p = (float*)h.DataPointer;
            for (int i = 0; i < outDim; i++)
                if (p[i] < 0f) p[i] = 0f;
            x = h;
        }

        using Tensor logit = new(new TensorShape(1, 1), DType.F32);
        backend.Linear(logit, x, _outW, _outB);
        x.Dispose();

        float v = ((float*)logit.DataPointer)[0];
        return 1f / (1f + MathF.Exp(-v));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }

    private readonly record struct Block(Tensor Weight, Tensor? Bias, Tensor? NormWeight, Tensor? NormBias);
}

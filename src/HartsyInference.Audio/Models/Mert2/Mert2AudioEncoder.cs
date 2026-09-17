using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Mert2;

/// <summary>The complete audio side of SheetSage2: mel frontend, MERT-v2 encoder, the learned 25-way layer mix,
/// and the 1024→512 projection whose output the score decoder cross-attends over.
///
/// <para>The mix parameter and the projection sit on the parent module in the released checkpoint, not inside
/// MERT2, but they are what turns hidden states into something a decoder can read, so they live here rather than
/// leaving every caller to reimplement them.</para>
///
/// <para>Input is mono 24 kHz. A clip shorter than the 300-second window is zero-padded to it, exactly as the
/// released model does — the encoder attends over the padding too, so the token count is always 7500 and the
/// working-set size never varies with clip length.</para></summary>
public sealed class Mert2AudioEncoder : IDisposable
{
    private readonly Mert2Config _config;
    private readonly Mert2MelFrontend _frontend;
    private readonly Mert2Encoder _encoder;
    private readonly float[] _mixWeights;
    private readonly List<Tensor> _owned = [];

    private Tensor? _projectionWeight;
    private Tensor? _projectionBias;
    private bool _loaded;
    private int _disposed;

    /// <summary>Builds the stack from <paramref name="config"/>; call <see cref="LoadWeights"/> before use.</summary>
    public Mert2AudioEncoder(Mert2Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _frontend = new Mert2MelFrontend(config);
        _encoder = new Mert2Encoder(config);
        _mixWeights = new float[config.Layers + 1];
    }

    /// <summary>The geometry this stack was built for.</summary>
    public Mert2Config Config => _config;

    /// <summary>The mel frontend, exposed so callers can reuse a computed spectrogram across encodes.</summary>
    public Mert2MelFrontend Frontend => _frontend;

    /// <summary>The encoder body, for callers that already hold a mel spectrogram.</summary>
    public Mert2Encoder Encoder => _encoder;

    /// <summary>Loads every tensor from the released single-file checkpoint. Its keys carry no model prefix:
    /// the encoder is under <c>encoder.*</c> while the mix and projection sit at the top level.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _frontend.LoadWeights(weights);
        _encoder.LoadWeights(weights);
        _projectionWeight = Mert2Ops.Load(weights, "encoder_projection.weight", _owned, _config.ProjectionDim, _config.Dim);
        _projectionBias = Mert2Ops.Load(weights, "encoder_projection.bias", _owned, _config.ProjectionDim);

        Tensor mix = Mert2Ops.Load(weights, "layer_weight", _owned, _mixWeights.Length);
        Softmax(mix.AsReadOnlySpan<float>(), _mixWeights);
        _loaded = true;
    }

    /// <summary>Turns a mono 24 kHz clip into the decoder's memory, <c>[tokens, projectionDim]</c>. The result is
    /// freshly allocated and owned by the caller.</summary>
    public Tensor Encode(IBackend backend, ReadOnlySpan<float> clip)
    {
        ThrowIfDisposed();
        if (!_loaded) throw new InvalidOperationException("Call LoadWeights before Encode.");
        ArgumentNullException.ThrowIfNull(backend);
        using Tensor mel = _frontend.Compute(clip);
        return EncodeMel(backend, mel);
    }

    /// <summary>The half of <see cref="Encode"/> that runs on the backend, for callers holding a mel spectrogram
    /// from <see cref="Mert2MelFrontend.Compute"/>.</summary>
    public Tensor EncodeMel(IBackend backend, Tensor mel)
    {
        ThrowIfDisposed();
        if (!_loaded) throw new InvalidOperationException("Call LoadWeights before EncodeMel.");
        ArgumentNullException.ThrowIfNull(backend);
        using Tensor mixed = _encoder.Forward(backend, mel, _mixWeights);
        Tensor projected = new(new TensorShape((int)mixed.Shape[0], _config.ProjectionDim), DType.F32);
        try
        {
            backend.Linear(projected, mixed, _projectionWeight!, _projectionBias);
            return projected;
        }
        catch
        {
            projected.Dispose();
            throw;
        }
    }

    /// <summary>The softmaxed layer-mix weights, one per hidden state the encoder produces.</summary>
    public ReadOnlySpan<float> MixWeights => _mixWeights;

    /// <summary>Every tensor the backend should preload before an encode.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor weight in _encoder.EnumerateWeights()) yield return weight;
        if (_projectionWeight is not null) yield return _projectionWeight;
        if (_projectionBias is not null) yield return _projectionBias;
    }

    /// <summary>Frees the F32 copies this model made of the checkpoint's BF16 tensors.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _encoder.Dispose();
        for (int i = 0; i < _owned.Count; i++) _owned[i].Dispose();
        _owned.Clear();
    }

    private static void Softmax(ReadOnlySpan<float> logits, Span<float> result)
    {
        float peak = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++) peak = MathF.Max(peak, logits[i]);
        float total = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = MathF.Exp(logits[i] - peak);
            total += result[i];
        }
        for (int i = 0; i < logits.Length; i++) result[i] /= total;
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
}

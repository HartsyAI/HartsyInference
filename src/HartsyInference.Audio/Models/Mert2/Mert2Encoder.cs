using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Mert2;

/// <summary>MERT-v2's encoder body: the ConvNeXt subsampling stack, 24 Conformer layers, and the learned mix that
/// collapses the subsampler output plus every layer's hidden state into one sequence.
///
/// <para>The mix weights are a softmax over <c>layer_weight</c>, which the released checkpoint stores on the
/// parent SheetSage2 module rather than inside the encoder — see <see cref="Mert2AudioEncoder"/>, which owns it
/// along with the projection the decoder actually reads.</para>
///
/// <para>Batching is deliberately not supported. GlobalResponseNorm reduces over the entire time axis, so two
/// clips in one call are not two independent forwards, and the production path is always a single 300-second
/// mono window.</para></summary>
public sealed class Mert2Encoder : IDisposable
{
    private readonly Mert2Config _config;
    private readonly Mert2ConvNextBlock[] _blocks;
    private readonly Mert2ConformerLayer[] _layers;
    private readonly List<Tensor> _owned = [];
    private bool _loaded;
    private int _disposed;

    /// <summary>Builds the stack from <paramref name="config"/>; call <see cref="LoadWeights"/> before
    /// <see cref="Forward"/>.</summary>
    public Mert2Encoder(Mert2Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _blocks = new Mert2ConvNextBlock[config.Depths.Count];
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i] = new Mert2ConvNextBlock(config, config.Channels[i], config.Channels[i + 1],
                config.Strides[i], config.Depths[i]);
        }
        _layers = new Mert2ConformerLayer[config.Layers];
        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i] = new Mert2ConformerLayer(config);
        }
    }

    /// <summary>The geometry this stack was built for.</summary>
    public Mert2Config Config => _config;

    /// <summary>Number of hidden states the mix spans: the subsampler output plus one per Conformer layer.</summary>
    public int MixInputCount => _layers.Length + 1;

    /// <summary>Loads the encoder from the released single-file checkpoint, whose keys are unprefixed apart from
    /// the module path (<c>encoder.subsampling_module.*</c>, <c>encoder.layers.N.*</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "encoder")
    {
        ArgumentNullException.ThrowIfNull(weights);
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].LoadWeights(weights, $"{prefix}.subsampling_module.{i}");
        }
        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].LoadWeights(weights, $"{prefix}.layers.{i}", _owned);
        }
        _loaded = true;
    }

    /// <summary>Runs the encoder over <c>mel [frames, melBins]</c> and returns the mixed hidden states
    /// <c>[tokens, dim]</c>, freshly allocated and owned by the caller.</summary>
    /// <param name="mixWeights">Softmaxed <c>layer_weight</c>, <see cref="MixInputCount"/> long: element 0 weights
    /// the subsampler output and element <c>i+1</c> weights layer <c>i</c>.</param>
    public Tensor Forward(IBackend backend, Tensor mel, ReadOnlySpan<float> mixWeights)
    {
        ThrowIfDisposed();
        if (!_loaded) throw new InvalidOperationException("Call LoadWeights before Forward.");
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(mel);
        if (mel.Shape.Rank != 2 || (int)mel.Shape[1] != _config.MelBins)
            throw new HartsyInferenceException($"MERT2 expects mel [frames, {_config.MelBins}]; got {mel.Shape}.");
        if (mixWeights.Length != MixInputCount)
            throw new HartsyInferenceException($"MERT2 needs {MixInputCount} mix weights; got {mixWeights.Length}.");

        Tensor subsampled = Subsample(backend, mel);
        int tokens = (int)subsampled.Shape[0];
        TensorShape shape = new(tokens, _config.Dim);
        // The scratch is by far the largest allocation, so it goes first: an out-of-memory here frees everything,
        // and the accumulator that survives the method is nulled out rather than disposed.
        Mert2ConformerLayer.Scratch? scratch = null;
        Tensor? mixed = null, mixedSpare = null, hidden = null, hiddenSpare = null;
        try
        {
            scratch = new Mert2ConformerLayer.Scratch(_config, tokens);
            mixed = new Tensor(shape, DType.F32);
            mixedSpare = new Tensor(shape, DType.F32);
            hidden = new Tensor(shape, DType.F32);
            hiddenSpare = new Tensor(shape, DType.F32);

            backend.Scale(mixed, subsampled, mixWeights[0]);
            Tensor current = subsampled;
            for (int i = 0; i < _layers.Length; i++)
            {
                Tensor next = ReferenceEquals(current, hidden) ? hiddenSpare : hidden;
                _layers[i].Forward(backend, current, next, scratch);
                backend.AffineMix(mixedSpare, mixed, next, 1f, mixWeights[i + 1]);
                (mixed, mixedSpare) = (mixedSpare, mixed);
                current = next;
            }
            Tensor result = mixed;
            mixed = null;
            return result;
        }
        finally
        {
            scratch?.Dispose();
            subsampled.Dispose();
            mixed?.Dispose();
            mixedSpare?.Dispose();
            hidden?.Dispose();
            hiddenSpare?.Dispose();
        }
    }

    /// <summary>Every tensor the backend should preload before a forward.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < _blocks.Length; i++)
        {
            foreach (Tensor weight in _blocks[i].EnumerateWeights()) yield return weight;
        }
        for (int i = 0; i < _layers.Length; i++)
        {
            foreach (Tensor weight in _layers[i].EnumerateWeights()) yield return weight;
        }
    }

    /// <summary>Frees the F32 copies this model made of the checkpoint's BF16 tensors.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].Dispose();
        for (int i = 0; i < _owned.Count; i++) _owned[i].Dispose();
        _owned.Clear();
    }

    /// <summary>The ConvNeXt stack alone, <c>mel [frames, melBins]</c> to <c>[tokens, dim]</c>. Internal so the
    /// parity gate can isolate the subsampler from the Conformer stack.</summary>
    internal Tensor Subsample(IBackend backend, Tensor mel)
    {
        Tensor current = _blocks[0].Forward(backend, mel);
        try
        {
            for (int i = 1; i < _blocks.Length; i++)
            {
                Tensor next = _blocks[i].Forward(backend, current);
                current.Dispose();
                current = next;
            }
        }
        catch
        {
            // The caller owns the return value, but only on the way out; a block that throws leaves the
            // intermediate owned by nobody.
            current.Dispose();
            throw;
        }
        return current;
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
}

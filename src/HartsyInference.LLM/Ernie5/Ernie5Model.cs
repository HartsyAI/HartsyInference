using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Mhc;

namespace HartsyInference.LLM.Ernie5;

/// <summary>Host reference model of ERNIE-5.5 (vLLM-ERNIE <c>Ernie5_5ForCausalLM</c>): the token embedding broadcast to <c>hc_mult</c> residual
/// streams, the decoder layers (HCA or DSA attention, dense or latent-MoE SiTU FFN, each under mHC), the mHC head read-out with the final
/// RMS norm, the LM head, and multimax. MTP layers are not loaded.</summary>
/// <remarks>
/// <para>Every op runs through <see cref="IBackend"/> on the CPU backend. Device backends are refused by name: no CUDA or Vulkan kernel takes the
/// checkpoint's 128x128 block-FP8 weights, and the DSA indexer's Hadamard and FP8x128 round trip are host-only.</para>
/// <para>FP8 weights are dequantized to F32 when a layer is built. <see cref="Ernie5LoadOptions.ResidentWeights"/> keeps every built layer;
/// otherwise each layer is built and dropped per forward. Not thread-safe: run one sequence at a time.</para>
/// </remarks>
public sealed class Ernie5Model : IDisposable
{
    private readonly IBackend _backend;
    private readonly IErnie5WeightSource _source;
    private readonly IDisposable? _ownedSource;
    private readonly Ernie5LoadOptions _options;
    private readonly DeepSeekV41RopeTable _hcaRope;
    private readonly DeepSeekV41RopeTable _dsaRope;
    private readonly Ernie5Layer?[] _layers;
    private readonly HyperConnection _head;
    private readonly float[] _headFn;
    private readonly float[] _headScale;
    private readonly float[] _headBase;
    private readonly float[] _finalNorm;
    private readonly float[]? _multimaxRanges;
    private readonly float[]? _multimaxGains;
    private Tensor? _embedding;
    private Tensor? _lmHead;
    private int _disposed;

    /// <param name="backend">The CPU backend.</param>
    /// <param name="config">The checkpoint's config.</param>
    /// <param name="source">The weights; every key must map (<see cref="Ernie5WeightMap"/>). Not owned unless loaded through
    /// <see cref="Load"/>.</param>
    /// <param name="options">Sequence length and weight residency.</param>
    /// <exception cref="NotSupportedException">The backend is not the CPU, or the config names a feature the model does not implement.</exception>
    /// <exception cref="InvalidDataException">A checkpoint key maps to nothing, or a parameter has no key.</exception>
    public Ernie5Model(IBackend backend, Ernie5Config config, IErnie5WeightSource source, Ernie5LoadOptions options)
        : this(backend, config, source, options, ownedSource: null)
    {
    }

    private Ernie5Model(IBackend backend, Ernie5Config config, IErnie5WeightSource source, Ernie5LoadOptions options, IDisposable? ownedSource)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.MaxSequenceTokens, 1);
        if (backend.Device != DeviceKind.Cpu)
            throw new NotSupportedException($"ERNIE-5.5 runs on the host reference path only; the {backend.Device} backend has no kernels "
                + "for its 128x128 block-FP8 weights, and the DSA indexer's Hadamard and FP8x128 round trip are host-only.");
        IReadOnlyList<string> missing = config.UnimplementedFeatures();
        if (missing.Count > 0)
            throw new NotSupportedException($"This ERNIE-5.5 config names features the model does not implement: {string.Join("; ", missing)}.");
        Coverage = Ernie5WeightMap.Check(config, source.Names);
        Coverage.RequireComplete();

        _backend = backend;
        Config = config;
        _source = source;
        _ownedSource = ownedSource;
        _options = options;
        int max = options.MaxSequenceTokens;
        _hcaRope = DeepSeekV41RopeTable.Build(config.QkRopeHeadDim, max, config.CompressRopeTheta, config.HcaRopeScaling);
        _dsaRope = DeepSeekV41RopeTable.Build(config.MlaQkRopeHeadDim, max, config.RopeTheta, null);
        _layers = new Ernie5Layer?[config.NumHiddenLayers];
        int hc = config.HcMult, flat = hc * config.HiddenSize, mixes = (2 + hc) * hc;
        // the head read-out uses only the formula config; its own projection and gates are passed to Head
        _head = new HyperConnection(backend, Ernie5Layer.HcConfig(config), new float[mixes * flat], new float[3], new float[mixes]);
        _headFn = Ernie5Layer.Transpose(Ernie5Layer.HostF32(source, Ernie5WeightNames.HeadFn), flat, hc);
        _headScale = Ernie5Layer.HostF32(source, Ernie5WeightNames.HeadScale);
        _headBase = Ernie5Layer.HostF32(source, Ernie5WeightNames.HeadBase);
        _finalNorm = Ernie5Layer.HostF32(source, Ernie5WeightNames.FinalNorm);
        if (config.Multimax is not null)
        {
            _multimaxRanges = Ernie5Layer.HostF32(source, Ernie5WeightNames.MultimaxRanges);
            _multimaxGains = Ernie5Layer.HostF32(source, Ernie5WeightNames.MultimaxGains);
        }
        try
        {
            _embedding = source.Read(Ernie5WeightNames.Embedding);
            _lmHead = source.Read(Ernie5WeightNames.LmHead);
            RequireMatrix(_embedding, config.VocabSize, config.HiddenSize, Ernie5WeightNames.Embedding);
            RequireMatrix(_lmHead, config.VocabSize, config.HiddenSize, Ernie5WeightNames.LmHead);
            if (options.ResidentWeights)
                for (int i = 0; i < _layers.Length; i++) _layers[i] = new Ernie5Layer(backend, config, i, source, _hcaRope, _dsaRope);
        }
        catch
        {
            DisposeWeights();
            throw;
        }
    }

    /// <summary>Opens a safetensors directory and loads it; the model owns the checkpoint.</summary>
    public static Ernie5Model Load(string directory, IBackend backend, Ernie5LoadOptions options)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        Ernie5Config config = Ernie5Config.Load(Path.Combine(directory, "config.json"));
        Ernie5CheckpointWeights weights = Ernie5CheckpointWeights.Open(directory, config.QuantBlock is [int block, _] ? block : 128);
        try
        {
            return new Ernie5Model(backend, config, weights, options, weights);
        }
        catch
        {
            weights.Dispose();
            throw;
        }
    }

    /// <summary>The checkpoint's config.</summary>
    public Ernie5Config Config { get; }

    /// <summary>How the checkpoint's keys mapped onto the model at load.</summary>
    public Ernie5WeightCoverage Coverage { get; }

    /// <summary>A new, empty sequence state of at most <see cref="Ernie5LoadOptions.MaxSequenceTokens"/> tokens.</summary>
    public Ernie5SequenceState CreateState(int capacity)
    {
        if (capacity > _options.MaxSequenceTokens)
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity,
                $"The model was loaded for sequences of up to {_options.MaxSequenceTokens} tokens.");
        return new Ernie5SequenceState(Config, capacity);
    }

    /// <summary>Appends <paramref name="tokenIds"/> to <paramref name="state"/> and returns the logits, <c>[rows, vocab]</c> after multimax: the last
    /// row only, or every row when <paramref name="allRows"/>.</summary>
    public float[] Forward(ReadOnlySpan<int> tokenIds, Ernie5SequenceState state, bool allRows = false)
    {
        ArgumentNullException.ThrowIfNull(state);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int tokens = tokenIds.Length, d = Config.HiddenSize, hc = Config.HcMult, vocab = Config.VocabSize, startPos = state.Length;
        if (state.Faulted) throw new InvalidOperationException("An earlier Forward on this state failed partway through; call Reset() before reusing it.");
        if (tokens < 1) throw new ArgumentException("Forward needs at least one token.", nameof(tokenIds));
        if (startPos + tokens > state.Capacity)
            throw new InvalidOperationException($"The sequence state holds {state.Capacity} tokens; {startPos + tokens} were asked for.");
        foreach (int id in tokenIds)
            if ((uint)id >= (uint)vocab) throw new ArgumentOutOfRangeException(nameof(tokenIds), id, "Token id is outside the vocabulary.");

        float[] stream = Embed(tokenIds, d, hc);
        bool completed = false;
        try
        {
            for (int i = 0; i < _layers.Length; i++)
            {
                Ernie5Layer layer = _layers[i] ?? new Ernie5Layer(_backend, Config, i, _source, _hcaRope, _dsaRope);
                try
                {
                    layer.Forward(stream, tokens, startPos, state);
                }
                finally
                {
                    if (!ReferenceEquals(layer, _layers[i])) layer.Dispose();
                }
            }
            completed = true;
        }
        finally
        {
            // Layers that ran already advanced their caches; Length no longer matches them.
            if (!completed) state.MarkFaulted();
        }
        state.Length = startPos + tokens;

        int rows = allRows ? tokens : 1;
        float[] hidden = new float[tokens * d];
        _head.Head(stream, tokens, _headFn, _headScale, _headBase, _finalNorm, hidden);
        using Tensor hiddenT = HostTensor(hidden.AsSpan((tokens - rows) * d, rows * d), rows, d);
        using Tensor logitsT = new(new TensorShape(rows, vocab), DType.F32);
        _backend.Linear(logitsT, hiddenT, _lmHead!, null);
        float[] logits = logitsT.AsReadOnlySpan<float>().ToArray();
        if (_multimaxRanges is not null) Ernie5Multimax.Apply(logits, _multimaxRanges, _multimaxGains!);
        return logits;
    }

    // The first layer's broadcast pre-mix (vLLM mhc_pre_broadcast) equals the ordinary pre-mix over hc identical copies of the embedding.
    private float[] Embed(ReadOnlySpan<int> tokenIds, int d, int hc)
    {
        float[] stream = new float[tokenIds.Length * hc * d];
        for (int t = 0; t < tokenIds.Length; t++)
        {
            using Tensor row = _embedding!.SliceRows(tokenIds[t], 1);
            using Tensor f32 = row.DType == DType.F32 ? row.To(row.Device) : row.CastTo(DType.F32);
            ReadOnlySpan<float> values = f32.AsReadOnlySpan<float>();
            for (int c = 0; c < hc; c++) values.CopyTo(stream.AsSpan((t * hc + c) * d, d));
        }
        return stream;
    }

    private static Tensor HostTensor(ReadOnlySpan<float> values, int rows, int cols)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static void RequireMatrix(Tensor t, int rows, int cols, string name)
    {
        if (t.Shape.Rank != 2 || t.Shape[0] != rows || t.Shape[1] != cols)
            throw new InvalidDataException($"'{name}' is {t.Shape}; the config needs [{rows}, {cols}].");
    }

    private void DisposeWeights()
    {
        foreach (Ernie5Layer? layer in _layers) layer?.Dispose();
        Array.Clear(_layers);
        Interlocked.Exchange(ref _embedding, null)?.Dispose();
        Interlocked.Exchange(ref _lmHead, null)?.Dispose();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        DisposeWeights();
        _ownedSource?.Dispose();
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Mhc;

namespace HartsyInference.LLM.Maxwell;

/// <summary>The Maxwell-V1.0-Flash text decoder on the CPU, CUDA or Vulkan backend: every projection and the MoE run through <see cref="IBackend"/>;
/// the residual stream, norms, RoPE and block-sparse attention are host arrays, so each op round-trips. MTP layers are not loaded.</summary>
public sealed class MaxwellModel : IDisposable
{
    private readonly IBackend _backend;
    private readonly IModelWeightSource _source;
    private readonly IDisposable? _ownedSource;
    private readonly MaxwellLoadOptions _options;
    private readonly MaxwellRope _rope;
    private readonly MaxwellLayer?[] _layers;
    private readonly HyperConnection _head;
    private readonly float[] _headFn;
    private readonly float[] _headScale;
    private readonly float[] _headBase;
    private readonly float[] _finalNorm;
    private Tensor? _embedding;
    private Tensor? _lmHead;
    private int _disposed;

    /// <param name="backend">The CPU, CUDA or Vulkan backend.</param>
    /// <param name="config">The checkpoint's config.</param>
    /// <param name="source">The weights; every main-model key must map (<see cref="MaxwellWeightMap"/>). Not owned unless loaded through <see cref="Load"/>.</param>
    /// <param name="options">Sequence length and weight residency.</param>
    /// <exception cref="InvalidDataException">A checkpoint key maps to nothing, or a parameter has no key.</exception>
    public MaxwellModel(IBackend backend, MaxwellConfig config, IModelWeightSource source, MaxwellLoadOptions options)
        : this(backend, config, source, options, ownedSource: null)
    {
    }

    private MaxwellModel(IBackend backend, MaxwellConfig config, IModelWeightSource source, MaxwellLoadOptions options, IDisposable? ownedSource)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfLessThan(options.MaxSequenceTokens, 1);
        if (backend.Device.IsGpu && !backend.Device.IsCuda && !backend.Device.IsVulkan)
            throw new NotSupportedException($"Maxwell has no path on the {backend.Device} backend; it runs on the CPU, CUDA and Vulkan backends.");
        Coverage = MaxwellWeightMap.Check(config, source.Names);
        Coverage.RequireComplete();
        _backend = backend;
        Config = config;
        _source = source;
        _ownedSource = ownedSource;
        _options = options;
        _rope = new MaxwellRope(config.HeadDim, options.MaxSequenceTokens, config.RopeTheta);
        _layers = new MaxwellLayer?[config.NumHiddenLayers];
        int hc = config.HcMult, flat = hc * config.HiddenSize, mixes = (2 + hc) * hc;
        // The head read-out uses only the formula config; its own projection and gates are passed to Head.
        _head = new HyperConnection(backend, MaxwellLayer.HcConfig(config), new float[mixes * flat], new float[3], new float[mixes]);
        string[] head = MaxwellWeightNames.HeadLeaves;
        _headFn = WeightReads.HostF32(source, MaxwellWeightNames.Key(MaxwellWeightNames.Head, head[0]));
        _headBase = WeightReads.HostF32(source, MaxwellWeightNames.Key(MaxwellWeightNames.Head, head[1]));
        _headScale = WeightReads.HostF32(source, MaxwellWeightNames.Key(MaxwellWeightNames.Head, head[2]));
        _finalNorm = WeightReads.HostF32(source, MaxwellWeightNames.FinalNorm);
        if (_headFn.Length != (long)hc * flat) throw new InvalidDataException($"{MaxwellWeightNames.Head}.{head[0]} holds {_headFn.Length} values, not {hc} x {flat}.");
        try
        {
            _embedding = source.Read(MaxwellWeightNames.Embedding);
            _lmHead = source.Read(MaxwellWeightNames.LmHead);
            // The gather reads plain values, so an FP8 table would read wrong ones; the release stores it BF16.
            if (_embedding.DType == DType.F8E4M3)
                throw new NotSupportedException($"'{MaxwellWeightNames.Embedding}' is FP8; the embedding gather needs BF16 or F32.");
            RequireMatrix(_embedding, config.VocabSize, config.HiddenSize, MaxwellWeightNames.Embedding);
            RequireMatrix(_lmHead, config.VocabSize, config.HiddenSize, MaxwellWeightNames.LmHead);
            if (options.ResidentWeights)
                for (int i = 0; i < _layers.Length; i++) _layers[i] = new MaxwellLayer(backend, config, i, source, _rope);
        }
        catch
        {
            DisposeWeights();
            throw;
        }
    }

    /// <summary>The checkpoint's config.</summary>
    public MaxwellConfig Config { get; }

    /// <summary>How the checkpoint's keys mapped onto the model at load.</summary>
    public CheckpointKeyCoverage Coverage { get; }

    /// <summary>Opens a safetensors directory (index shards or one <c>model.safetensors</c>) and loads it; the model owns the checkpoint.</summary>
    public static MaxwellModel Load(string directory, IBackend backend, MaxwellLoadOptions options)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        MaxwellConfig config = MaxwellConfig.Parse(File.ReadAllText(Path.Combine(directory, "config.json")));
        BlockFp8CheckpointWeights weights = BlockFp8CheckpointWeights.Open(directory, "Maxwell", config.Fp8Block > 0 ? config.Fp8Block : 128);
        try
        {
            return new MaxwellModel(backend, config, weights, options, weights);
        }
        catch
        {
            weights.Dispose();
            throw;
        }
    }

    /// <summary>A new, empty sequence state of at most <see cref="MaxwellLoadOptions.MaxSequenceTokens"/> tokens.</summary>
    public MaxwellSequenceState CreateState(int capacity)
    {
        if (capacity > _options.MaxSequenceTokens)
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity, $"The model was loaded for sequences of up to {_options.MaxSequenceTokens} tokens.");
        return new MaxwellSequenceState(Config, capacity);
    }

    /// <summary>Appends <paramref name="tokenIds"/> to <paramref name="state"/> and returns the logits, <c>[rows, vocab]</c>: the last row only, or
    /// every row when <paramref name="allRows"/>.</summary>
    public float[] Forward(ReadOnlySpan<int> tokenIds, MaxwellSequenceState state, bool allRows = false)
    {
        int rows = allRows ? tokenIds.Length : 1;
        return Logits(ForwardHidden(tokenIds, state, allRows), rows);
    }

    /// <summary>Appends <paramref name="tokenIds"/> and returns the final normed hidden rows, <c>[rows, hidden]</c>: the last only, or every row.</summary>
    public float[] ForwardHidden(ReadOnlySpan<int> tokenIds, MaxwellSequenceState state, bool allRows = false)
    {
        float[] stream = ForwardStream(tokenIds, state);
        int tokens = tokenIds.Length, rows = allRows ? tokens : 1, flat = Config.HcMult * Config.HiddenSize;
        return ReadOut(stream.AsSpan((tokens - rows) * flat, rows * flat));
    }

    /// <summary>Appends <paramref name="tokenIds"/> and returns the residual stream after the last layer, before the head read-out,
    /// <c>[tokens, hc, hidden]</c>. <paramref name="afterLayer"/> sees the stream after each layer, for parity checks.</summary>
    public float[] ForwardStream(ReadOnlySpan<int> tokenIds, MaxwellSequenceState state, Action<int, float[]>? afterLayer = null)
    {
        ArgumentNullException.ThrowIfNull(state);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int tokens = tokenIds.Length, startPos = state.Length;
        RequireAppendable(tokenIds, state, startPos);
        float[] stream = Embed(tokenIds);
        bool completed = false;
        try
        {
            for (int i = 0; i < _layers.Length; i++)
            {
                MaxwellLayer layer = _layers[i] ?? new MaxwellLayer(_backend, Config, i, _source, _rope);
                try
                {
                    layer.Forward(stream, tokens, startPos, state);
                    afterLayer?.Invoke(i, stream);
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
            // Layers that ran already extended their caches, so Length no longer matches them.
            if (!completed) state.MarkFaulted();
        }
        state.Length = startPos + tokens;
        return stream;
    }

    /// <summary>The head read-out and final RMS norm of residual-stream rows from <see cref="ForwardStream"/>, <c>[rows, hidden]</c>.</summary>
    public float[] ReadOut(ReadOnlySpan<float> stream)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int flat = Config.HcMult * Config.HiddenSize;
        if (stream.Length < flat || stream.Length % flat != 0)
            throw new ArgumentException($"stream must hold whole residual-stream rows of {flat} values.", nameof(stream));
        int rows = stream.Length / flat;
        float[] hidden = new float[rows * Config.HiddenSize];
        _head.Head(stream, rows, _headFn, _headScale, _headBase, _finalNorm, hidden);
        return hidden;
    }

    /// <summary>The LM head over <paramref name="rows"/> hidden rows from <see cref="ForwardHidden"/>, <c>[rows, vocab]</c>.</summary>
    public float[] Logits(ReadOnlySpan<float> hidden, int rows)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int d = Config.HiddenSize;
        if (rows < 1 || hidden.Length != rows * d) throw new ArgumentException($"hidden must hold {rows} rows of {d} values.", nameof(hidden));
        using Tensor hiddenT = HostMath.Tensor(hidden, rows, d);
        using Tensor logitsT = new(new TensorShape(rows, Config.VocabSize), DType.F32);
        _backend.Linear(logitsT, hiddenT, _lmHead!, null);
        return logitsT.AsReadOnlySpan<float>().ToArray();
    }

    // Each embedding row repeated hc times: the first layer's broadcast of the single stream.
    private float[] Embed(ReadOnlySpan<int> tokenIds)
    {
        int d = Config.HiddenSize, copies = Config.HcMult;
        float[] stream = new float[tokenIds.Length * copies * d];
        for (int t = 0; t < tokenIds.Length; t++)
        {
            using Tensor row = _embedding!.SliceRows(tokenIds[t], 1);
            if (row.DType == DType.F32)
            {
                CopyRow(row.AsReadOnlySpan<float>(), stream, t * copies, copies, d);
                continue;
            }
            using Tensor f32 = row.CastTo(DType.F32);
            CopyRow(f32.AsReadOnlySpan<float>(), stream, t * copies, copies, d);
        }
        return stream;
    }

    private static void CopyRow(ReadOnlySpan<float> values, float[] stream, int firstRow, int copies, int d)
    {
        for (int c = 0; c < copies; c++) values.CopyTo(stream.AsSpan((firstRow + c) * d, d));
    }

    private void RequireAppendable(ReadOnlySpan<int> tokenIds, MaxwellSequenceState state, int startPos)
    {
        if (state.Faulted) throw new InvalidOperationException("An earlier forward on this state failed partway through; call Reset() before reusing it.");
        if (tokenIds.Length < 1) throw new ArgumentException("A forward needs at least one token.", nameof(tokenIds));
        if (state.Capacity > _options.MaxSequenceTokens)
            throw new InvalidOperationException($"The sequence state holds {state.Capacity} tokens; the model was loaded for {_options.MaxSequenceTokens}.");
        if (startPos + tokenIds.Length > state.Capacity)
            throw new InvalidOperationException($"The sequence state holds {state.Capacity} tokens; {startPos + tokenIds.Length} were asked for.");
        foreach (int id in tokenIds)
            if ((uint)id >= (uint)Config.VocabSize) throw new ArgumentOutOfRangeException(nameof(tokenIds), id, "Token id is outside the vocabulary.");
    }

    private static void RequireMatrix(Tensor t, int rows, int cols, string name)
    {
        if (t.Shape.Rank != 2 || t.Shape[0] != rows || t.Shape[1] != cols)
            throw new InvalidDataException($"'{name}' is {t.Shape}; the config needs [{rows}, {cols}].");
    }

    private void DisposeWeights()
    {
        foreach (MaxwellLayer? layer in _layers) layer?.Dispose();
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

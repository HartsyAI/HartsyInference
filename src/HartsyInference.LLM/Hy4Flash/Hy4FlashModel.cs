using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>The Hy4 Flash text model as a host reference (transformers <c>HYV4FlashForConditionalGeneration</c>, text path): token embedding copied into <c>hc</c> residual streams, the 42
/// decoder layers with the Engram memories added before their layers, the hyper-connection head, the final RMS norm and an F32 LM head. Every projection goes through <see cref="IBackend"/>.</summary>
/// <remarks>Image spans take their features from the vision tower through the <c>images</c> argument of <see cref="Forward"/>; without it the placeholder id is an ordinary token. Each state owns its M-RoPE rows; the layers read the
/// calling state's rows, so <see cref="Forward"/> runs one call at a time. Epsilon of the Engram norms is a constructor argument (see <see cref="Hy4FlashEngramLayer"/>).</remarks>
public sealed class Hy4FlashModel : IDisposable
{
    private readonly IBackend _backend;
    private readonly Hy4FlashConfig _cfg;
    private readonly List<Tensor> _owned = [];
    private readonly Hy4FlashEngramRows _embedding;
    private readonly Hy4FlashLayer[] _layers;
    private readonly Hy4FlashEngramLayer?[] _engram;
    private readonly Hy4FlashHyperConnection _head;
    private readonly Tensor _finalNorm;
    private readonly Tensor _lmHead;
    private readonly Hy4FlashBoundRope _indexedRope;
    private readonly Hy4FlashBoundRope _slidingRope;
    private readonly object _gate = new();
    private int _disposed;

    /// <param name="maxPositions">Positions the rotary tables and caches cover.</param>
    /// <param name="engramNormEps">Epsilon of every Engram norm.</param>
    public Hy4FlashModel(IBackend backend, Hy4FlashConfig config, IModelWeightSource source, int maxPositions, float engramNormEps = 1.1920929e-7f)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(source);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxPositions, 1);
        _backend = backend;
        _cfg = config;
        MaxPositions = maxPositions;
        _layers = new Hy4FlashLayer[config.NumHiddenLayers];
        _engram = new Hy4FlashEngramLayer?[config.NumHiddenLayers];
        _indexedRope = new Hy4FlashBoundRope(config.QkRopeHeadDim / 2, maxPositions);
        _slidingRope = new Hy4FlashBoundRope(config.QkRopeHeadDim / 2, maxPositions);
        try
        {
            _embedding = new Hy4FlashEngramRows(Own(source.Read("model.embed_tokens.weight")));
            _finalNorm = F32(source, "model.norm.weight");
            _lmHead = Own(source.Read("lm_head.weight"));
            using Tensor scale = source.Read("model.hc_head.hc_head_scale"), bias = source.Read("model.hc_head.hc_head_base");
            _head = new Hy4FlashHyperConnection(backend, config.HcMult, config.HiddenSize, (float)config.HcEps, (float)config.HcMagnitude, (float)config.RmsNormEps,
                F32(source, "model.hc_head.hc_head_fn"), bias.AsReadOnlySpan<float>().ToArray(), scale.AsReadOnlySpan<float>().ToArray(), head: true);
            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i] = new Hy4FlashLayer(backend, config, i, source, config.LayerKinds[i] == Hy4FlashLayerKind.Indexed ? _indexedRope : _slidingRope);
                if (config.HasEngram(i)) _engram[i] = BuildEngram(i, source, engramNormEps);
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The model config.</summary>
    public Hy4FlashConfig Config => _cfg;

    /// <summary>Positions the caches cover.</summary>
    public int MaxPositions { get; }

    /// <summary>Receives named stage arrays (<c>embed</c>, <c>layer_{i}_out</c>, <c>engram_{i}_out</c>, <c>layer_{i}_attn_hc_post</c> and the other layer stages, <c>hc_head_out</c>, <c>final_norm_out</c>, <c>logits</c>) when set.</summary>
    public Action<string, float[]>? Trace
    {
        get => _trace;
        set
        {
            _trace = value;
            for (int i = 0; i < _layers.Length; i++)
            {
                int layer = i;
                _layers[i].Trace = value is null ? null : (name, data) => value($"layer_{layer}_{name}", data);
            }
        }
    }

    private Action<string, float[]>? _trace;

    /// <summary>A fresh sequence state.</summary>
    public Hy4FlashSequenceState CreateState() => new([.. _layers.Select(l => l.CreateState(MaxPositions))],
        new Hy4FlashMropeRows(_cfg.QkRopeHeadDim / 2, MaxPositions, _cfg.IndexedRopeTheta, _cfg.MropeSection),
        new Hy4FlashMropeRows(_cfg.QkRopeHeadDim / 2, MaxPositions, _cfg.SlidingRopeTheta, _cfg.MropeSection)) { Owner = this };

    /// <summary>Runs <paramref name="tokenIds"/> at the positions after the state's cache and returns the logits of the last token, or of every token with <paramref name="allRows"/>.
    /// With <paramref name="images"/> given, each run of image placeholder ids takes its next entry (without it the placeholder id is an ordinary token, as in the reference): its feature rows replace the embeddings and its tokens get the frame positions (time constant,
    /// height by merged row, width by merged column, all offset by the position the run starts at); a run must lie inside one call.</summary>
    public float[] Forward(ReadOnlySpan<int> tokenIds, Hy4FlashSequenceState state, bool allRows = false, IReadOnlyList<Hy4FlashImage>? images = null)
    {
        int rows = allRows ? tokenIds.Length : 1;
        return Logits(ForwardHidden(tokenIds, state, allRows, images), rows);
    }

    /// <summary>Runs the tokens and returns the final normed hidden rows, <c>[rows, HiddenSize]</c> (the last row only unless <paramref name="allRows"/>).</summary>
    public float[] ForwardHidden(ReadOnlySpan<int> tokenIds, Hy4FlashSequenceState state, bool allRows = false, IReadOnlyList<Hy4FlashImage>? images = null)
    {
        lock (_gate) return ForwardLocked(tokenIds, state, allRows, images);
    }

    private float[] ForwardLocked(ReadOnlySpan<int> tokenIds, Hy4FlashSequenceState state, bool allRows, IReadOnlyList<Hy4FlashImage>? images)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (!ReferenceEquals(state.Owner, this)) throw new ArgumentException("The state was created by another model; use this model's CreateState.", nameof(state));
        int tokens = tokenIds.Length, h = _cfg.HiddenSize, hc = _cfg.HcMult, start = state.Length;
        if (tokens < 1) throw new ArgumentException("Forward needs at least one token.", nameof(tokenIds));
        if (start + tokens > MaxPositions) throw new InvalidOperationException($"The state holds {MaxPositions} positions; {start + tokens} were asked for.");

        // The state's token history is only extended once every layer has run, so a refused or failed call leaves Ids and Length as they were.
        int[] ids = new int[start + tokens];
        state.Ids.CopyTo(ids, 0);
        tokenIds.CopyTo(ids.AsSpan(start));

        // The layers read this state's rotary rows; the positions assigned below are written into them.
        _indexedRope.Bind(state.IndexedRope);
        _slidingRope.Bind(state.SlidingRope);
        float[] embedded = new float[tokens * h];
        for (int t = 0; t < tokens; t++) _embedding.Read(tokenIds[t], embedded.AsSpan(t * h, h));
        _trace?.Invoke("embed", embedded.ToArray());
        AssignPositions(tokenIds, start, state, images, embedded, out int nextPosition, out int imagesUsed);
        float[] stream = new float[tokens * hc * h];
        for (int t = 0; t < tokens; t++)
            for (int i = 0; i < hc; i++) embedded.AsSpan(t * h, h).CopyTo(stream.AsSpan((t * hc + i) * h, h));

        for (int l = 0; l < _layers.Length; l++)
        {
            if (_engram[l] is { } engram)
            {
                (int[] ngram, bool[] mask) = Hy4FlashEngramLayer.BuildInputs(_cfg, ids, start, tokens);
                float[] residual = engram.Residual(stream, tokens, ngram, mask);
                _trace?.Invoke($"engram_{l}_out", residual);
                for (int i = 0; i < stream.Length; i++) stream[i] += residual[i];
            }
            _layers[l].Forward(stream, tokens, start, state.Layers[l]);
            _trace?.Invoke($"layer_{l}_out", stream.ToArray());
        }
        state.Ids.AddRange(tokenIds.ToArray());
        state.Length = start + tokens;
        state.NextPosition = nextPosition;
        state.ImagesUsed = imagesUsed;
        state.EndsInImageRun = images is not null && tokenIds[^1] == _cfg.ImageTokenId;

        _head.Gates(stream, tokens, out float[] pre, out _);
        float[] collapsed = _head.Collapse(stream, pre, tokens);
        _trace?.Invoke("hc_head_out", collapsed.ToArray());
        HostOps.RmsNorm(_backend, collapsed, h, _finalNorm, (float)_cfg.RmsNormEps);
        _trace?.Invoke("final_norm_out", collapsed.ToArray());
        return allRows ? collapsed : collapsed.AsSpan((tokens - 1) * h, h).ToArray();
    }

    /// <summary>Projects final hidden rows through the LM head: F32 logits, <c>[rows, VocabSize]</c>.</summary>
    public float[] Logits(ReadOnlySpan<float> hidden, int rows)
    {
        if (rows < 1 || hidden.Length != (long)rows * _cfg.HiddenSize) throw new ArgumentException($"hidden must hold {rows} rows of {_cfg.HiddenSize} values.", nameof(hidden));
        float[] logits = HostOps.Project(_backend, hidden, rows, _cfg.HiddenSize, _lmHead, _cfg.VocabSize);
        _trace?.Invoke("logits", logits.ToArray());
        return logits;
    }

    // Text tokens take consecutive positions; an image run takes its frame positions and its feature rows, and the next text token continues one past the largest.
    // Returns the next text position and the images used; the state keeps them only once the layers have run.
    private void AssignPositions(ReadOnlySpan<int> ids, int start, Hy4FlashSequenceState state, IReadOnlyList<Hy4FlashImage>? images, float[] embedded, out int next, out int used)
    {
        int h = _cfg.HiddenSize, m = _cfg.Vision.SpatialMergeSize;
        next = state.NextPosition;
        used = state.ImagesUsed;
        if (images is not null && ids.Length > 0 && ids[0] == _cfg.ImageTokenId && state.EndsInImageRun)
            throw new ArgumentException("An image placeholder run is split across two Forward calls; a run must lie inside one call.", nameof(ids));
        for (int t = 0; t < ids.Length;)
        {
            if (images is null || ids[t] != _cfg.ImageTokenId)
            {
                SetSlot(state, start + t, next, next, next);
                next++;
                t++;
                continue;
            }
            int end = t;
            while (end < ids.Length && ids[end] == _cfg.ImageTokenId) end++;
            if (used >= images.Count) throw new ArgumentException("An image placeholder run has no image to take.", nameof(images));
            Hy4FlashImage image = images[used];
            if (image.GridHeight < 1 || image.GridWidth < 1 || image.GridHeight % m != 0 || image.GridWidth % m != 0)
                throw new ArgumentException($"An image's patch grid must be a positive multiple of the merge size {m} on each side.", nameof(images));
            int rows = image.GridHeight / m, cols = image.GridWidth / m + 1;
            if (end - t != rows * cols || image.Features.Length != rows * cols * h)
                throw new ArgumentException($"The image yields {rows * cols} tokens but its placeholder run holds {end - t}.", nameof(images));
            image.Features.CopyTo(embedded.AsSpan(t * h, rows * cols * h));
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    SetSlot(state, start + t + r * cols + c, next, next + r, next + c);
            next += Math.Max(rows - 1, cols - 1) + 1;
            used++;
            t = end;
        }
    }

    // Writes one slot's rotary rows into the state, which keeps them for every later call.
    private static void SetSlot(Hy4FlashSequenceState state, int slot, int time, int height, int width)
    {
        state.IndexedRope.SetPosition(slot, time, height, width);
        state.SlidingRope.SetPosition(slot, time, height, width);
    }

    private Hy4FlashEngramLayer BuildEngram(int layerIndex, IModelWeightSource source, float eps)
    {
        string e = $"{Hy4FlashWeightMap.Layer(layerIndex)}.engram_layer.";
        int hc = _cfg.HcMult;
        Hy4FlashEngramWeights weights = new()
        {
            PerHeadNorms = [.. Enumerable.Range(0, _cfg.EngramHashHeads).Select(i => F32(source, $"{e}lookup_per_head_norms.{i}.weight"))],
            ValueProj = Own(source.Read(e + "value_proj.weight")),
            ValueBias = F32(source, e + "value_proj.bias"),
            KeyProjs = [.. Enumerable.Range(0, hc).Select(i => Own(source.Read($"{e}key_projs.{i}.weight")))],
            KeyBiases = [.. Enumerable.Range(0, hc).Select(i => F32(source, $"{e}key_projs.{i}.bias"))],
            NormKey = [.. Enumerable.Range(0, hc).Select(i => F32(source, $"{e}norm_key.{i}.weight"))],
            NormQuery = [.. Enumerable.Range(0, hc).Select(i => F32(source, $"{e}norm_query.{i}.weight"))],
            UnifiedPreGamma = F32(source, e + "unified_pre_gamma"),
            PerHcScales = [.. Enumerable.Range(0, hc).Select(i => F32(source, $"{e}unified_per_hc_scales.{i}"))],
        };
        int[] lookup;
        using (Tensor table = source.Read(e + "lookup_table")) lookup = table.AsReadOnlySpan<int>().ToArray();
        return new Hy4FlashEngramLayer(_backend, _cfg, layerIndex + 1, weights, new Hy4FlashEngramRows(Own(source.Read(e + "multi_head_embedding.embedding.weight"))), lookup, eps);
    }

    private Tensor F32(IModelWeightSource source, string name) => Own(WeightReads.ReadF32(source, name));

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Hy4FlashLayer? layer in _layers) layer?.Dispose();
        foreach (Hy4FlashEngramLayer? engram in _engram) engram?.Dispose();
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }
}

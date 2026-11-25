using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>The Hy4 Flash text model as a host reference (transformers <c>HYV4FlashForConditionalGeneration</c>, text path): token embedding copied into <c>hc</c> residual streams, the 42
/// decoder layers with the Engram memories added before their layers, the hyper-connection head, the final RMS norm and an F32 LM head. Every projection goes through <see cref="IBackend"/>.</summary>
/// <remarks>Text-only: image spans need the vision tower and positions passed separately. Epsilon of the Engram norms is a constructor argument (see <see cref="Hy4FlashEngramLayer"/>).</remarks>
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
    private int _disposed;

    /// <param name="maxPositions">Positions the rotary tables and caches cover.</param>
    /// <param name="engramNormEps">Epsilon of every Engram norm.</param>
    public Hy4FlashModel(IBackend backend, Hy4FlashConfig config, IHy4FlashWeightSource source, int maxPositions, float engramNormEps = 1.1920929e-7f)
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
        try
        {
            _embedding = new Hy4FlashEngramRows(Own(source.Read("model.embed_tokens.weight")));
            _finalNorm = F32(source, "model.norm.weight");
            _lmHead = Own(source.Read("lm_head.weight"));
            using Tensor scale = source.Read("model.hc_head.hc_head_scale"), bias = source.Read("model.hc_head.hc_head_base");
            _head = new Hy4FlashHyperConnection(backend, config.HcMult, config.HiddenSize, (float)config.HcEps, (float)config.HcMagnitude, (float)config.RmsNormEps,
                F32(source, "model.hc_head.hc_head_fn"), bias.AsReadOnlySpan<float>().ToArray(), scale.AsReadOnlySpan<float>().ToArray(), head: true);
            Hy4FlashRopeRows indexedRope = new(config.QkRopeHeadDim / 2, maxPositions, config.IndexedRopeTheta);
            Hy4FlashRopeRows slidingRope = new(config.QkRopeHeadDim / 2, maxPositions, config.SlidingRopeTheta);
            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i] = new Hy4FlashLayer(backend, config, i, source, config.LayerKinds[i] == Hy4FlashLayerKind.Indexed ? indexedRope : slidingRope);
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
    public Hy4FlashSequenceState CreateState() => new([.. _layers.Select(l => l.CreateState(MaxPositions))]);

    /// <summary>Runs <paramref name="tokenIds"/> at the positions after the state's cache and returns the logits of the last token, or of every token with <paramref name="allRows"/>.</summary>
    public float[] Forward(ReadOnlySpan<int> tokenIds, Hy4FlashSequenceState state, bool allRows = false)
    {
        ArgumentNullException.ThrowIfNull(state);
        int tokens = tokenIds.Length, h = _cfg.HiddenSize, hc = _cfg.HcMult, start = state.Length;
        if (tokens < 1) throw new ArgumentException("Forward needs at least one token.", nameof(tokenIds));
        if (start + tokens > MaxPositions) throw new InvalidOperationException($"The state holds {MaxPositions} positions; {start + tokens} were asked for.");

        // The state's token history is only extended once every layer has run, so a refused or failed call leaves Ids and Length as they were.
        int[] ids = new int[start + tokens];
        state.Ids.CopyTo(ids, 0);
        tokenIds.CopyTo(ids.AsSpan(start));

        float[] embedded = new float[tokens * h];
        for (int t = 0; t < tokens; t++) _embedding.Read(tokenIds[t], embedded.AsSpan(t * h, h));
        _trace?.Invoke("embed", embedded.ToArray());
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

        _head.Gates(stream, tokens, out float[] pre, out _);
        float[] collapsed = _head.Collapse(stream, pre, tokens);
        _trace?.Invoke("hc_head_out", collapsed.ToArray());
        HostOps.RmsNorm(_backend, collapsed, h, _finalNorm, (float)_cfg.RmsNormEps);
        _trace?.Invoke("final_norm_out", collapsed.ToArray());
        int rows = allRows ? tokens : 1;
        float[] last = allRows ? collapsed : collapsed.AsSpan((tokens - 1) * h, h).ToArray();
        float[] logits = HostOps.Project(_backend, last, rows, h, _lmHead, _cfg.VocabSize);
        _trace?.Invoke("logits", logits);
        return logits;
    }

    private Hy4FlashEngramLayer BuildEngram(int layerIndex, IHy4FlashWeightSource source, float eps)
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

    private Tensor F32(IHy4FlashWeightSource source, string name)
    {
        Tensor t = source.Read(name);
        if (t.DType == DType.F32) return Own(t);
        using (t) return Own(ProjectionOps.EnsureF32(t));
    }

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

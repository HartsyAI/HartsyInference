using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.TextEncoders;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Gemma 4 decoder transformer run as a conditioning text encoder (LTX-2.5's Gemma-4-12B). Single
/// forward pass, no KV cache, no sampling. Exposes the same all-hidden-states surface the LTX-2 pipeline already
/// consumes from <see cref="LlamaStyleEncoder"/>.</summary>
///
/// <remarks>Gemma 4 differs from every Llama-family encoder in ways that cannot be expressed as a
/// <see cref="LlamaStyleEncoderConfig"/>: layers alternate 5 sliding + 1 global, and the two kinds have different
/// head dimensions (256 / 512), different KV-head counts (8 / 1) and different RoPE bases (1e4 / 1e6). Global
/// layers use partial rotary (only the first quarter of the head dimension rotates) and carry no <c>v_proj</c> —
/// V is the raw <c>k_proj</c> output taken before <c>k_norm</c> and before RoPE. Attention is scaled by 1.0, not
/// <c>1/sqrt(head_dim)</c>. V additionally passes through a weightless RMS norm on <b>both</b> layer kinds, each
/// block ends with a scalar <c>layer_scalar</c> multiply, and the four sandwich norms store their scales directly
/// (verified against the real checkpoint's bf16 bytes — this is <b>not</b> Gemma 3's <c>1 + w</c> convention).</remarks>
public sealed unsafe class Gemma4TextEncoder : IDisposable
{
    private readonly Gemma4TextEncoderConfig _config;
    private readonly Gemma4Block[] _blocks;

    private Tensor? _embedWeight;
    private Tensor? _finalNormWeight;

    // Weightless RMS scale (all ones) for the V norm, one per distinct head dimension in use.
    private readonly Dictionary<int, Tensor> _onesByHeadDim = [];

    // Half-width cos/sin tables per layer kind, lazily grown to the longest prompt seen. Keyed on the kind, not
    // the head dimension: the two kinds may share a head dimension while using different RoPE bases.
    private readonly Dictionary<bool, RopeTable> _ropeTables = [];

    private int _disposed;

    public Gemma4TextEncoder(Gemma4TextEncoderConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        Validate(config);
        _config = config;
        _onesByHeadDim[config.HeadDim] = Ones(config.HeadDim);
        if (config.GlobalHeadDim != config.HeadDim) _onesByHeadDim[config.GlobalHeadDim] = Ones(config.GlobalHeadDim);
        _blocks = new Gemma4Block[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new Gemma4Block(config, i, this);
    }

    /// <summary>Number of transformer blocks. <see cref="EncodeMultiLayer"/> accepts state indices <c>0 … NumLayers</c>.</summary>
    public int NumLayers => _config.NumLayers;

    /// <summary>Loads all weights from a HuggingFace-style key dict. Only the language-tower keys are read, so the
    /// checkpoint's unused <c>vision_model.*</c> / <c>audio_projector.*</c> / <c>multi_modal_projector.*</c> towers
    /// and the <c>tokenizer_json</c> / <c>hf_asset__*</c> sidecars are skipped by construction.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ThrowIfDisposed();
        weights = TextEncoderQuantNormalizer.Normalize(weights);

        string prefix = weights.ContainsKey("model.embed_tokens.weight") ? "model." : "";
        _embedWeight = CastToF32IfNeeded(weights[$"{prefix}embed_tokens.weight"]);
        _finalNormWeight = CastToF32IfNeeded(weights[$"{prefix}norm.weight"]);

        for (int i = 0; i < _config.NumLayers; i++)
            _blocks[i].LoadWeights(weights, $"{prefix}layers.{i}");

        Logs.Verbose($"Gemma4TextEncoder loaded: {_config.NumLayers} layers, hidden={_config.HiddenSize}, " +
                     $"sliding head_dim={_config.HeadDim}/kv={_config.NumKvHeads}, " +
                     $"global head_dim={_config.GlobalHeadDim}/kv={_config.NumGlobalKvHeads}");
    }

    /// <summary>Enumerates all weight tensors, used by GPU-preload paths.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedWeight is not null) yield return _embedWeight;
        if (_finalNormWeight is not null) yield return _finalNormWeight;
        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights())
                yield return w;
    }

    /// <summary>Encodes a batch and returns <c>last_hidden_state</c> shaped <c>[B, seq, hidden]</c> (final norm applied).</summary>
    public Tensor Encode(IBackend backend, int[][] tokenIds)
    {
        Tensor packed = EncodeMultiLayer(backend, tokenIds, [_config.NumLayers]);
        return packed;
    }

    /// <summary>Captures several HF-indexed hidden states in one forward pass and packs them into
    /// <c>[B, seq, K·hidden]</c>. State <c>0</c> is the scaled embedding output, state <c>i</c> is the output of
    /// block <c>i-1</c>, and state <c>NumLayers</c> additionally passes through <c>model.norm</c> when
    /// <see cref="Gemma4TextEncoderConfig.ApplyFinalNormToLastState"/> is set. Indices must be strictly ascending.
    /// The default layout is layer-outer (feature index <c>layer·hidden + channel</c>);
    /// <paramref name="interleavedLayout"/> transposes each token to hidden-major
    /// (<c>channel·K + layer</c>).</summary>
    public Tensor EncodeMultiLayer(IBackend backend, int[][] tokenIds, int[] layerIndices, bool interleavedLayout = false)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(backend);
        if (layerIndices is null || layerIndices.Length == 0)
            throw new ArgumentException("Must request at least one layer.", nameof(layerIndices));
        for (int i = 0; i < layerIndices.Length; i++)
        {
            if (layerIndices[i] < 0 || layerIndices[i] > _config.NumLayers)
                throw new ArgumentOutOfRangeException(nameof(layerIndices),
                    $"Layer index {layerIndices[i]} out of range [0, {_config.NumLayers}].");
            if (i > 0 && layerIndices[i] <= layerIndices[i - 1])
                throw new ArgumentException("Layer indices must be strictly ascending.", nameof(layerIndices));
        }
        if (_embedWeight is null || _finalNormWeight is null)
            throw new InvalidOperationException("Gemma4TextEncoder.LoadWeights must run before encoding.");

        (int batch, int seqLen) = ValidateTokenBatch(tokenIds);
        if (seqLen > _config.MaxPositionEmbeddings)
            throw new InvalidOperationException(
                $"Prompt length {seqLen} exceeds MaxPositionEmbeddings ({_config.MaxPositionEmbeddings}).");

        int hiddenSize = _config.HiddenSize;
        int stateCount = layerIndices.Length;
        bool[] requested = new bool[_config.NumLayers + 1];
        foreach (int index in layerIndices) requested[index] = true;

        List<Tensor> captures = new(stateCount);
        Tensor? hidden = null;
        bool hiddenIsCaptured = false;
        using ForwardConstants constants = PrepareForwardConstants(backend, batch, seqLen);
        try
        {
            hidden = EmbeddingLookup(backend, tokenIds, batch, seqLen);
            for (int i = 0; i < _config.NumLayers; i++)
            {
                if (requested[i])
                {
                    captures.Add(hidden);
                    hiddenIsCaptured = true;
                }
                Tensor next = _blocks[i].Forward(backend, hidden, constants, seqLen);
                if (!hiddenIsCaptured) DisposeBestEffort(hidden, "previous hidden activation");
                hiddenIsCaptured = false;
                hidden = next;
            }

            if (requested[_config.NumLayers])
            {
                if (_config.ApplyFinalNormToLastState)
                {
                    Tensor normed = new(new TensorShape(batch, seqLen, hiddenSize), DType.F32);
                    backend.RmsNorm(normed, hidden, _finalNormWeight, _config.RmsNormEps);
                    captures.Add(normed);
                }
                else
                {
                    captures.Add(hidden);
                    hiddenIsCaptured = true;
                }
            }

            if (captures.Count != stateCount)
                throw new InvalidOperationException($"Captured {captures.Count} of {stateCount} requested Gemma 4 hidden states.");

            if (stateCount == 1)
            {
                Tensor only = captures[0];
                captures.Clear();
                if (!ReferenceEquals(only, hidden)) DisposeBestEffort(hidden, "hidden activation after single capture");
                hidden = null;
                hiddenIsCaptured = false;
                return only;
            }

            TensorShape outputShape = new(batch, seqLen, checked(stateCount * hiddenSize));
            Tensor layerMajor = new(outputShape, DType.F32);
            try
            {
                backend.Concat(layerMajor, captures.ToArray(), dim: 2);
                if (!interleavedLayout) return layerMajor;

                Tensor output = new(outputShape, DType.F32);
                try
                {
                    backend.Transpose2D(output, layerMajor, stateCount, hiddenSize);
                }
                catch
                {
                    DisposeBestEffort(output, "interleaved multi-layer output");
                    throw;
                }
                DisposeBestEffort(layerMajor, "layer-major staging output");
                return output;
            }
            catch
            {
                DisposeBestEffort(layerMajor, "layer-major output after pack failure");
                throw;
            }
        }
        finally
        {
            if (!hiddenIsCaptured) DisposeBestEffort(hidden, "trailing hidden activation");
            foreach (Tensor capture in captures)
                DisposeBestEffort(capture, "captured hidden activation");
        }
    }

    /// <summary>Per-request causal / sliding masks plus one full-width RoPE table per layer kind.</summary>
    internal sealed class ForwardConstants : IDisposable
    {
        private readonly List<Tensor> _owned;
        private int _disposed;

        public ForwardConstants(Tensor causalMask, Tensor? slidingMask,
            Tensor slidingCos, Tensor slidingSin, Tensor globalCos, Tensor globalSin)
        {
            CausalMask = causalMask;
            SlidingMask = slidingMask;
            SlidingCos = slidingCos;
            SlidingSin = slidingSin;
            GlobalCos = globalCos;
            GlobalSin = globalSin;
            _owned = [causalMask, slidingCos, slidingSin, globalCos, globalSin];
            if (slidingMask is not null) _owned.Add(slidingMask);
        }

        public Tensor CausalMask { get; }

        /// <summary>Causal mask further restricted to the sliding window; null when the sequence is no longer
        /// than the window, in which case the plain causal mask already is the sliding mask.</summary>
        public Tensor? SlidingMask { get; }

        public Tensor SlidingCos { get; }
        public Tensor SlidingSin { get; }
        public Tensor GlobalCos { get; }
        public Tensor GlobalSin { get; }

        public Tensor MaskFor(bool global) => global ? CausalMask : SlidingMask ?? CausalMask;

        public (Tensor Cos, Tensor Sin) RopeFor(bool global) => global ? (GlobalCos, GlobalSin) : (SlidingCos, SlidingSin);

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            foreach (Tensor tensor in _owned)
            {
                try { tensor.Dispose(); }
                catch (Exception error) { Logs.Error("Failed to release a Gemma 4 encoder request constant.", error); }
            }
        }
    }

    /// <summary>Shared all-ones RMS scale for the weightless V norm, one per head dimension.</summary>
    internal Tensor OnesFor(int headDim) => _onesByHeadDim[headDim];

    private static Tensor Ones(int length)
    {
        Tensor ones = new(new TensorShape(length), DType.F32);
        float* p = (float*)ones.DataPointer;
        for (int i = 0; i < length; i++) p[i] = 1f;
        return ones;
    }

    private ForwardConstants PrepareForwardConstants(IBackend backend, int batch, int seqLen)
    {
        Tensor? causal = null, sliding = null, sCos = null, sSin = null, gCos = null, gSin = null;
        try
        {
            causal = MakeResidentCopy(backend, BuildCausalMask(seqLen, int.MaxValue));
            if (seqLen > _config.SlidingWindow)
                sliding = MakeResidentCopy(backend, BuildCausalMask(seqLen, _config.SlidingWindow));

            int slidingLayer = 0;
            int globalLayer = _config.GlobalLayerPeriod - 1;
            RopeTable slidingTable = EnsureRopeTable(slidingLayer, seqLen);
            RopeTable globalTable = EnsureRopeTable(globalLayer, seqLen);
            sCos = MakeResidentCopy(backend, BuildFullWidthTable(slidingTable.Cos, batch, seqLen, _config.HeadDim));
            sSin = MakeResidentCopy(backend, BuildFullWidthTable(slidingTable.Sin, batch, seqLen, _config.HeadDim));
            gCos = MakeResidentCopy(backend, BuildFullWidthTable(globalTable.Cos, batch, seqLen, _config.GlobalHeadDim));
            gSin = MakeResidentCopy(backend, BuildFullWidthTable(globalTable.Sin, batch, seqLen, _config.GlobalHeadDim));
            return new ForwardConstants(causal, sliding, sCos, sSin, gCos, gSin);
        }
        catch
        {
            foreach (Tensor? tensor in new[] { causal, sliding, sCos, sSin, gCos, gSin })
            {
                try { tensor?.Dispose(); }
                catch (Exception error) { Logs.Error("Failed to release a partially prepared Gemma 4 constant.", error); }
            }
            throw;
        }
    }

    private readonly record struct RopeTable(float[] Cos, float[] Sin, int BuiltForLength);

    /// <summary>Grows the half-width cos/sin table for the given layer's kind to cover <paramref name="seqLen"/>.</summary>
    private RopeTable EnsureRopeTable(int layerIndex, int seqLen)
    {
        bool global = _config.IsGlobalLayer(layerIndex);
        if (_ropeTables.TryGetValue(global, out RopeTable cached) && cached.BuiltForLength >= seqLen)
            return cached;

        int target = Math.Max(seqLen, cached.BuiltForLength);
        int rounded = 1;
        while (rounded < target) rounded <<= 1;
        target = Math.Min(rounded, _config.MaxPositionEmbeddings);

        double[] inv = _config.BuildInverseFrequencies(layerIndex);
        int half = inv.Length;
        float[] cos = new float[(long)target * half];
        float[] sin = new float[(long)target * half];
        for (int p = 0; p < target; p++)
        {
            for (int k = 0; k < half; k++)
            {
                double angle = p * inv[k];
                cos[p * half + k] = (float)Math.Cos(angle);
                sin[p * half + k] = (float)Math.Sin(angle);
            }
        }
        RopeTable table = new(cos, sin, target);
        _ropeTables[global] = table;
        return table;
    }

    /// <summary>Expands a half-width table <c>[S, D/2]</c> to the backend RoPE contract <c>[B, S, D]</c>,
    /// duplicating each frequency into both halves of the vector and across batch rows.</summary>
    private static Tensor BuildFullWidthTable(float[] halfTable, int batch, int seqLen, int headDim)
    {
        int half = headDim / 2;
        Tensor full = new(new TensorShape(batch, seqLen, headDim), DType.F32);
        float* output = (float*)full.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int sourceBase = s * half;
                long destinationBase = ((long)b * seqLen + s) * headDim;
                for (int i = 0; i < half; i++)
                {
                    float value = halfTable[sourceBase + i];
                    output[destinationBase + i] = value;
                    output[destinationBase + half + i] = value;
                }
            }
        }
        return full;
    }

    /// <summary>Additive attention mask: query <c>i</c> may attend key <c>j</c> when <c>i - window &lt; j &lt;= i</c>.
    /// Pass <see cref="int.MaxValue"/> for plain causal attention.</summary>
    private static Tensor BuildCausalMask(int seqLen, int window)
    {
        const float negInf = -1e30f;
        Tensor mask = new(new TensorShape(seqLen, seqLen), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int i = 0; i < seqLen; i++)
        {
            for (int j = 0; j < seqLen; j++)
            {
                bool blocked = j > i || (window != int.MaxValue && j <= i - window);
                p[(long)i * seqLen + j] = blocked ? negInf : 0f;
            }
        }
        return mask;
    }

    private static Tensor MakeResidentCopy(IBackend backend, Tensor host)
    {
        Tensor resident = new(host.Shape, host.DType);
        try
        {
            backend.SliceRowsGeneric(resident, host, rowOffset: 0);
            return resident;
        }
        catch
        {
            DisposeBestEffort(resident, "request constant after copy failure");
            throw;
        }
        finally
        {
            DisposeBestEffort(host, "host request constant");
        }
    }

    private Tensor EmbeddingLookup(IBackend backend, int[][] tokenIds, int batch, int seqLen)
    {
        int hiddenSize = _config.HiddenSize;
        int rowCount = checked(batch * seqLen);
        int[] rows = new int[rowCount];
        int rowIndex = 0;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                rows[rowIndex++] = tokenIds[b][s];

        TensorShape shape = new(batch, seqLen, hiddenSize);
        Tensor gathered = new(shape, DType.F32);
        try
        {
            // Uploading the 262k-row Gemma vocabulary purely to collect one prompt costs multiple GB; only use
            // the device gather when the table is already resident.
            if (!backend.TryGatherRowsResident(gathered, _embedWeight!, rows))
            {
                DisposeBestEffort(gathered, "unused resident-gather output");
                return EmbeddingLookupHost(tokenIds, batch, seqLen);
            }
            Tensor scaled = new(shape, DType.F32);
            try
            {
                backend.Scale(scaled, gathered, _config.EmbeddingScale);
            }
            catch
            {
                DisposeBestEffort(scaled, "scaled embedding output after failure");
                throw;
            }
            DisposeBestEffort(gathered, "unscaled gathered embedding");
            return scaled;
        }
        catch
        {
            DisposeBestEffort(gathered, "gathered embedding after failure");
            throw;
        }
    }

    private Tensor EmbeddingLookupHost(int[][] tokenIds, int batch, int seqLen)
    {
        int hiddenSize = _config.HiddenSize;
        Tensor output = new(new TensorShape(batch, seqLen, hiddenSize), DType.F32);
        float* embedPtr = (float*)_embedWeight!.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                long src = (long)tokenIds[b][s] * hiddenSize;
                long dst = ((long)b * seqLen + s) * hiddenSize;
                Buffer.MemoryCopy(embedPtr + src, outPtr + dst, hiddenSize * sizeof(float), hiddenSize * sizeof(float));
            }
        }
        float scale = _config.EmbeddingScale;
        long total = (long)batch * seqLen * hiddenSize;
        for (long i = 0; i < total; i++) outPtr[i] *= scale;
        return output;
    }

    private (int Batch, int Sequence) ValidateTokenBatch(int[][] tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        if (tokenIds.Length == 0)
            throw new ArgumentException("Token batch must contain at least one row.", nameof(tokenIds));
        if (tokenIds[0] is null || tokenIds[0].Length == 0)
            throw new ArgumentException("Token rows must be non-null and non-empty.", nameof(tokenIds));

        int seqLen = tokenIds[0].Length;
        for (int b = 0; b < tokenIds.Length; b++)
        {
            int[]? row = tokenIds[b];
            if (row is null || row.Length != seqLen)
                throw new ArgumentException(
                    $"Token batch must be rectangular with sequence length {seqLen}; row {b} has length {row?.Length ?? 0}.",
                    nameof(tokenIds));
            for (int s = 0; s < row.Length; s++)
            {
                if ((uint)row[s] >= (uint)_config.VocabSize)
                    throw new ArgumentOutOfRangeException(
                        nameof(tokenIds), row[s], $"Token id at [{b},{s}] is outside vocabulary size {_config.VocabSize}.");
            }
        }
        return (tokenIds.Length, seqLen);
    }

    private static void Validate(Gemma4TextEncoderConfig config)
    {
        if (config.HiddenSizePerLayerInput != 0)
            throw new NotSupportedException(
                $"Gemma4TextEncoder does not implement the Gemma-3n per-layer-input mechanism (hidden_size_per_layer_input={config.HiddenSizePerLayerInput}).");
        if (config.NumKvSharedLayers != 0)
            throw new NotSupportedException(
                $"Gemma4TextEncoder does not implement cross-layer KV sharing (num_kv_shared_layers={config.NumKvSharedLayers}).");
        // Period 1 would make every layer global, leaving the sliding path unreachable and its RoPE table wrong.
        if (config.GlobalLayerPeriod < 2)
            throw new ArgumentException($"GlobalLayerPeriod must be at least 2; got {config.GlobalLayerPeriod}.", nameof(config));
        if (config.NumQueryHeads <= 0 || config.NumKvHeads <= 0 || config.NumGlobalKvHeads <= 0
            || config.NumQueryHeads % config.NumKvHeads != 0 || config.NumQueryHeads % config.NumGlobalKvHeads != 0)
        {
            throw new ArgumentException(
                $"Invalid grouped-query geometry Hq={config.NumQueryHeads}, sliding Hkv={config.NumKvHeads}, global Hkv={config.NumGlobalKvHeads}.",
                nameof(config));
        }
        if (config.HeadDim <= 0 || (config.HeadDim & 1) != 0 || config.GlobalHeadDim <= 0 || (config.GlobalHeadDim & 1) != 0)
            throw new ArgumentException("Split-half RoPE requires positive even head dimensions.", nameof(config));
        if (config.SlidingWindow <= 0)
            throw new ArgumentException($"SlidingWindow must be positive; got {config.SlidingWindow}.", nameof(config));
    }

    internal static Tensor CastToF32IfNeeded(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    internal static void DisposeBestEffort(Tensor? tensor, string description)
    {
        if (tensor is null) return;
        try { tensor.Dispose(); }
        catch (Exception error) { Logs.Error($"Failed to release Gemma 4 encoder {description}.", error); }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor ones in _onesByHeadDim.Values)
            DisposeBestEffort(ones, "weightless V norm scale");
        _onesByHeadDim.Clear();
        _ropeTables.Clear();
        _embedWeight = null;
        _finalNormWeight = null;
    }

    /// <summary>One Gemma 4 transformer block. Geometry is fixed at construction from the layer index, so sliding
    /// and global blocks differ only in the numbers they were built with.</summary>
    private sealed class Gemma4Block
    {
        private readonly Gemma4TextEncoderConfig _config;
        private readonly Gemma4TextEncoder _owner;
        private readonly int _index;
        private readonly bool _isGlobal;
        private readonly int _headDim;
        private readonly int _kvHeads;
        private readonly bool _kEqualsV;

        private Tensor? _inputNorm, _postAttnNorm, _preFfnNorm, _postFfnNorm;
        private Tensor? _qProj, _kProj, _vProj, _oProj;
        private Tensor? _qHeadNorm, _kHeadNorm;
        private Tensor? _gateProj, _upProj, _downProj;
        private float _layerScalar = 1f;

        public Gemma4Block(Gemma4TextEncoderConfig config, int index, Gemma4TextEncoder owner)
        {
            _config = config;
            _owner = owner;
            _index = index;
            _isGlobal = config.IsGlobalLayer(index);
            _headDim = config.HeadDimFor(index);
            _kvHeads = config.KvHeadsFor(index);
            _kEqualsV = config.KEqualsVFor(index);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
        {
            _inputNorm = CastToF32IfNeeded(weights[$"{prefix}.input_layernorm.weight"]);
            _postAttnNorm = CastToF32IfNeeded(weights[$"{prefix}.post_attention_layernorm.weight"]);
            _preFfnNorm = CastToF32IfNeeded(weights[$"{prefix}.pre_feedforward_layernorm.weight"]);
            _postFfnNorm = CastToF32IfNeeded(weights[$"{prefix}.post_feedforward_layernorm.weight"]);

            _qProj = weights[$"{prefix}.self_attn.q_proj.weight"];
            _kProj = weights[$"{prefix}.self_attn.k_proj.weight"];
            _oProj = weights[$"{prefix}.self_attn.o_proj.weight"];
            if (!_kEqualsV) _vProj = weights[$"{prefix}.self_attn.v_proj.weight"];
            else if (weights.ContainsKey($"{prefix}.self_attn.v_proj.weight"))
                throw new InvalidOperationException(
                    $"Layer {_index} is a k_eq_v global layer but the checkpoint carries {prefix}.self_attn.v_proj.weight.");

            _qHeadNorm = CastToF32IfNeeded(weights[$"{prefix}.self_attn.q_norm.weight"]);
            _kHeadNorm = CastToF32IfNeeded(weights[$"{prefix}.self_attn.k_norm.weight"]);

            _gateProj = weights[$"{prefix}.mlp.gate_proj.weight"];
            _upProj = weights[$"{prefix}.mlp.up_proj.weight"];
            _downProj = weights[$"{prefix}.mlp.down_proj.weight"];

            // Read the per-block scalar on the host: it is one number consumed as a GEMM-free Scale, so uploading
            // a 1-element tensor to every device would be pure overhead.
            Tensor rawScalar = weights[$"{prefix}.layer_scalar"];
            if (rawScalar.DType == DType.F32)
            {
                _layerScalar = ((float*)rawScalar.DataPointer)[0];
            }
            else
            {
                using Tensor cast = rawScalar.CastTo(DType.F32);
                _layerScalar = ((float*)cast.DataPointer)[0];
            }
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all =
            [
                _inputNorm, _postAttnNorm, _preFfnNorm, _postFfnNorm,
                _qProj, _kProj, _vProj, _oProj, _qHeadNorm, _kHeadNorm,
                _gateProj, _upProj, _downProj,
            ];
            foreach (Tensor? t in all)
                if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor hidden, ForwardConstants constants, int seqLen)
        {
            int batch = (int)hidden.Shape[0];
            int hiddenSize = _config.HiddenSize;
            int queryHeads = _config.NumQueryHeads;
            int qDim = queryHeads * _headDim;
            TensorShape hShape = new(batch, seqLen, hiddenSize);
            TensorShape qMhShape = new(batch, queryHeads, seqLen, _headDim);
            TensorShape kvMhShape = new(batch, _kvHeads, seqLen, _headDim);
            (Tensor ropeCos, Tensor ropeSin) = constants.RopeFor(_isGlobal);

            Tensor preAttn = new(hShape, DType.F32);
            backend.RmsNorm(preAttn, hidden, _inputNorm!, _config.RmsNormEps);

            Tensor qFlat = new(new TensorShape(batch, seqLen, queryHeads, _headDim), DType.F32);
            Tensor kFlat = new(new TensorShape(batch, seqLen, _kvHeads, _headDim), DType.F32);
            backend.Linear(qFlat, preAttn, _qProj!, null);
            backend.Linear(kFlat, preAttn, _kProj!, null);
            Tensor? vFlat = null;
            if (!_kEqualsV)
            {
                vFlat = new Tensor(new TensorShape(batch, seqLen, _kvHeads, _headDim), DType.F32);
                backend.Linear(vFlat, preAttn, _vProj!, null);
            }
            preAttn.Dispose();

            Tensor qMh = new(qMhShape, DType.F32);
            Tensor kRaw = new(kvMhShape, DType.F32);
            backend.Permute0213(qMh, qFlat, seqLen, queryHeads, _headDim);
            backend.Permute0213(kRaw, kFlat, seqLen, _kvHeads, _headDim);
            qFlat.Dispose();
            kFlat.Dispose();
            Tensor vRaw;
            if (_kEqualsV)
            {
                // k_eq_v: V is the k_proj output BEFORE k_norm and before RoPE — kRaw is exactly that, and
                // RmsNorm below is out-of-place so it stays intact.
                vRaw = kRaw;
            }
            else
            {
                vRaw = new Tensor(kvMhShape, DType.F32);
                backend.Permute0213(vRaw, vFlat!, seqLen, _kvHeads, _headDim);
                vFlat!.Dispose();
            }

            Tensor qNormed = new(qMhShape, DType.F32);
            Tensor kNormed = new(kvMhShape, DType.F32);
            Tensor vNormed = new(kvMhShape, DType.F32);
            backend.RmsNorm(qNormed, qMh, _qHeadNorm!, _config.RmsNormEps);
            backend.RmsNorm(kNormed, kRaw, _kHeadNorm!, _config.RmsNormEps);
            // V's norm is weightless on BOTH layer kinds (the reference applies it outside the v_proj branch).
            backend.RmsNorm(vNormed, vRaw, _owner.OnesFor(_headDim), _config.RmsNormEps);
            qMh.Dispose();
            kRaw.Dispose();
            if (!ReferenceEquals(vRaw, kRaw)) vRaw.Dispose();

            backend.ApplyRopeSingleHeadMajor(qNormed, ropeCos, ropeSin, _headDim);
            backend.ApplyRopeSingleHeadMajor(kNormed, ropeCos, ropeSin, _headDim);

            Tensor kAttn = kNormed;
            Tensor vAttn = vNormed;
            if (_kvHeads != queryHeads)
            {
                kAttn = new Tensor(qMhShape, DType.F32);
                vAttn = new Tensor(qMhShape, DType.F32);
                backend.RepeatKvHeads(kAttn, kNormed, _kvHeads, queryHeads / _kvHeads);
                backend.RepeatKvHeads(vAttn, vNormed, _kvHeads, queryHeads / _kvHeads);
                kNormed.Dispose();
                vNormed.Dispose();
            }

            Tensor attnOut = new(qMhShape, DType.F32);
            backend.ScaledDotProductAttention(attnOut, qNormed, kAttn, vAttn, constants.MaskFor(_isGlobal), _config.AttentionScale);
            qNormed.Dispose();
            kAttn.Dispose();
            vAttn.Dispose();

            Tensor attnFlat = new(new TensorShape(batch, seqLen, qDim), DType.F32);
            backend.Permute0213(attnFlat, attnOut, queryHeads, seqLen, _headDim);
            attnOut.Dispose();

            Tensor attnProj = new(hShape, DType.F32);
            backend.Linear(attnProj, attnFlat, _oProj!, null);
            attnFlat.Dispose();

            Tensor postAttn = new(hShape, DType.F32);
            backend.RmsNorm(postAttn, attnProj, _postAttnNorm!, _config.RmsNormEps);
            attnProj.Dispose();

            Tensor afterAttn = new(hShape, DType.F32);
            backend.Add(afterAttn, hidden, postAttn);
            postAttn.Dispose();

            Tensor preMlp = new(hShape, DType.F32);
            backend.RmsNorm(preMlp, afterAttn, _preFfnNorm!, _config.RmsNormEps);

            TensorShape mlpShape = new(batch, seqLen, _config.IntermediateSize);
            Tensor gate = new(mlpShape, DType.F32);
            Tensor up = new(mlpShape, DType.F32);
            backend.Linear(gate, preMlp, _gateProj!, null);
            backend.Linear(up, preMlp, _upProj!, null);
            preMlp.Dispose();

            Tensor activated = new(mlpShape, DType.F32);
            backend.Gelu(activated, gate);
            gate.Dispose();
            Tensor gated = new(mlpShape, DType.F32);
            backend.Mul(gated, activated, up);
            activated.Dispose();
            up.Dispose();

            Tensor mlpOut = new(hShape, DType.F32);
            backend.Linear(mlpOut, gated, _downProj!, null);
            gated.Dispose();

            Tensor postMlp = new(hShape, DType.F32);
            backend.RmsNorm(postMlp, mlpOut, _postFfnNorm!, _config.RmsNormEps);
            mlpOut.Dispose();

            Tensor summed = new(hShape, DType.F32);
            backend.Add(summed, afterAttn, postMlp);
            afterAttn.Dispose();
            postMlp.Dispose();

            Tensor result = new(hShape, DType.F32);
            backend.Scale(result, summed, _layerScalar);
            summed.Dispose();
            return result;
        }
    }
}

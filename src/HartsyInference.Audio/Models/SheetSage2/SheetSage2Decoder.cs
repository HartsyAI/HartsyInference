using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.SheetSage2;

/// <summary>SheetSage2's score decoder: an autoregressive transformer over <see cref="ScoreTokenizer"/> ids that
/// cross-attends to the encoder's memory and writes a transcription.
///
/// <para>Post-LN throughout — see <see cref="SheetSage2Config"/>. The other two things the released decode hangs
/// on: token index <c>i</c> reads position row <c>i + 2</c>, and the decode is greedy argmax over logits masked
/// by <see cref="PromptGrammar"/>, so the mask is part of the model rather than a guard around it.</para>
///
/// <para><b>Memory contract.</b> <see cref="StartDecode"/> takes the encoder's output already through
/// <c>encoder_projection</c> — F32, either <c>[memoryTokens, dim]</c> or <c>[1, memoryTokens, dim]</c>, normally
/// the fixed <see cref="SheetSage2Config.EncoderMemoryTokens"/> because the encoder always reads a padded
/// 300-second window. Cross-attention K/V are projected from it once here, never per step.</para></summary>
public sealed unsafe class SheetSage2Decoder : IDisposable
{
    private readonly SheetSage2Config _config;
    private readonly ScoreTokenizer _tokenizer;
    private readonly SheetSage2DecoderLayer[] _layers;
    private readonly List<Tensor> _ownedCasts = [];

    private Tensor? _tokenEmbedding;
    private Tensor? _positionEmbedding;
    private Tensor? _embedNormWeight;
    private Tensor? _embedNormBias;
    private Tensor? _outputProjection;
    private bool _weightsLoaded;
    private int _disposed;

    public SheetSage2Decoder(SheetSage2Config config, ScoreTokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (config.Dim % config.NumHeads != 0)
            throw new ArgumentException($"SheetSage2 width {config.Dim} does not divide into {config.NumHeads} heads.", nameof(config));
        _config = config;
        _tokenizer = tokenizer;
        _layers = new SheetSage2DecoderLayer[config.Layers];
        for (int i = 0; i < config.Layers; i++) _layers[i] = new SheetSage2DecoderLayer(config);
    }

    /// <summary>Shape the decoder was built for.</summary>
    public SheetSage2Config Config => _config;

    /// <summary>The vocabulary the decoder writes in; its size is the output projection's width.</summary>
    public ScoreTokenizer Tokenizer => _tokenizer;

    /// <summary>Loads the decode side of the released checkpoint, whose keys are unprefixed:
    /// <c>decoder.*</c> alongside <c>token_embedding.weight</c> and <c>output_projection.weight</c>.</summary>
    /// <param name="prefix">Model-root prefix, empty for the released single file.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(weights);
        ThrowIfDisposed();
        string root = prefix.Length == 0 ? string.Empty : $"{prefix}.";

        // The vocabulary is derived, not written down, so a checkpoint the tokenizer disagrees with has to fail
        // here: every id past the shorter of the two would otherwise decode as a different field.
        int dim = _config.Dim;
        _tokenEmbedding = SheetSage2Weights.Take(weights, $"{root}token_embedding.weight", _ownedCasts, _tokenizer.TokenCount, dim);
        _positionEmbedding = SheetSage2Weights.Take(weights, $"{root}decoder.embed_positions.weight", _ownedCasts, _config.PositionRows, dim);
        _embedNormWeight = SheetSage2Weights.Take(weights, $"{root}decoder.layernorm_embedding.weight", _ownedCasts, dim);
        _embedNormBias = SheetSage2Weights.Take(weights, $"{root}decoder.layernorm_embedding.bias", _ownedCasts, dim);
        _outputProjection = SheetSage2Weights.Take(weights, $"{root}output_projection.weight", _ownedCasts, _tokenizer.TokenCount, dim);

        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"{root}decoder.layers.{i}", _ownedCasts);
        _weightsLoaded = true;
    }

    /// <summary>Allocates a window's decode state and projects the cross-attention K/V from the encoder memory.</summary>
    /// <param name="memory">Projected encoder output, F32 <c>[memoryTokens, dim]</c> or <c>[1, memoryTokens, dim]</c>.</param>
    /// <param name="maxTokens">Positions the self-attention cache must hold; the decode refuses to pass it.</param>
    public SheetSage2DecodeState StartDecode(IBackend backend, Tensor memory, int maxTokens)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(memory);
        ThrowIfDisposed();
        if (!_weightsLoaded) throw new InvalidOperationException("Call LoadWeights before StartDecode.");
        if (memory.DType != DType.F32)
            throw new ArgumentException($"SheetSage2 memory must be F32; got {memory.DType}.", nameof(memory));
        bool batched = memory.Shape.Rank == 3 && memory.Shape[0] == 1;
        if (!batched && memory.Shape.Rank != 2)
            throw new ArgumentException($"SheetSage2 memory must be [tokens, {_config.Dim}] or [1, tokens, {_config.Dim}]; got {memory.Shape}.", nameof(memory));
        if (memory.Shape[memory.Shape.Rank - 1] != _config.Dim)
            throw new ArgumentException($"SheetSage2 memory must be {_config.Dim} wide; got {memory.Shape}.", nameof(memory));
        if (maxTokens < 2 || maxTokens > _config.MaxTokens)
            throw new ArgumentOutOfRangeException(nameof(maxTokens), maxTokens, $"Must be in [2, {_config.MaxTokens}].");

        int memoryTokens = (int)memory.Shape[batched ? 1 : 0];
        SheetSage2DecodeState state = new(_config, memoryTokens, maxTokens, _tokenizer.TokenCount);
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].PrecomputeCrossKv(backend, memory, state.CrossKey[i], state.CrossValue[i], memoryTokens);
        return state;
    }

    /// <summary>Runs the decoder over <paramref name="tokenIds"/> and leaves the last position's logits in
    /// <see cref="SheetSage2DecodeState.Logits"/>.</summary>
    /// <remarks>A multi-token call is the prompt prefill and is only accepted on a fresh state; every later call
    /// carries one token, which is the path the KV cache and the buffer reuse are built for.</remarks>
    public void Decode(IBackend backend, ReadOnlySpan<int> tokenIds, SheetSage2DecodeState state)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(state);
        ThrowIfDisposed();
        if (!_weightsLoaded) throw new InvalidOperationException("Call LoadWeights before Decode.");
        int chunk = tokenIds.Length;
        if (chunk == 0) throw new ArgumentException("No tokens to decode.", nameof(tokenIds));
        if (chunk > 1 && state.Position != 0)
            throw new NotSupportedException("SheetSage2 accepts a multi-token chunk only as the prefill of a fresh state.");
        if (state.Position + chunk > state.Capacity)
            throw new InvalidOperationException($"Decoding {chunk} token(s) at {state.Position} would pass the cache's {state.Capacity} positions.");

        SheetSage2Scratch scratch = state.Scratch;
        scratch.EnsureChunk(chunk);
        EmbedTokens(scratch.Embed, tokenIds, state.Position);
        backend.LayerNorm(scratch.Hidden, scratch.Embed, _embedNormWeight!, _embedNormBias!, _config.LayerNormEps);
        if (chunk > 1) FillCausalMask(scratch.CausalMask!, chunk);

        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].Forward(backend, scratch, state.SelfKey[i], state.SelfValue[i],
                state.CrossKey[i], state.CrossValue[i], state.Position, chunk);
        }

        Tensor last = chunk == 1 ? scratch.Hidden : CopyLastRow(scratch.LastHidden, scratch.Hidden, chunk);
        backend.Linear(state.Logits, last, _outputProjection!, null);
        state.Position += chunk;
    }

    /// <summary>Transcribes one window: greedy argmax under the grammar mask, exactly as the released decode runs it.</summary>
    /// <param name="prefix">Tokens to seed from; empty uses <see cref="ScoreTokenizer.PromptPrefix"/>. An overlap
    /// prefix replays the previous window's events, so the grammar is advanced over everything after the out token.</param>
    /// <param name="maxTokens">Token budget including the prefix; 0 uses <see cref="SheetSage2Config.MaxTokens"/>.</param>
    public List<int> GenerateTokens(IBackend backend, Tensor memory, double stopSeconds,
        ReadOnlySpan<int> prefix = default, int maxTokens = 0)
    {
        ThrowIfDisposed();
        ReadOnlySpan<int> prompt = prefix.IsEmpty ? ScoreTokenizer.PromptPrefix() : prefix;
        int limit = maxTokens <= 0 ? _config.MaxTokens : maxTokens;
        if (limit > _config.MaxTokens)
            throw new ArgumentOutOfRangeException(nameof(maxTokens), maxTokens, $"Must not pass {_config.MaxTokens}.");
        if (prompt.Length >= limit)
            throw new ArgumentException($"A {prompt.Length}-token prefix leaves no room in a {limit}-token budget.", nameof(prefix));
        int outIndex = prompt.IndexOf(ScoreTokenizer.OutToken);
        if (outIndex < 0)
            throw new ArgumentException("A SheetSage2 prefix must contain the out token that opens the transcription.", nameof(prefix));

        PromptGrammar grammar = new(_tokenizer);
        for (int i = outIndex + 1; i < prompt.Length; i++) grammar.Update(prompt[i]);

        List<int> tokens = new(limit + 1);
        for (int i = 0; i < prompt.Length; i++) tokens.Add(prompt[i]);

        using SheetSage2DecodeState state = StartDecode(backend, memory, limit);
        Decode(backend, prompt, state);

        // Both buffers live for the whole decode: the mask is as wide as the vocabulary and the decode runs one
        // step per token, so allocating either inside the loop allocates it thousands of times per window.
        bool[] allowed = new bool[_tokenizer.TokenCount];
        Span<int> next = stackalloc int[1];
        for (int step = prompt.Length; step < limit; step++)
        {
            grammar.Allowed(allowed);
            int token = ArgmaxAllowed(state.Logits, allowed);
            tokens.Add(token);
            if (grammar.Update(token)) break;
            if (token >= _tokenizer.TimeStart && token < _tokenizer.TimeEnd && _tokenizer.TokenToSeconds(token) >= stopSeconds)
            {
                tokens.Add(ScoreTokenizer.EosToken);
                break;
            }
            if (tokens.Count == limit)
            {
                Logs.Warning($"SheetSage2 reached its {limit}-token limit; the transcription may be incomplete.");
                tokens.Add(ScoreTokenizer.EosToken);
                break;
            }
            next[0] = token;
            Decode(backend, next, state);
        }
        return tokens;
    }

    /// <summary>Enumerates every loaded weight, for backend preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_tokenEmbedding is not null) yield return _tokenEmbedding;
        if (_positionEmbedding is not null) yield return _positionEmbedding;
        if (_embedNormWeight is not null) yield return _embedNormWeight;
        if (_embedNormBias is not null) yield return _embedNormBias;
        foreach (SheetSage2DecoderLayer layer in _layers)
        {
            foreach (Tensor weight in layer.EnumerateWeights()) yield return weight;
        }
        if (_outputProjection is not null) yield return _outputProjection;
    }

    /// <summary>Greedy argmax over the allowed ids, first index winning a tie as the released <c>argmax</c> does.</summary>
    private static int ArgmaxAllowed(Tensor logits, ReadOnlySpan<bool> allowed)
    {
        float* scores = (float*)logits.DataPointer;
        int best = -1;
        float bestScore = float.NegativeInfinity;
        for (int i = 0; i < allowed.Length; i++)
        {
            if (allowed[i] && scores[i] > bestScore)
            {
                bestScore = scores[i];
                best = i;
            }
        }
        if (best < 0) throw new InvalidOperationException("The SheetSage2 grammar allowed no token to follow.");
        return best;
    }

    /// <summary>Writes token + position embeddings; position rows are offset by
    /// <see cref="SheetSage2Config.PositionOffset"/> from the token index.</summary>
    private void EmbedTokens(Tensor output, ReadOnlySpan<int> tokenIds, int position)
    {
        int dim = _config.Dim;
        float* target = (float*)output.DataPointer;
        float* tokens = (float*)_tokenEmbedding!.DataPointer;
        float* positions = (float*)_positionEmbedding!.DataPointer;
        for (int s = 0; s < tokenIds.Length; s++)
        {
            int id = tokenIds[s];
            if (id < 0 || id >= _tokenizer.TokenCount)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), id, $"Outside the {_tokenizer.TokenCount}-token vocabulary.");
            long tokenRow = (long)id * dim;
            long positionRow = (long)(position + s + _config.PositionOffset) * dim;
            long outRow = (long)s * dim;
            for (int k = 0; k < dim; k++) target[outRow + k] = tokens[tokenRow + k] + positions[positionRow + k];
        }
    }

    /// <summary>Fills a <c>[1, 1, chunk, chunk]</c> mask: 0 up to the diagonal, -inf past it.</summary>
    private static void FillCausalMask(Tensor mask, int chunk)
    {
        float* target = (float*)mask.DataPointer;
        for (int q = 0; q < chunk; q++)
        {
            int row = q * chunk;
            for (int k = 0; k < chunk; k++) target[row + k] = k <= q ? 0f : float.NegativeInfinity;
        }
    }

    /// <summary>Copies the last of <paramref name="chunk"/> rows out of a prefill's hidden state; only the final
    /// position is projected to logits.</summary>
    private Tensor CopyLastRow(Tensor target, Tensor hidden, int chunk)
    {
        int dim = _config.Dim;
        float* to = (float*)target.DataPointer;
        float* from = (float*)hidden.DataPointer + (long)(chunk - 1) * dim;
        Buffer.MemoryCopy(from, to, (long)dim * sizeof(float), (long)dim * sizeof(float));
        return target;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(SheetSage2Decoder));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor cast in _ownedCasts) cast.Dispose();
        _ownedCasts.Clear();
    }
}

/// <summary>One SheetSage2 decoder layer: self-attention, cross-attention, then the feed-forward, each closed by
/// a layer norm over its residual sum.</summary>
/// <remarks>Self-attention is genuine incremental decode — the new K/V are appended into the fixed cache and
/// attended with the decode-tuned flash kernel. Cross-attention is not: it reads thousands of memory tokens every
/// step, which is encoder-shaped work, so it goes to <see cref="IBackend.ScaledDotProductAttention"/> instead.</remarks>
internal sealed unsafe class SheetSage2DecoderLayer(SheetSage2Config config)
{
    private readonly SheetSage2Config _config = config;

    private Tensor? _selfQWeight, _selfQBias, _selfKWeight, _selfKBias, _selfVWeight, _selfVBias, _selfOutWeight, _selfOutBias;
    private Tensor? _selfNormWeight, _selfNormBias;
    private Tensor? _crossQWeight, _crossQBias, _crossKWeight, _crossKBias, _crossVWeight, _crossVBias, _crossOutWeight, _crossOutBias;
    private Tensor? _crossNormWeight, _crossNormBias;
    private Tensor? _fc1Weight, _fc1Bias, _fc2Weight, _fc2Bias;
    private Tensor? _finalNormWeight, _finalNormBias;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix, List<Tensor> ownedCasts)
    {
        int dim = _config.Dim;
        int inner = _config.IntermediateSize;
        _selfQWeight = SheetSage2Weights.Take(weights, $"{prefix}.self_attn.q_proj.weight", ownedCasts, dim, dim);
        _selfQBias = SheetSage2Weights.Take(weights, $"{prefix}.self_attn.q_proj.bias", ownedCasts, dim);
        _selfKWeight = SheetSage2Weights.Take(weights, $"{prefix}.self_attn.k_proj.weight", ownedCasts, dim, dim);
        _selfKBias = SheetSage2Weights.Take(weights, $"{prefix}.self_attn.k_proj.bias", ownedCasts, dim);
        _selfVWeight = SheetSage2Weights.Take(weights, $"{prefix}.self_attn.v_proj.weight", ownedCasts, dim, dim);
        _selfVBias = SheetSage2Weights.Take(weights, $"{prefix}.self_attn.v_proj.bias", ownedCasts, dim);
        _selfOutWeight = SheetSage2Weights.Take(weights, $"{prefix}.self_attn.out_proj.weight", ownedCasts, dim, dim);
        _selfOutBias = SheetSage2Weights.Take(weights, $"{prefix}.self_attn.out_proj.bias", ownedCasts, dim);
        _selfNormWeight = SheetSage2Weights.Take(weights, $"{prefix}.self_attn_layer_norm.weight", ownedCasts, dim);
        _selfNormBias = SheetSage2Weights.Take(weights, $"{prefix}.self_attn_layer_norm.bias", ownedCasts, dim);

        _crossQWeight = SheetSage2Weights.Take(weights, $"{prefix}.encoder_attn.q_proj.weight", ownedCasts, dim, dim);
        _crossQBias = SheetSage2Weights.Take(weights, $"{prefix}.encoder_attn.q_proj.bias", ownedCasts, dim);
        _crossKWeight = SheetSage2Weights.Take(weights, $"{prefix}.encoder_attn.k_proj.weight", ownedCasts, dim, dim);
        _crossKBias = SheetSage2Weights.Take(weights, $"{prefix}.encoder_attn.k_proj.bias", ownedCasts, dim);
        _crossVWeight = SheetSage2Weights.Take(weights, $"{prefix}.encoder_attn.v_proj.weight", ownedCasts, dim, dim);
        _crossVBias = SheetSage2Weights.Take(weights, $"{prefix}.encoder_attn.v_proj.bias", ownedCasts, dim);
        _crossOutWeight = SheetSage2Weights.Take(weights, $"{prefix}.encoder_attn.out_proj.weight", ownedCasts, dim, dim);
        _crossOutBias = SheetSage2Weights.Take(weights, $"{prefix}.encoder_attn.out_proj.bias", ownedCasts, dim);
        _crossNormWeight = SheetSage2Weights.Take(weights, $"{prefix}.encoder_attn_layer_norm.weight", ownedCasts, dim);
        _crossNormBias = SheetSage2Weights.Take(weights, $"{prefix}.encoder_attn_layer_norm.bias", ownedCasts, dim);

        _fc1Weight = SheetSage2Weights.Take(weights, $"{prefix}.fc1.weight", ownedCasts, inner, dim);
        _fc1Bias = SheetSage2Weights.Take(weights, $"{prefix}.fc1.bias", ownedCasts, inner);
        _fc2Weight = SheetSage2Weights.Take(weights, $"{prefix}.fc2.weight", ownedCasts, dim, inner);
        _fc2Bias = SheetSage2Weights.Take(weights, $"{prefix}.fc2.bias", ownedCasts, dim);
        _finalNormWeight = SheetSage2Weights.Take(weights, $"{prefix}.final_layer_norm.weight", ownedCasts, dim);
        _finalNormBias = SheetSage2Weights.Take(weights, $"{prefix}.final_layer_norm.bias", ownedCasts, dim);
    }

    /// <summary>Projects the memory's cross-attention K/V into the decode state, once per window.</summary>
    public void PrecomputeCrossKv(IBackend backend, Tensor memory, Tensor key, Tensor value, int memoryTokens)
    {
        int dim = _config.Dim;
        using Tensor flatKey = new(new TensorShape(1, memoryTokens, dim), DType.F32);
        using Tensor flatValue = new(new TensorShape(1, memoryTokens, dim), DType.F32);
        backend.Linear(flatKey, memory, _crossKWeight!, _crossKBias);
        backend.Linear(flatValue, memory, _crossVWeight!, _crossVBias);
        backend.Permute0213(key, flatKey, memoryTokens, _config.NumHeads, _config.HeadDim);
        backend.Permute0213(value, flatValue, memoryTokens, _config.NumHeads, _config.HeadDim);
    }

    /// <summary>Runs the layer over <see cref="SheetSage2Scratch.Hidden"/>, in place.</summary>
    public void Forward(IBackend backend, SheetSage2Scratch scratch, Tensor selfKey, Tensor selfValue,
        Tensor crossKey, Tensor crossValue, int position, int chunk)
    {
        int heads = _config.NumHeads;
        int headDim = _config.HeadDim;
        float scale = _config.AttentionScale;

        backend.Linear(scratch.Query, scratch.Hidden, _selfQWeight!, _selfQBias);
        backend.Linear(scratch.Key, scratch.Hidden, _selfKWeight!, _selfKBias);
        backend.Linear(scratch.Value, scratch.Hidden, _selfVWeight!, _selfVBias);
        backend.Permute0213(scratch.QueryHeads, scratch.Query, chunk, heads, headDim);
        backend.Permute0213(scratch.KeyHeads, scratch.Key, chunk, heads, headDim);
        backend.Permute0213(scratch.ValueHeads, scratch.Value, chunk, heads, headDim);
        backend.KvCacheAppend(selfKey, scratch.KeyHeads, position);
        backend.KvCacheAppend(selfValue, scratch.ValueHeads, position);
        if (chunk == 1)
        {
            backend.FlashAttention(scratch.Attention, scratch.QueryHeads, selfKey, selfValue,
                kvLen: position + 1, kvGroup: 1, causal: true, qOffset: position, scale);
        }
        else
        {
            // A prefill starts from an empty cache, so the keys it just wrote are the whole prefix and the
            // over-allocated cache never has to be sliced back out.
            backend.ScaledDotProductAttention(scratch.Attention, scratch.QueryHeads, scratch.KeyHeads,
                scratch.ValueHeads, scratch.CausalMask, scale);
        }
        backend.Permute0213(scratch.AttentionFlat, scratch.Attention, heads, chunk, headDim);
        backend.Linear(scratch.Projected, scratch.AttentionFlat, _selfOutWeight!, _selfOutBias);
        backend.Add(scratch.ResidualSum, scratch.Hidden, scratch.Projected);
        backend.LayerNorm(scratch.Normed, scratch.ResidualSum, _selfNormWeight!, _selfNormBias!, _config.LayerNormEps);

        backend.Linear(scratch.Query, scratch.Normed, _crossQWeight!, _crossQBias);
        backend.Permute0213(scratch.QueryHeads, scratch.Query, chunk, heads, headDim);
        // Encoder-shaped work at thousands of memory tokens, where this repo's decode-tuned flash kernel is far
        // slower; F16 is in range on this memory (scores ~1.4e3, |V| ~22, both measured against the encoder's own).
        backend.ScaledDotProductAttention(scratch.Attention, scratch.QueryHeads, crossKey, crossValue,
            mask: null, scale, allowF16: true);
        backend.Permute0213(scratch.AttentionFlat, scratch.Attention, heads, chunk, headDim);
        backend.Linear(scratch.Projected, scratch.AttentionFlat, _crossOutWeight!, _crossOutBias);
        backend.Add(scratch.ResidualSum, scratch.Normed, scratch.Projected);
        backend.LayerNorm(scratch.Hidden, scratch.ResidualSum, _crossNormWeight!, _crossNormBias!, _config.LayerNormEps);

        backend.Linear(scratch.Feed, scratch.Hidden, _fc1Weight!, _fc1Bias);
        backend.GeluErf(scratch.FeedActivated, scratch.Feed);
        backend.Linear(scratch.Projected, scratch.FeedActivated, _fc2Weight!, _fc2Bias);
        backend.Add(scratch.ResidualSum, scratch.Hidden, scratch.Projected);
        backend.LayerNorm(scratch.Hidden, scratch.ResidualSum, _finalNormWeight!, _finalNormBias!, _config.LayerNormEps);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all =
        [
            _selfQWeight, _selfQBias, _selfKWeight, _selfKBias, _selfVWeight, _selfVBias, _selfOutWeight, _selfOutBias,
            _selfNormWeight, _selfNormBias,
            _crossQWeight, _crossQBias, _crossKWeight, _crossKBias, _crossVWeight, _crossVBias, _crossOutWeight, _crossOutBias,
            _crossNormWeight, _crossNormBias,
            _fc1Weight, _fc1Bias, _fc2Weight, _fc2Bias, _finalNormWeight, _finalNormBias,
        ];
        foreach (Tensor? weight in all)
        {
            if (weight is not null) yield return weight;
        }
    }

}

/// <summary>Reads the decoder's tensors out of a checkpoint, casting once to F32 and checking every shape.</summary>
/// <remarks>The shapes are checked rather than trusted: <see cref="IBackend.Linear"/> takes its row count from
/// the element count, so a weight of the wrong width is silently mis-read instead of refused.</remarks>
internal static class SheetSage2Weights
{
    public static Tensor Take(IReadOnlyDictionary<string, Tensor> weights, string key, List<Tensor> ownedCasts, params int[] shape)
    {
        if (!weights.TryGetValue(key, out Tensor? raw))
            throw new ArgumentException($"The SheetSage2 checkpoint is missing '{key}'.", nameof(weights));
        if (raw.Shape.Rank != shape.Length)
            throw new ArgumentException($"SheetSage2 '{key}' must have rank {shape.Length}; the checkpoint has {raw.Shape}.", nameof(weights));
        for (int i = 0; i < shape.Length; i++)
        {
            if (raw.Shape[i] != shape[i])
                throw new ArgumentException($"SheetSage2 '{key}' must be [{string.Join(", ", shape)}]; the checkpoint has {raw.Shape}.", nameof(weights));
        }
        Tensor cast = WhisperOps.EnsureF32(raw);
        if (!ReferenceEquals(cast, raw)) ownedCasts.Add(cast);
        return cast;
    }
}

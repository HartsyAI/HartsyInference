using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Music;

/// <summary>MiniMax Music 3's local language model, which autoregressively predicts the seven residual RVQ codebooks within one audio frame and exposes the per-step hidden states that condition the flow-matching stage.</summary>
/// <remarks>Steps are decoded against a <see cref="MiniMaxMusic3DepthCache"/> rather than by re-running the sequence
/// per codebook as the reference does. It also owns the residual codebooks' embedding table, which the
/// autoregressive loop uses to embed complete frames for the global model's feedback.</remarks>
public sealed unsafe class MiniMaxMusic3DepthDecoder : IDisposable
{
    /// <summary>RVQ layers: one semantic codebook plus seven residual ones.</summary>
    public const int NumCodebooks = 8;

    /// <summary>Entries per residual codebook.</summary>
    public const int AudioVocabSize = 1024;

    /// <summary>Longest depth sequence: the global hidden state, the frame's semantic code, and the six residual codes that are fed back — the seventh is sampled from the last step and never re-enters.</summary>
    public const int MaxDepthSteps = NumCodebooks;

    private const int HiddenSize = 4096;
    private const int IntermediateSize = 6144;
    private const int NumLayers = 4;
    private const int NumHeads = 16;
    private const float RmsNormEps = 1e-6f;

    private readonly Layer[] _layers = new Layer[NumLayers];
    private readonly List<Tensor> _owned = [];
    private Tensor? _audioEmbeddings;
    private Tensor? _projection;
    private Tensor? _positionEmbedding;
    private Tensor? _finalNorm;
    private readonly Tensor?[] _audioHeads = new Tensor?[NumCodebooks - 1];
    private int _disposed;

    public MiniMaxMusic3DepthDecoder()
    {
        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i] = new Layer();
        }
    }

    /// <summary>Head width — <c>hidden / heads</c>.</summary>
    public static int HeadDim => HiddenSize / NumHeads;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        // The checkpoint is BF16. The embedding tables and the norm weights are read directly rather than through
        // a backend op, so they are widened here; the projections keep their dtype for the matmul path.
        _audioEmbeddings = Widen(weights["audio_embeddings.weight"]);
        _projection = weights["projection.weight"];
        _positionEmbedding = Widen(weights["pos_embedding.weight"]);
        _finalNorm = Widen(weights["norm.weight"]);
        for (int i = 0; i < _audioHeads.Length; i++)
        {
            _audioHeads[i] = weights[$"audio_heads.{i}.weight"];
        }
        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].Load(this, weights, $"layers.{i}");
        }
    }

    /// <summary>Projects <paramref name="hidden"/> <c>[rows, 4096]</c> through the shared input projection — the depth sequence is built from these, both for the global hidden state and for each embedded code.</summary>
    public Tensor Project(IBackend backend, Tensor hidden)
    {
        Tensor projected = new Tensor(hidden.Shape, DType.F32);
        backend.Linear(projected, hidden, _projection!, null);
        return projected;
    }

    /// <summary>Embeds residual code <paramref name="code"/> of codebook <paramref name="index"/> (1-based over the residual books, matching <c>audio_heads[index-1]</c>) into <c>[1, 4096]</c>.</summary>
    public Tensor EmbedResidual(int index, int code)
    {
        Tensor embedded = new Tensor(new TensorShape(1, HiddenSize), DType.F32);
        CopyEmbeddingRow(_audioEmbeddings!, ((index - 1) * AudioVocabSize) + code, (float*)embedded.DataPointer);
        return embedded;
    }

    /// <summary>Adds the residual codebooks' embeddings for one complete frame into <paramref name="destination"/> <c>[4096]</c>, the sum the global model's feedback embedding needs.</summary>
    public void AccumulateFrameResiduals(ReadOnlySpan<int> frameCodes, Span<float> destination)
    {
        for (int index = 1; index < NumCodebooks; index++)
        {
            AddEmbeddingRow(_audioEmbeddings!, ((index - 1) * AudioVocabSize) + frameCodes[index], destination);
        }
    }

    /// <summary>A cache for one frame's depth sequence; <paramref name="batch"/> is the classifier-free row count.</summary>
    public MiniMaxMusic3DepthCache CreateCache(int batch)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batch);
        return new MiniMaxMusic3DepthCache(NumLayers, batch * NumHeads, HeadDim, MaxDepthSteps);
    }

    /// <summary>Appends <paramref name="sequence"/> <c>[batch, steps, 4096]</c> to <paramref name="cache"/> and returns those steps' final-normed hidden states, same shape.</summary>
    /// <remarks>The absolute position of each step is the cache's length plus its offset in the block, which is what
    /// lets the absolute position embedding be added here — it is applied to the INPUT, before layer 0, so a cached
    /// step only ever needs its own row.</remarks>
    public Tensor Forward(IBackend backend, Tensor sequence, MiniMaxMusic3DepthCache cache)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(sequence);
        ArgumentNullException.ThrowIfNull(cache);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_projection is null)
        {
            throw new InvalidOperationException($"{nameof(MiniMaxMusic3DepthDecoder)}.{nameof(LoadWeights)} must run before {nameof(Forward)}.");
        }
        int batch = (int)sequence.Shape[0];
        int steps = (int)sequence.Shape[1];
        int position = cache.Length;

        Tensor hidden = new Tensor(sequence.Shape, DType.F32);
        float* destination = (float*)hidden.DataPointer;
        ReadOnlySpan<float> source = sequence.AsReadOnlySpan<float>();
        ReadOnlySpan<float> positions = _positionEmbedding!.AsReadOnlySpan<float>();
        for (int row = 0; row < batch * steps; row++)
        {
            int step = position + (row % steps);
            for (int channel = 0; channel < HiddenSize; channel++)
            {
                destination[((long)row * HiddenSize) + channel] =
                    source[(row * HiddenSize) + channel] + positions[(step * HiddenSize) + channel];
            }
        }

        for (int index = 0; index < _layers.Length; index++)
        {
            Tensor next = _layers[index].Forward(backend, hidden, batch, steps, cache, index);
            hidden.Dispose();
            hidden = next;
        }
        cache.Advance(steps);
        Tensor normed = new Tensor(hidden.Shape, DType.F32);
        backend.RmsNorm(normed, hidden, _finalNorm!, RmsNormEps);
        hidden.Dispose();
        return normed;
    }

    /// <summary>Logits of residual codebook <paramref name="index"/> (1-based) for <paramref name="hidden"/> <c>[rows, 4096]</c>.</summary>
    public Tensor Head(IBackend backend, int index, Tensor hidden)
    {
        Tensor logits = new Tensor(new TensorShape((int)hidden.Shape[0], AudioVocabSize), DType.F32);
        backend.Linear(logits, hidden, _audioHeads[index - 1]!, null);
        return logits;
    }

    /// <summary>Every loaded weight, for <see cref="IBackend.PreloadWeights"/>/<see cref="IBackend.FreeWeights"/>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_audioEmbeddings, _projection, _positionEmbedding, _finalNorm];
        foreach (Tensor? tensor in top)
        {
            if (tensor is not null) { yield return tensor; }
        }
        foreach (Tensor? head in _audioHeads)
        {
            if (head is not null) { yield return head; }
        }
        foreach (Layer layer in _layers)
        {
            foreach (Tensor tensor in layer.EnumerateWeights())
            {
                yield return tensor;
            }
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        foreach (Tensor tensor in _owned)
        {
            tensor.Dispose();
        }
        _owned.Clear();
    }

    /// <summary>Returns an F32 view of <paramref name="tensor"/>, tracking the cast for disposal when one happens.</summary>
    private Tensor Widen(Tensor tensor)
    {
        Tensor widened = WhisperOps.EnsureF32(tensor);
        if (!ReferenceEquals(widened, tensor))
        {
            _owned.Add(widened);
        }
        return widened;
    }

    private static void CopyEmbeddingRow(Tensor table, int row, float* destination)
    {
        ReadOnlySpan<float> values = table.AsReadOnlySpan<float>();
        values.Slice(row * HiddenSize, HiddenSize).CopyTo(new Span<float>(destination, HiddenSize));
    }

    private static void AddEmbeddingRow(Tensor table, int row, Span<float> destination)
    {
        ReadOnlySpan<float> values = table.AsReadOnlySpan<float>();
        for (int channel = 0; channel < HiddenSize; channel++)
        {
            destination[channel] += values[(row * HiddenSize) + channel];
        }
    }

    private sealed class Layer
    {
        private Tensor? _inputNorm;
        private Tensor? _postAttentionNorm;
        private Tensor? _toQ;
        private Tensor? _toK;
        private Tensor? _toV;
        private Tensor? _toOut;
        private Tensor? _gate;
        private Tensor? _up;
        private Tensor? _down;

        public void Load(MiniMaxMusic3DepthDecoder owner, IReadOnlyDictionary<string, Tensor> weights, string prefix)
        {
            _inputNorm = owner.Widen(weights[$"{prefix}.input_layernorm.weight"]);
            _postAttentionNorm = owner.Widen(weights[$"{prefix}.post_attention_layernorm.weight"]);
            _toQ = weights[$"{prefix}.attn.to_q.weight"];
            _toK = weights[$"{prefix}.attn.to_k.weight"];
            _toV = weights[$"{prefix}.attn.to_v.weight"];
            _toOut = weights[$"{prefix}.attn.to_out.weight"];
            _gate = weights[$"{prefix}.gate_proj.weight"];
            _up = weights[$"{prefix}.up_proj.weight"];
            _down = weights[$"{prefix}.down_proj.weight"];
        }

        public Tensor Forward(IBackend backend, Tensor hidden, int batch, int steps, MiniMaxMusic3DepthCache cache, int index)
        {
            Tensor result = new Tensor(hidden.Shape, DType.F32);
            using (Tensor normed = new Tensor(hidden.Shape, DType.F32))
            {
                backend.RmsNorm(normed, hidden, _inputNorm!, RmsNormEps);
                // Head-major with the batch folded into the heads, the layout the cache stores. For ONE step that
                // is byte-for-byte what Linear already emits — [batch, 1, heads, headDim] and [1, batch·heads, 1,
                // headDim] are the same buffer — so the common path skips all four permutes.
                TensorShape headMajor = new TensorShape(1, batch * NumHeads, steps, HeadDim);
                using Tensor query = new Tensor(headMajor, DType.F32);
                using Tensor key = new Tensor(headMajor, DType.F32);
                using Tensor value = new Tensor(headMajor, DType.F32);
                if (steps == 1)
                {
                    backend.Linear(query, normed, _toQ!, null);
                    backend.Linear(key, normed, _toK!, null);
                    backend.Linear(value, normed, _toV!, null);
                }
                else
                {
                    // Allocated at rank 4 rather than reshaped into it: Tensor.Reshape reads DataPointer, which syncs
                    // a device tensor to the host and returns a HOST pointer, so the view has no GPU residency. The
                    // `merged` write target was the damaging one — the permute wrote host memory and the projection
                    // below then read a device copy nothing had written, corrupting every depth hidden state on CUDA.
                    TensorShape tokenMajor = new TensorShape(batch, steps, NumHeads, HeadDim);
                    using Tensor tokenQuery = new Tensor(tokenMajor, DType.F32);
                    using Tensor tokenKey = new Tensor(tokenMajor, DType.F32);
                    using Tensor tokenValue = new Tensor(tokenMajor, DType.F32);
                    backend.Linear(tokenQuery, normed, _toQ!, null);
                    backend.Linear(tokenKey, normed, _toK!, null);
                    backend.Linear(tokenValue, normed, _toV!, null);
                    backend.Permute0213(query, tokenQuery, steps, NumHeads, HeadDim);
                    backend.Permute0213(key, tokenKey, steps, NumHeads, HeadDim);
                    backend.Permute0213(value, tokenValue, steps, NumHeads, HeadDim);
                }
                Tensor mask = cache.Mask(steps);
                cache.Append(backend, index, key, value);
                using Tensor attention = new Tensor(headMajor, DType.F32);
                backend.ScaledDotProductAttention(attention, query, cache.Keys(index), cache.Values(index), mask,
                    1f / MathF.Sqrt(HeadDim));
                if (steps == 1)
                {
                    backend.Linear(result, attention, _toOut!, null);
                }
                else
                {
                    using Tensor merged = new Tensor(new TensorShape(batch, steps, NumHeads, HeadDim), DType.F32);
                    backend.Permute0213(merged, attention, NumHeads, steps, HeadDim);
                    backend.Linear(result, merged, _toOut!, null);
                }
            }
            backend.Add(result, result, hidden);

            using Tensor feedNorm = new Tensor(hidden.Shape, DType.F32);
            backend.RmsNorm(feedNorm, result, _postAttentionNorm!, RmsNormEps);
            TensorShape wide = new TensorShape(batch, steps, IntermediateSize);
            using Tensor gated = new Tensor(wide, DType.F32);
            using Tensor lifted = new Tensor(wide, DType.F32);
            backend.Linear(gated, feedNorm, _gate!, null);
            backend.Linear(lifted, feedNorm, _up!, null);
            backend.Silu(gated, gated);
            backend.Mul(gated, gated, lifted);
            using Tensor projected = new Tensor(hidden.Shape, DType.F32);
            backend.Linear(projected, gated, _down!, null);
            backend.Add(result, result, projected);
            return result;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [_inputNorm, _postAttentionNorm, _toQ, _toK, _toV, _toOut, _gate, _up, _down];
            foreach (Tensor? tensor in all)
            {
                if (tensor is not null) { yield return tensor; }
            }
        }
    }
}

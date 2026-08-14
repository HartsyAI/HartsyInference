using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Music;

/// <summary>MiniMax Music 3's local language model: within one audio frame it autoregressively predicts the seven
/// residual RVQ codebooks from the global language model's hidden state and the frame's semantic code, and exposes
/// the per-step hidden states that condition the flow-matching stage.
///
/// <para>It runs without a KV cache, exactly as the reference does — the depth sequence never exceeds
/// <see cref="NumCodebooks"/> steps, so recomputing it is cheaper than maintaining one. It also owns the residual
/// codebooks' embedding table, which the autoregressive loop uses to embed complete frames for the global model's
/// feedback.</para></summary>
public sealed unsafe class MiniMaxMusic3DepthDecoder : IDisposable
{
    /// <summary>RVQ layers: one semantic codebook plus seven residual ones.</summary>
    public const int NumCodebooks = 8;

    /// <summary>Entries per residual codebook.</summary>
    public const int AudioVocabSize = 1024;

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
    private Tensor? _causalMask;
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

    /// <summary>Projects <paramref name="hidden"/> <c>[rows, 4096]</c> through the shared input projection — the
    /// depth sequence is built from these, both for the global hidden state and for each embedded code.</summary>
    public Tensor Project(IBackend backend, Tensor hidden)
    {
        Tensor projected = new Tensor(hidden.Shape, DType.F32);
        backend.Linear(projected, hidden, _projection!, null);
        return projected;
    }

    /// <summary>Embeds residual code <paramref name="code"/> of codebook <paramref name="index"/> (1-based over the
    /// residual books, matching <c>audio_heads[index-1]</c>) into <c>[1, 4096]</c>.</summary>
    public Tensor EmbedResidual(int index, int code)
    {
        Tensor embedded = new Tensor(new TensorShape(1, HiddenSize), DType.F32);
        CopyEmbeddingRow(_audioEmbeddings!, ((index - 1) * AudioVocabSize) + code, (float*)embedded.DataPointer);
        return embedded;
    }

    /// <summary>Adds the residual codebooks' embeddings for one complete frame into <paramref name="destination"/>
    /// <c>[4096]</c>, the sum the global model's feedback embedding needs.</summary>
    public void AccumulateFrameResiduals(ReadOnlySpan<int> frameCodes, Span<float> destination)
    {
        for (int index = 1; index < NumCodebooks; index++)
        {
            AddEmbeddingRow(_audioEmbeddings!, ((index - 1) * AudioVocabSize) + frameCodes[index], destination);
        }
    }

    /// <summary>Runs the decoder stack over <paramref name="sequence"/> <c>[batch, steps, 4096]</c> and returns the
    /// final-normed hidden states.</summary>
    public Tensor Forward(IBackend backend, Tensor sequence)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(sequence);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_projection is null)
        {
            throw new InvalidOperationException($"{nameof(MiniMaxMusic3DepthDecoder)}.{nameof(LoadWeights)} must run before {nameof(Forward)}.");
        }
        int batch = (int)sequence.Shape[0];
        int steps = (int)sequence.Shape[1];

        Tensor hidden = new Tensor(sequence.Shape, DType.F32);
        float* destination = (float*)hidden.DataPointer;
        ReadOnlySpan<float> source = sequence.AsReadOnlySpan<float>();
        ReadOnlySpan<float> positions = _positionEmbedding!.AsReadOnlySpan<float>();
        for (int row = 0; row < batch * steps; row++)
        {
            int step = row % steps;
            for (int channel = 0; channel < HiddenSize; channel++)
            {
                destination[((long)row * HiddenSize) + channel] =
                    source[(row * HiddenSize) + channel] + positions[(step * HiddenSize) + channel];
            }
        }

        Tensor mask = CausalMask(steps);
        foreach (Layer layer in _layers)
        {
            Tensor next = layer.Forward(backend, hidden, batch, steps, mask);
            hidden.Dispose();
            hidden = next;
        }
        Tensor normed = new Tensor(hidden.Shape, DType.F32);
        backend.RmsNorm(normed, hidden, _finalNorm!, RmsNormEps);
        hidden.Dispose();
        return normed;
    }

    /// <summary>Logits of residual codebook <paramref name="index"/> (1-based) for <paramref name="hidden"/>
    /// <c>[rows, 4096]</c>.</summary>
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
        _causalMask?.Dispose();
        _causalMask = null;
    }

    private Tensor CausalMask(int steps)
    {
        if (_causalMask is not null && _causalMask.Shape[0] >= steps)
        {
            if (_causalMask.Shape[0] == steps)
            {
                return _causalMask;
            }
            _causalMask.Dispose();
        }
        Tensor mask = new Tensor(new TensorShape(steps, steps), DType.F32);
        float* data = (float*)mask.DataPointer;
        for (int query = 0; query < steps; query++)
        {
            for (int key = 0; key < steps; key++)
            {
                data[(query * steps) + key] = key <= query ? 0f : float.NegativeInfinity;
            }
        }
        _causalMask = mask;
        return mask;
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

        public Tensor Forward(IBackend backend, Tensor hidden, int batch, int steps, Tensor mask)
        {
            Tensor result = new Tensor(hidden.Shape, DType.F32);
            using (Tensor normed = new Tensor(hidden.Shape, DType.F32))
            {
                backend.RmsNorm(normed, hidden, _inputNorm!, RmsNormEps);
                using Tensor query = new Tensor(hidden.Shape, DType.F32);
                using Tensor key = new Tensor(hidden.Shape, DType.F32);
                using Tensor value = new Tensor(hidden.Shape, DType.F32);
                backend.Linear(query, normed, _toQ!, null);
                backend.Linear(key, normed, _toK!, null);
                backend.Linear(value, normed, _toV!, null);

                TensorShape tokenMajor = new TensorShape(batch, steps, NumHeads, HeadDim);
                TensorShape headMajor = new TensorShape(batch, NumHeads, steps, HeadDim);
                using Tensor q = new Tensor(headMajor, DType.F32);
                using Tensor k = new Tensor(headMajor, DType.F32);
                using Tensor v = new Tensor(headMajor, DType.F32);
                backend.Permute0213(q, query.Reshape(tokenMajor), steps, NumHeads, HeadDim);
                backend.Permute0213(k, key.Reshape(tokenMajor), steps, NumHeads, HeadDim);
                backend.Permute0213(v, value.Reshape(tokenMajor), steps, NumHeads, HeadDim);
                using Tensor attention = new Tensor(headMajor, DType.F32);
                backend.ScaledDotProductAttention(attention, q, k, v, mask, 1f / MathF.Sqrt(HeadDim));
                using Tensor merged = new Tensor(hidden.Shape, DType.F32);
                backend.Permute0213(merged.Reshape(tokenMajor), attention, NumHeads, steps, HeadDim);
                backend.Linear(result, merged, _toOut!, null);
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

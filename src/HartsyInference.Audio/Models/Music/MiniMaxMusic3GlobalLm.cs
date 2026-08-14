using HartsyInference.Audio.Models.LanguageModels.Qwen3;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.Music;

/// <summary>MiniMax Music 3's global language model: a Qwen3-8B backbone that emits one semantic RVQ code per 25 Hz
/// audio frame. The decoder body is the shared <see cref="Qwen3Model"/>; this type owns the pieces the checkpoint
/// puts outside it — the token embedding table and the output head.
///
/// <para>The head is stored pre-sliced to the rows generation can ever choose: the 16384 semantic-code rows plus the
/// end-of-audio row, which is exactly the reference's vocabulary mask expressed as a smaller matrix. That turns a
/// 200000-row projection per frame into a 16385-row one and drops the head from 1.6 GB to 268 MB.</para></summary>
public sealed unsafe class MiniMaxMusic3GlobalLm : IDisposable
{
    /// <summary>Token id of the first audio semantic code; code <c>c</c> is token <c>AudioCodeOffset + c</c>.</summary>
    public const int AudioCodeOffset = 151_675;

    /// <summary>Entries in the semantic codebook.</summary>
    public const int SemanticVocabSize = 16_384;

    /// <summary>Token id that ends generation.</summary>
    public const int AudioEndTokenId = 151_670;

    /// <summary>Index of the end-of-audio row within <see cref="SemanticLogits"/>'s output.</summary>
    public const int AudioEndLogitIndex = SemanticVocabSize;

    /// <summary>Backbone width.</summary>
    public const int HiddenSize = 4096;

    /// <summary>Classifier-free branches decoded together: conditional, then unconditional.</summary>
    public const int CfgRows = 2;

    private readonly Qwen3Model _backbone;
    private readonly bool _halfPrecisionKv;
    private Tensor? _embedTokens;
    private Tensor? _semanticHead;
    private int _disposed;

    /// <param name="halfPrecisionKv">Store the KV cache as F16. A five-minute song reaches ~7500 frames, which is
    /// ~2.3 GB per branch at F32 and ~4.5 GB across the guided pair — more headroom than the quantized variants have
    /// on a 12 GB card. CUDA-only; the CPU path keeps F32, which is also what the parity runs compare.</param>
    public MiniMaxMusic3GlobalLm(bool halfPrecisionKv = false)
    {
        _halfPrecisionKv = halfPrecisionKv;
        _backbone = new Qwen3Model(new Qwen3Config
        {
            HiddenSize = HiddenSize,
            NumHiddenLayers = 36,
            NumAttentionHeads = 32,
            NumKeyValueHeads = 8,
            HeadDim = 128,
            IntermediateSize = 12_288,
            MaxPositionEmbeddings = 10_240,
            RopeTheta = 1_000_000f,
            RmsNormEps = 1e-6f,
        });
    }

    /// <summary>Loads the backbone and builds the sliced output head. The embedding table keeps its checkpoint dtype;
    /// rows are gathered and widened on demand.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        _backbone.LoadWeightsHeadless(weights, prefix: "model");
        _embedTokens = weights["model.embed_tokens.weight"];
        _semanticHead = SliceHead(weights["lm_head.weight"]);
    }

    /// <summary>Allocates a decode cache sized for <paramref name="maxSeqLen"/> tokens. One per classifier-free
    /// branch; the caller disposes them.</summary>
    public IKvCache CreateCache(int maxSeqLen) => _halfPrecisionKv
        ? new FixedKvCache(_backbone.NumLayers, batch: 1, _backbone.KvHeads, _backbone.HeadDim, Math.Max(1, maxSeqLen), DType.F16)
        : _backbone.CreateDecodeCache(maxSeqLen);

    /// <summary>Embeds <paramref name="tokenIds"/> into <c>[1, count, 4096]</c>. Caller owns the result.</summary>
    public Tensor Embed(ReadOnlySpan<int> tokenIds)
    {
        Tensor embeds = new Tensor(new TensorShape(1, tokenIds.Length, HiddenSize), DType.F32);
        Span<float> destination = new Span<float>((float*)embeds.DataPointer, tokenIds.Length * HiddenSize);
        for (int i = 0; i < tokenIds.Length; i++)
        {
            ReadEmbeddingRow(tokenIds[i], destination.Slice(i * HiddenSize, HiddenSize));
        }
        return embeds;
    }

    /// <summary>Copies embedding row <paramref name="tokenId"/> into <paramref name="destination"/>
    /// <c>[4096]</c>, widening from the checkpoint's dtype.</summary>
    public void ReadEmbeddingRow(int tokenId, Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ReadRow(_embedTokens!, tokenId, destination);
    }

    /// <summary>Widens one row of a <c>[rows, 4096]</c> table into <paramref name="destination"/>.</summary>
    private static void ReadRow(Tensor table, int row, Span<float> destination)
    {
        long offset = (long)row * HiddenSize;
        if (table.DType == DType.F32)
        {
            table.AsReadOnlySpan<float>().Slice((int)offset, HiddenSize).CopyTo(destination);
            return;
        }
        if (table.DType == DType.BF16)
        {
            ReadOnlySpan<ushort> raw = table.AsReadOnlySpan<ushort>();
            for (int i = 0; i < HiddenSize; i++)
            {
                destination[i] = BitConverter.UInt32BitsToSingle((uint)raw[(int)offset + i] << 16);
            }
            return;
        }
        if (table.DType == DType.F16)
        {
            ReadOnlySpan<Half> raw = table.AsReadOnlySpan<Half>();
            for (int i = 0; i < HiddenSize; i++)
            {
                destination[i] = (float)raw[(int)offset + i];
            }
            return;
        }
        throw new NotSupportedException($"MiniMax Music 3 table has unsupported dtype {table.DType}.");
    }

    /// <summary>Runs the decoder over <paramref name="embeds"/> <c>[1, steps, 4096]</c> and returns the final-normed
    /// hidden state of the LAST step, <c>[1, 4096]</c>. Caller owns the result.</summary>
    public Tensor Forward(IBackend backend, Tensor embeds, IKvCache cache)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(embeds);
        ArgumentNullException.ThrowIfNull(cache);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        int steps = (int)embeds.Shape[1];
        using Tensor hidden = _backbone.ForwardEmbeds(backend, embeds, steps, cache.CurrentLength, cache);
        Tensor last = new Tensor(new TensorShape(1, HiddenSize), DType.F32);
        hidden.AsReadOnlySpan<float>()
            .Slice((steps - 1) * HiddenSize, HiddenSize)
            .CopyTo(new Span<float>((float*)last.DataPointer, HiddenSize));
        return last;
    }

    /// <summary>Runs ONE decode step for both classifier-free branches as a single batch-2 forward and returns their
    /// final-normed hidden states <c>[2, 4096]</c>, row 0 conditional. <paramref name="embeds"/> is
    /// <c>[1, 2, 4096]</c> in the same row order and each row appends to its own cache.
    ///
    /// <para>Single-token decode is bound by streaming the weights, not by the arithmetic, so running the two
    /// branches as separate batch-1 forwards reads all 8.6 GB twice per frame. One batch-2 forward reads it once.
    /// The branches stay position-aligned by construction (identical prompt length, one frame appended to each per
    /// step), which is what lets them share a step at all. Caller owns the result.</para></summary>
    public Tensor ForwardCfgStep(IBackend backend, Tensor embeds, IKvCache conditional, IKvCache unconditional)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(embeds);
        ArgumentNullException.ThrowIfNull(conditional);
        ArgumentNullException.ThrowIfNull(unconditional);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (embeds.Shape[1] != CfgRows)
        {
            throw new ArgumentException($"the classifier-free step takes [1, {CfgRows}, {HiddenSize}] embeds, got {embeds.Shape}.", nameof(embeds));
        }
        Span<int> positions = [conditional.CurrentLength, unconditional.CurrentLength];
        using Tensor both = _backbone.ForwardBatchDecode(backend, embeds, positions, [conditional, unconditional]);
        Tensor rows = new Tensor(new TensorShape(CfgRows, HiddenSize), DType.F32);
        both.AsReadOnlySpan<float>()
            .Slice(0, CfgRows * HiddenSize)
            .CopyTo(new Span<float>((float*)rows.DataPointer, CfgRows * HiddenSize));
        return rows;
    }

    /// <summary>Projects <paramref name="hidden"/> <c>[rows, 4096]</c> onto the reachable vocabulary:
    /// <c>[rows, 16385]</c>, where column <c>c</c> is semantic code <c>c</c> and the final column is end-of-audio.
    /// Caller owns the result.</summary>
    public Tensor SemanticLogits(IBackend backend, Tensor hidden)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        Tensor logits = new Tensor(new TensorShape((int)hidden.Shape[0], SemanticVocabSize + 1), DType.F32);
        backend.Linear(logits, hidden, _semanticHead!, null);
        return logits;
    }

    /// <summary>Every loaded weight, for <see cref="IBackend.PreloadWeights"/>/<see cref="IBackend.FreeWeights"/>.
    /// The embedding table is excluded: rows are gathered host-side, so uploading all 1.6 GB would be pure waste.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor tensor in _backbone.EnumerateWeights())
        {
            yield return tensor;
        }
        if (_semanticHead is not null) { yield return _semanticHead; }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        _semanticHead?.Dispose();
        _semanticHead = null;
        _embedTokens = null;
        _backbone.Dispose();
    }

    /// <summary>Copies the 16384 semantic rows plus the end-of-audio row out of the full head, widened to F32 row by
    /// row — casting the whole 200000-row head first would transiently cost 3.2 GB to keep 268 MB.</summary>
    private static Tensor SliceHead(Tensor head)
    {
        if (head.Shape[1] != HiddenSize)
        {
            throw new ArgumentException($"lm_head must be [vocab, {HiddenSize}], got {head.Shape}.", nameof(head));
        }
        Tensor sliced = new Tensor(new TensorShape(SemanticVocabSize + 1, HiddenSize), DType.F32);
        Span<float> destination = new Span<float>((float*)sliced.DataPointer, (SemanticVocabSize + 1) * HiddenSize);
        for (int code = 0; code < SemanticVocabSize; code++)
        {
            ReadRow(head, AudioCodeOffset + code, destination.Slice(code * HiddenSize, HiddenSize));
        }
        ReadRow(head, AudioEndTokenId, destination.Slice(SemanticVocabSize * HiddenSize, HiddenSize));
        return sliced;
    }
}

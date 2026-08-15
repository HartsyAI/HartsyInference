using HartsyInference.Audio.Models.LanguageModels.Qwen3;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Runtime;
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

    /// <summary>Frames run eagerly through the fixed buffers before the step is captured, so every weight cast,
    /// activation buffer and KV page is already resident and nothing allocates inside the capture.</summary>
    private const int GraphWarmupFrames = 2;

    private readonly Qwen3Model _backbone;
    private readonly bool _halfPrecisionKv;
    // Off by default: it forces F32 KV (see CreateCache), which caps a 12 GB card near four minutes of song.
    private readonly bool _graphDecode = EnvSwitch.IsEnabled("HARTSY_MM3_LM_GRAPH", defaultOn: false);
    // Diagnostic: keeps the dual step and its F32 cache but never captures, which is the only way to A/B the
    // capture itself against the identical kernel sequence run eagerly.
    private readonly bool _graphCapture = EnvSwitch.IsEnabled("HARTSY_MM3_LM_GRAPH_CAPTURE", defaultOn: true);
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
    /// branch; the caller disposes them.
    ///
    /// <para>The graph-captured step forces F32 storage: its device-position attention has no F16 variant, so the
    /// F16 cache would silently drop the whole step back to the eager path. (The KV-scatter half does have one —
    /// <c>lm_kv_append_f16</c> already takes a device position; only the C# guard routes it away.) That doubles
    /// the cache to ~576 KB per frame across the guided pair — measured at 8.9 GB steady state for 375 frames on a
    /// 12 GB card, leaving room for roughly 5,800 frames, so <c>HARTSY_MM3_LM_GRAPH=1</c> buys ~4.6 s on a short
    /// song and costs the back half of a six-minute one. It stays opt-in until that choice is made per request
    /// rather than per build.</para></summary>
    public IKvCache CreateCache(int maxSeqLen) => _halfPrecisionKv && !_graphDecode
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
    public Tensor ForwardCfgStep(IBackend backend, Tensor embeds, IKvCache conditional, IKvCache unconditional,
        CfgGraphSession? graph = null)
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
        // One shared device position serves both rows' RoPE, both KV scatters and both attention calls, so the
        // captured step is only valid while the branches stay aligned. They are by construction; a drift falls
        // back to the eager batched step rather than producing wrong positions.
        if (graph is not null && conditional.CurrentLength == unconditional.CurrentLength)
        {
            return ForwardCfgGraphStep(backend, embeds, conditional, unconditional, graph);
        }
        Span<int> positions = [conditional.CurrentLength, unconditional.CurrentLength];
        using Tensor both = _backbone.ForwardBatchDecode(backend, embeds, positions, [conditional, unconditional]);
        Tensor rows = new Tensor(new TensorShape(CfgRows, HiddenSize), DType.F32);
        both.AsReadOnlySpan<float>()
            .Slice(0, CfgRows * HiddenSize)
            .CopyTo(new Span<float>((float*)rows.DataPointer, CfgRows * HiddenSize));
        return rows;
    }

    /// <summary>Allocates the per-generation state the graph-captured classifier-free step needs, or null when the
    /// backend or the architecture cannot replay it (the caller then gets the eager batched step). The capture bakes
    /// the KV caches' device addresses, so the session belongs to exactly the caches it was created against and must
    /// be disposed with them.</summary>
    public CfgGraphSession? CreateCfgGraph(IBackend backend, int maxSeqLen)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_graphDecode || !_backbone.SupportsDualGraphDecode(backend))
        {
            return null;
        }
        return new CfgGraphSession(backend, maxSeqLen);
    }

    /// <summary>The classifier-free step as one graph replay. The whole batch-2 forward is ~36 layers of kernel
    /// launches whose host cost is paid on every frame; captured once, it replays as a single launch. Positions,
    /// the input embedding and the output hidden all live in fixed device buffers refreshed OUTSIDE the capture,
    /// so one capture is valid for every later frame.</summary>
    private Tensor ForwardCfgGraphStep(IBackend backend, Tensor embeds, IKvCache conditional, IKvCache unconditional,
        CfgGraphSession session)
    {
        GraphStream stream = session.Stream;
        int position = conditional.CurrentLength;
        backend.CopyInto(stream.InEmbed, embeds);
        backend.WriteDevicePos(stream.DevicePos, position + 1, position);
        (Tensor cos, Tensor sin) = _backbone.EnsureRopeTableForGraphDecode(backend, session.MaxSequenceLength);
        if (stream.Graph is not null)
        {
            backend.LaunchGraph(stream.Graph);
        }
        else if (_graphCapture && !session.CaptureDeclined && stream.Warmed >= GraphWarmupFrames)
        {
            // Capture records the step without executing it, so the capturing frame still has to be launched.
            stream.Graph = backend.CaptureGraph(() => _backbone.ForwardGraphDecodeStepDualEmbeds(
                backend, stream.InEmbed, conditional, unconditional, cos, sin, stream.DevicePos, stream.OutHidden));
            if (stream.Graph is not null)
            {
                backend.LaunchGraph(stream.Graph);
            }
            else
            {
                // Backend declined to capture: run eagerly for the rest of the song rather than retrying the
                // (not free) capture on every frame.
                session.CaptureDeclined = true;
                _backbone.ForwardGraphDecodeStepDualEmbeds(backend, stream.InEmbed, conditional, unconditional,
                    cos, sin, stream.DevicePos, stream.OutHidden);
            }
        }
        else
        {
            _backbone.ForwardGraphDecodeStepDualEmbeds(backend, stream.InEmbed, conditional, unconditional,
                cos, sin, stream.DevicePos, stream.OutHidden);
            stream.Warmed++;
        }
        conditional.AdvanceLength(1);
        unconditional.AdvanceLength(1);
        Tensor rows = new Tensor(new TensorShape(CfgRows, HiddenSize), DType.F32);
        backend.ReadResidentInto(stream.OutHidden, session.Scratch);
        session.Scratch.AsSpan(0, CfgRows * HiddenSize)
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

    /// <summary>Per-generation state for the graph-captured classifier-free step: the fixed input/output buffers and
    /// device position the capture bakes in, plus the host landing buffer the step's hidden states are read into.</summary>
    public sealed class CfgGraphSession : IDisposable
    {
        internal GraphStream Stream { get; }

        internal float[] Scratch { get; }

        internal int MaxSequenceLength { get; }

        /// <summary>Set once the backend refuses to capture, so the step stops paying for doomed capture attempts.</summary>
        internal bool CaptureDeclined { get; set; }

        internal CfgGraphSession(IBackend backend, int maxSeqLen)
        {
            Stream = new GraphStream(backend, HiddenSize, CfgRows);
            Scratch = new float[CfgRows * HiddenSize];
            MaxSequenceLength = Math.Max(1, maxSeqLen);
        }

        public void Dispose() => Stream.Dispose();
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

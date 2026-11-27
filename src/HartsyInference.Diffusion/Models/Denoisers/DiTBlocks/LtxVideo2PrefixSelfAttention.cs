using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Streaming;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>What one pass of a chunk-causal LTX-2 video stream does with the stream's state.</summary>
internal enum LtxVideo2ChunkPassKind
{
    /// <summary>Reads the committed state and predicts; writes nothing.</summary>
    Predict,

    /// <summary>Stages the sink frames' raw key and value projections. The frames attend to themselves only.</summary>
    CaptureSink,

    /// <summary>Reads the committed state and stages the chunk's rotated keys and values for the sliding history.</summary>
    CaptureHistory,
}

/// <summary>The inputs of one pass that every block shares. All tensors are borrowed for the pass.</summary>
internal sealed class LtxVideo2ChunkPass
{
    /// <summary>What the pass does with the stream's state.</summary>
    public required LtxVideo2ChunkPassKind Kind { get; init; }

    /// <summary>Latent frames of the chunk.</summary>
    public required int Frames { get; init; }

    /// <summary>The projected caption <c>[L, dim]</c> the cross-attention reads.</summary>
    public required Tensor Encoder { get; init; }

    /// <summary>The pass's AdaLN-Single modulation, <c>[1, 6·dim]</c>.</summary>
    public required Tensor Temb { get; init; }

    /// <summary>The video rotation the tables were built with.</summary>
    public required LtxVideo2Rope Rope { get; init; }

    /// <summary>The chunk's rotation tables at its absolute frames.</summary>
    public required Tensor Cos { get; init; }

    /// <inheritdoc cref="Cos"/>
    public required Tensor Sin { get; init; }

    /// <summary>The sink's tables at the frame just ahead of the sliding history; null when the pass reads no sink.</summary>
    public Tensor? SinkCos { get; init; }

    /// <inheritdoc cref="SinkCos"/>
    public Tensor? SinkSin { get; init; }

    /// <summary>The sink's tables at the frame just ahead of the chunk, for a block that holds no history.</summary>
    public Tensor? SinkCosAlone { get; init; }

    /// <inheritdoc cref="SinkCosAlone"/>
    public Tensor? SinkSinAlone { get; init; }

    /// <summary>A unit gate <c>[dim]</c> for the ungated residual of the cross-attention.</summary>
    public required Tensor Ones { get; init; }
}

/// <summary>LTX-2 video self-attention for a chunk-causal stream (upstream SANA-WM <c>_streaming_self_attention</c> in its
/// autoregressive mode): the chunk's queries attend to a prefix of earlier keys and values followed by the chunk's own. The
/// prefix is the stream's sink, re-rotated to sit just ahead of the window, then the sliding history, whose keys keep the
/// rotation of the frames they were produced at. Without a stream it is plain LTX-2 self-attention.</summary>
/// <remarks>Flow: <c>to_q/to_k/to_v</c> of the normed hidden state (plus <c>to_gate_logits</c> where the checkpoint has it);
/// the full-width QK RMSNorm fused with the split rotation, as the dual-stream LTX-2 path runs it; the prefix joined in one
/// concatenation per stream; token-major attention; <c>to_out</c>. Everything stays token-major <c>[S, inner]</c>, so the
/// joins are contiguous copies and no head permute runs on a backend with the token-major entry.</remarks>
internal sealed class LtxVideo2PrefixSelfAttention
{
    private readonly int _dim, _heads, _headDim, _inner;
    private readonly float _qkEps;
    private readonly AttentionLayer _attend;

    private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _nq, _nk, _gateW, _gateB;

    public LtxVideo2PrefixSelfAttention(int dim, int heads, int headDim, float qkEps)
    {
        _dim = dim;
        _heads = heads;
        _headDim = headDim;
        _inner = heads * headDim;
        _qkEps = qkEps;
        // The dual-stream path's attention call: token-major where the backend serves it, F16 scores allowed because the
        // RMS-normed Q and K bound them.
        _attend = new AttentionLayer(new AttentionLayerConfig
        {
            Heads = heads, KvHeads = heads, HeadDim = headDim,
            TokenMajorRoute = AttentionRoute.TokenMajor, TokenMajorKnob = EngineKnobs.Ltx2Tokenmajor, AllowF16 = true,
        });
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _qW = w[$"{p}.to_q.weight"]; w.TryGetValue($"{p}.to_q.bias", out _qB);
        _kW = w[$"{p}.to_k.weight"]; w.TryGetValue($"{p}.to_k.bias", out _kB);
        _vW = w[$"{p}.to_v.weight"]; w.TryGetValue($"{p}.to_v.bias", out _vB);
        _oW = w[$"{p}.to_out.0.weight"]; w.TryGetValue($"{p}.to_out.0.bias", out _oB);
        if (_qW.Shape[0] != _inner || _qW.Shape[_qW.Shape.Rank - 1] != _dim)
            throw new InvalidOperationException($"'{p}.to_q' is {_qW.Shape}; the block declared [{_inner}, {_dim}].");
        _nq = TensorCasts.LoadF32(w, LtxVideo2Attention.NormKey(w, p, "q_norm", "norm_q"));
        _nk = TensorCasts.LoadF32(w, LtxVideo2Attention.NormKey(w, p, "k_norm", "norm_k"));
        if (w.TryGetValue($"{p}.to_gate_logits.weight", out Tensor? gate))
        {
            _gateW = gate;
            w.TryGetValue($"{p}.to_gate_logits.bias", out _gateB);
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _nq, _nk, _gateW, _gateB })
            if (t is not null) yield return t;
    }

    /// <summary>Self-attention of the chunk's rows <paramref name="x"/> <c>[S, dim]</c>. Returns <c>[S, dim]</c>, which the
    /// caller owns, or null when <paramref name="captureOnly"/> ends the pass once the keys and values are staged.</summary>
    /// <param name="stream">This block's carried state; null attends to the chunk alone.</param>
    public Tensor? Forward(IBackend backend, Tensor x, LtxVideo2ChunkPass pass, LtxVideo2LayerStream? stream, bool captureOnly)
    {
        if (_qW is null)
            throw new InvalidOperationException("LtxVideo2PrefixSelfAttention.LoadWeights has not run.");
        if (pass.Kind != LtxVideo2ChunkPassKind.Predict && stream is null)
            throw new ArgumentException("A capture pass needs the stream it writes to.", nameof(stream));
        if (captureOnly && pass.Kind == LtxVideo2ChunkPassKind.Predict)
            throw new ArgumentException("Only a capture pass can stop at the keys and values.", nameof(captureOnly));
        int tokens = (int)x.Shape[0];
        DType act = x.DType;
        TensorShape flat = new(tokens, _inner);
        Tensor? gate = null, q = null, k = null, v = null, keys = null, values = null, attended = null;
        try
        {
            if (_gateW is not null && !captureOnly)
                gate = new Tensor(new TensorShape(tokens, _heads), act);
            if (!captureOnly)
                q = new Tensor(flat, act);
            k = new Tensor(flat, act);
            v = new Tensor(flat, act);
            Project(backend, x, gate, q, k, v);

            if (pass.Kind == LtxVideo2ChunkPassKind.CaptureSink)
            {
                // The slot keeps the raw key: every later chunk norms and rotates it to its own position.
                if (!captureOnly)
                {
                    keys = NormRotate(backend, k, _nk!, pass, pass.Cos, pass.Sin, tokens);
                    values = StreamingTensors.Clone(backend, v);
                }

                stream!.SinkKey.Stage(Take(ref k));
                stream.SinkValue.Stage(Take(ref v));
                if (captureOnly)
                    return null;
            }
            else
            {
                Tensor rotated = NormRotate(backend, k, _nk!, pass, pass.Cos, pass.Sin, tokens);
                k.Dispose();
                k = rotated;
                RollingTokenWindow? staging = pass.Kind == LtxVideo2ChunkPassKind.CaptureHistory ? stream!.History : null;
                if (!captureOnly)
                {
                    keys = Join(backend, stream, pass, k, LtxVideo2LayerStream.KeyStream);
                    values = Join(backend, stream, pass, v, LtxVideo2LayerStream.ValueStream);
                }

                // A join copies the chunk's rows, so they are free to move into the window; without a prefix they are the
                // attention's operands and the window takes copies.
                if (staging is not null)
                {
                    Tensor stagedKey = ReferenceEquals(keys, k) ? StreamingTensors.Clone(backend, k) : Take(ref k);
                    Tensor stagedValue;
                    try
                    {
                        stagedValue = ReferenceEquals(values, v) ? StreamingTensors.Clone(backend, v) : Take(ref v);
                    }
                    catch
                    {
                        stagedKey.Dispose();
                        throw;
                    }

                    staging.Stage([stagedKey, stagedValue], pass.Frames);
                }

                if (captureOnly)
                    return null;
                if (ReferenceEquals(keys, k))
                    k = null;
                if (ReferenceEquals(values, v))
                    v = null;
            }

            Tensor query = NormRotate(backend, q!, _nq!, pass, pass.Cos, pass.Sin, tokens);
            q!.Dispose();
            q = null;
            // The attention call takes over its three operands, also when it throws.
            Tensor keyOperand = Take(ref keys), valueOperand = Take(ref values);
            attended = _attend.AttendFlat(backend, query, keyOperand, valueOperand, flat, default);

            Tensor output = new(new TensorShape(tokens, _dim), act);
            try
            {
                if (gate is not null)
                    backend.LinearFused(output, attended, _oW!, _oB, LinearFusion.HeadGated(gate, _heads, _headDim));
                else
                    backend.Linear(output, attended, _oW!, _oB);
                return output;
            }
            catch
            {
                output.Dispose();
                throw;
            }
        }
        finally
        {
            gate?.Dispose();
            q?.Dispose();
            k?.Dispose();
            v?.Dispose();
            keys?.Dispose();
            values?.Dispose();
            attended?.Dispose();
        }
    }

    // Grouped by input, as the dual-stream path projects: a backend that preprocesses the activation does it once.
    private void Project(IBackend backend, Tensor x, Tensor? gate, Tensor? q, Tensor k, Tensor v)
    {
        Span<LinearOp> ops = new LinearOp[4];
        int count = 0;
        if (gate is not null)
            ops[count++] = new LinearOp(gate, _gateW!, _gateB);
        if (q is not null)
            ops[count++] = new LinearOp(q, _qW!, _qB);
        ops[count++] = new LinearOp(k, _kW!, _kB);
        ops[count++] = new LinearOp(v, _vW!, _vB);
        backend.LinearMulti(x, ops[..count]);
    }

    // The stream's prefix in front of the chunk's rows: the sink (its key normed and rotated to this chunk's position), then
    // the history oldest first. Returns the chunk's own tensor when there is no prefix, else a joined copy.
    private Tensor Join(IBackend backend, LtxVideo2LayerStream? stream, LtxVideo2ChunkPass pass, Tensor chunk, int which)
    {
        if (stream is null)
            return chunk;
        bool key = which == LtxVideo2LayerStream.KeyStream;
        RollingTokenWindow? history = stream.History is { IsEmpty: false } window ? window : null;
        Tensor? sink = key ? stream.SinkKey.Current : stream.SinkValue.Current;
        if (sink is null && history is null)
            return chunk;
        Tensor[] parts = new Tensor[(sink is null ? 0 : 1) + (history?.PartCount ?? 0) + 1];
        int at = 0;
        Tensor? rotatedSink = null;
        try
        {
            if (sink is not null)
            {
                if (key)
                {
                    Tensor? cos = history is null ? pass.SinkCosAlone : pass.SinkCos;
                    Tensor? sin = history is null ? pass.SinkSinAlone : pass.SinkSin;
                    if (cos is null || sin is null)
                        throw new InvalidOperationException("The pass carries no rotation tables for the stream's sink.");
                    rotatedSink = NormRotate(backend, sink, _nk!, pass, cos, sin, (int)sink.Shape[0]);
                    parts[at++] = rotatedSink;
                }
                else
                {
                    parts[at++] = sink;
                }
            }

            if (history is not null)
                at += history.CopyParts(which, parts.AsSpan(at));
            parts[at] = chunk;
            return StreamingTensors.Concat(backend, parts, 0);
        }
        finally
        {
            rotatedSink?.Dispose();
        }
    }

    // Full-width RMS norm, then the rotation: split halves fused with the norm (LTX-2), or interleaved pairs after it.
    private Tensor NormRotate(IBackend backend, Tensor input, Tensor normWeight, LtxVideo2ChunkPass pass, Tensor cos, Tensor sin,
        int tokens)
    {
        if (cos.Shape[0] != tokens)
            throw new ArgumentException($"The rotation tables cover {cos.Shape[0]} tokens; the rows hold {tokens}.", nameof(cos));
        Tensor output = new(new TensorShape(tokens, _inner), input.DType);
        try
        {
            if (pass.Rope.Flavor == LtxVideo2Rope.RopeType.Split)
            {
                backend.RmsNormRotary(output, input, normWeight, _qkEps, cos, sin, RotaryViews.SplitRows(tokens, _heads, _headDim));
            }
            else
            {
                backend.RmsNorm(output, input, normWeight, _qkEps);
                backend.Rotary(RotaryCall.Rows(output, cos, sin, tokens, 1, _inner, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair));
            }

            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    private static Tensor Take(ref Tensor? tensor)
    {
        Tensor taken = tensor!;
        tensor = null;
        return taken;
    }
}

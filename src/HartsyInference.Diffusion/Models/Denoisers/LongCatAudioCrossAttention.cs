using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The avatar's audio cross-attention (<c>SingleStreamAttention</c>, <c>modules/avatar/attention.py</c> @ 58eb336). Each latent frame's
/// <c>S</c> video tokens attend to that frame's <c>ctx</c> audio tokens only (with several speakers, every speaker's tokens of that frame):
/// q is a projection of the video tokens with a per-head RMSNorm, k and v are the two halves of one <c>kv_linear</c> over the audio tokens with
/// a per-head RMSNorm on k, and the output is projected back to <c>dim</c>.</summary>
/// <remarks>
/// Layout: <c>x</c> is <c>[frames·S, dim]</c> (frame-major) and <c>audio</c> is <c>[frames·ctx, encoderDim]</c>, both F32. The noisy frames
/// are batched as the attention's batch axis, so each frame still attends only to its own audio. With <c>numCond</c> conditioning frames the
/// reference returns zeros for their tokens and computes only the noisy frames; this returns the same, so the output is <c>[frames·S, dim]</c>.
/// Weights under <c>attn.</c>: <c>q_linear</c>, <c>q_norm</c>, <c>kv_linear</c>, <c>k_norm</c>, <c>proj</c>. The reference's RoPE1D runs only for
/// several speakers: the caller gives its tables (<see cref="LongCatSpeakerRope"/>), and q and k are rotated after their norms.
/// </remarks>
public sealed class LongCatAudioCrossAttention : IDisposable
{
    /// <summary>Epsilon of the per-head RMSNorm (<c>SingleStreamAttention</c> default).</summary>
    public const float NormEps = 1e-6f;

    private readonly int _dim, _encoderDim, _heads, _headDim;
    private readonly AttentionLayerConfig _config;
    private AttentionLayer? _attention;
    private bool _bf16Route;

    /// <summary>Turns the BF16 route on or off: the attention layer's projections round as the reference's do (see <see cref="AttentionLayer.RoundProjection"/>).</summary>
    internal void SetBf16Route(bool on)
    {
        _bf16Route = on;
        if (_attention is not null)
            _attention.RoundProjection = on ? LongCatOps.RoundFn : null;
    }
    private Tensor? _qW, _qB, _qNorm, _kvW, _kvB, _kNorm, _projW, _projB;
    private List<Tensor> _owned = [];
    private bool _loaded;
    private int _disposed;

    /// <summary>Creates the layer geometry. <paramref name="dim"/> must divide evenly by <paramref name="heads"/>.</summary>
    public LongCatAudioCrossAttention(int dim, int encoderDim, int heads)
    {
        if (dim < 1 || encoderDim < 1 || heads < 1)
            throw new ArgumentOutOfRangeException(nameof(dim), "dim, encoderDim and heads must be positive.");
        if (dim % heads != 0)
            throw new ArgumentException($"dim {dim} must divide evenly by heads {heads}.", nameof(heads));
        _dim = dim;
        _encoderDim = encoderDim;
        _heads = heads;
        _headDim = dim / heads;
        _config = new AttentionLayerConfig
        {
            Heads = heads, KvHeads = heads, HeadDim = _headDim, Projection = AttentionProjection.PerWeight,
            QkNorm = AttentionQkNorm.PerHead, ComputeDType = DType.F32,
        };
    }

    /// <summary>Reads the eight tensors under <paramref name="prefix"/>. All-or-nothing: every tensor is checked before the previous set
    /// (if any) is released, so a failed load keeps the layer as it was.</summary>
    /// <exception cref="ObjectDisposedException">The layer was disposed.</exception>
    /// <exception cref="InvalidDataException">A tensor is missing or has the wrong shape.</exception>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "attn.")
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        ArgumentNullException.ThrowIfNull(weights);
        List<Tensor> owned = [];
        try
        {
            // Matrices stay as stored (a quantized or F16 checkpoint keeps its adjunct path); vectors are widened, and the copies are owned here.
            Tensor qW = CheckpointTensors.Require(weights, "LongCat", prefix + "q_linear.weight", [_dim, _dim]);
            Tensor qB = CheckpointTensors.RequireF32(weights, "LongCat", prefix + "q_linear.bias", owned, [_dim]);
            Tensor qN = CheckpointTensors.RequireF32(weights, "LongCat", prefix + "q_norm.weight", owned, [_headDim]);
            Tensor kvW = CheckpointTensors.Require(weights, "LongCat", prefix + "kv_linear.weight", [2 * _dim, _encoderDim]);
            Tensor kvB = CheckpointTensors.RequireF32(weights, "LongCat", prefix + "kv_linear.bias", owned, [2 * _dim]);
            Tensor kN = CheckpointTensors.RequireF32(weights, "LongCat", prefix + "k_norm.weight", owned, [_headDim]);
            Tensor pW = CheckpointTensors.Require(weights, "LongCat", prefix + "proj.weight", [_dim, _dim]);
            Tensor pB = CheckpointTensors.RequireF32(weights, "LongCat", prefix + "proj.bias", owned, [_dim]);

            AttentionLayer attention = new(_config, new AttentionNormWeights(qN, kN, NormEps, NormEps));
            attention.RoundProjection = _bf16Route ? LongCatOps.RoundFn : null;
            ReleaseOwned();
            _owned = owned;
            _attention = attention;
            _qW = qW; _qB = qB; _qNorm = qN; _kvW = kvW; _kvB = kvB; _kNorm = kN; _projW = pW; _projB = pB;
            _loaded = true;
        }
        catch
        {
            foreach (Tensor t in owned)
                t.Dispose();
            throw;
        }
    }

    /// <summary>Audio cross-attention over <paramref name="frames"/> latent frames. See the type remarks for the layout.</summary>
    /// <returns>A new <c>[frames·S, dim]</c> tensor the caller owns.</returns>
    /// <exception cref="ObjectDisposedException">The layer was disposed.</exception>
    /// <exception cref="InvalidOperationException">No weights are loaded.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="frames"/> or <paramref name="numCond"/> is out of range.</exception>
    /// <exception cref="ArgumentException"><paramref name="x"/> or <paramref name="audio"/> is not F32, has the wrong width, or has rows that do not split per frame.</exception>
    public Tensor Forward(IBackend backend, Tensor x, Tensor audio, int frames, int numCond) =>
        Forward(backend, x, audio, frames, numCond, null);

    /// <summary>The forward with <paramref name="rotation"/>, the multi-speaker L-RoPE tables over the noisy tokens and their audio tokens; null for
    /// one speaker.</summary>
    internal Tensor Forward(IBackend backend, Tensor x, Tensor audio, int frames, int numCond, LongCatSpeakerRotation? rotation)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(audio);
        if (!_loaded)
            throw new InvalidOperationException("The audio cross-attention has no weights: LoadWeights was never called, or it failed.");
        if (frames < 1)
            throw new ArgumentOutOfRangeException(nameof(frames), frames, "frames must be at least 1.");
        if (numCond < 0 || numCond >= frames)
            throw new ArgumentOutOfRangeException(nameof(numCond), numCond, $"numCond must lie in [0, {frames - 1}] (at least one noisy frame).");
        if (x.DType != DType.F32)
            throw new ArgumentException($"x must be F32, got {x.DType}.", nameof(x));
        if (audio.DType != DType.F32)
            throw new ArgumentException($"audio must be F32, got {audio.DType}.", nameof(audio));
        if (x.Shape.Rank != 2 || x.Shape[1] != _dim || x.Shape[0] % frames != 0)
            throw new ArgumentException($"x must be [frames·S, {_dim}] with frames={frames}, got {x.Shape}.", nameof(x));
        if (audio.Shape.Rank != 2 || audio.Shape[1] != _encoderDim || audio.Shape[0] % frames != 0)
            throw new ArgumentException($"audio must be [frames·ctx, {_encoderDim}] with frames={frames} (a row count that is not a multiple "
                + $"of frames cannot be split per frame), got {audio.Shape}.", nameof(audio));

        int tokens = (int)x.Shape[0] / frames, ctx = (int)audio.Shape[0] / frames;
        int noisy = frames - numCond;
        int noisyTokens = noisy * tokens, noisyCtx = noisy * ctx;
        if (rotation is { } tables && (tables.QueryCos.Shape[0] != noisyTokens || tables.KeyCos.Shape[0] != noisyCtx))
            throw new ArgumentException($"The speaker rotation covers {tables.QueryCos.Shape[0]} queries and {tables.KeyCos.Shape[0]} keys; "
                + $"the noisy frames have {noisyTokens} and {noisyCtx}.", nameof(rotation));

        // The noisy frames, batched on the attention's batch axis: [noisy, tokens, heads, D] for q and [noisy, ctx, heads, D] for k and v.
        // With no conditioning frames the noisy rows are all of x and audio, so they are used in place rather than copied.
        Tensor? noisyX = null, noisyAudio = null;
        Tensor projected;
        try
        {
            if (numCond > 0)
            {
                noisyX = new(new TensorShape(noisyTokens, _dim), DType.F32);
                backend.SliceRows(noisyX, x, numCond * tokens);
                noisyAudio = new(new TensorShape(noisyCtx, _encoderDim), DType.F32);
                backend.SliceRows(noisyAudio, audio, numCond * ctx);
            }
            projected = AttendBatched(backend, noisyX ?? x, noisyAudio ?? audio, noisy, tokens, ctx, rotation);
        }
        finally
        {
            noisyX?.Dispose();
            noisyAudio?.Dispose();
        }
        if (numCond == 0)
            return projected;

        using (projected)
        {
            // The reference returns zeros for the conditioning frames' tokens (output_cond), then the noisy frames' output.
            using Tensor zeros = new(new TensorShape(numCond * tokens, _dim), DType.F32);
            backend.Fill(zeros, 0f);
            Tensor output = new(new TensorShape(frames * tokens, _dim), DType.F32);
            backend.Concat(output, [zeros, projected], 0);
            return output;
        }
    }

    /// <summary>All noisy frames at once: q, k and v, one attention over the batch of frames, then the output projection.</summary>
    private Tensor AttendBatched(IBackend backend, Tensor noisyX, Tensor noisyAudio, int batch, int tokens, int ctx,
        LongCatSpeakerRotation? rotation)
    {
        // The Linear and LinearWeightRows writes use the 4-D layout the attention layer reads; the element counts match.
        Tensor q = new(new TensorShape(batch, tokens, _heads, _headDim), DType.F32);
        Tensor k = new(new TensorShape(batch, ctx, _heads, _headDim), DType.F32);
        Tensor v = new(new TensorShape(batch, ctx, _heads, _headDim), DType.F32);
        try
        {
            backend.Linear(q, noisyX, _qW!, _qB);
            backend.LinearWeightRows(k, noisyAudio, _kvW!, _kvB, 0, _dim);
            backend.LinearWeightRows(v, noisyAudio, _kvW!, _kvB, _dim, _dim);
        }
        catch
        {
            q.Dispose();
            k.Dispose();
            v.Dispose();
            throw;
        }
        // Once its argument checks pass, Attend owns q, k and v and disposes them, even if it throws later; so it sits outside the catch above.
        // A rotation table covers the batch's rows in order, so one table serves all frames; the layer applies it after the QK norm.
        AttentionCallContext context = rotation is { } tables
            ? new AttentionCallContext
            {
                QueryRotation = new QkRotation(QkRotationKind.InterleavedPairs, tables.QueryCos, tables.QuerySin, batch * tokens),
                KeyRotation = new QkRotation(QkRotationKind.InterleavedPairs, tables.KeyCos, tables.KeySin, batch * ctx),
            }
            : default;
        Tensor attended = _attention!.Attend(backend, q, k, v, new TensorShape(batch * tokens, _dim), in context);
        using (attended)
        {
            Tensor projected = new(new TensorShape(batch * tokens, _dim), DType.F32);
            backend.Linear(projected, attended, _projW!, _projB);
            return projected;
        }
    }

    /// <summary>The loaded weights (matrices as stored, vectors widened); empty before a load.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _qW, _qB, _qNorm, _kvW, _kvB, _kNorm, _projW, _projB })
            if (t is not null)
                yield return t;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        ReleaseOwned();
        _qW = _qB = _kvW = _kvB = _projW = _projB = null;
        _attention = null;
        _loaded = false;
    }

    private void ReleaseOwned()
    {
        foreach (Tensor t in _owned)
            t.Dispose();
        _owned = [];
    }

}

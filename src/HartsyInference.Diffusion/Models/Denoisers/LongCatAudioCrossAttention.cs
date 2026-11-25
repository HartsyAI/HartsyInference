using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The avatar's audio cross-attention (<c>SingleStreamAttention</c>, <c>modules/avatar/attention.py</c> @ 58eb336) for a single speaker
/// without the multi-speaker reference map. Each latent frame's <c>S</c> video tokens attend to that frame's <c>ctx</c> audio tokens only:
/// q is a projection of the video tokens with a per-head RMSNorm, k and v are the two halves of one <c>kv_linear</c> over the audio tokens with
/// a per-head RMSNorm on k, and the output is projected back to <c>dim</c>.</summary>
/// <remarks>
/// Layout: <c>x</c> is <c>[frames·S, dim]</c> (frame-major), <c>audio</c> is <c>[frames·ctx, encoderDim]</c>. With <c>numCond</c> conditioning
/// frames the reference returns zeros for their tokens and computes only the noisy frames, which is what this returns too. The output is
/// <c>[frames·S, dim]</c>. Weights under <c>attn.</c>: <c>q_linear</c>, <c>q_norm</c>, <c>kv_linear</c>, <c>k_norm</c>, <c>proj</c>. The
/// reference's RoPE1D on q and k runs only when a multi-speaker reference map is given; that path is not in this class.
/// </remarks>
public sealed class LongCatAudioCrossAttention : IDisposable
{
    /// <summary>Epsilon of the per-head RMSNorm (<c>SingleStreamAttention</c> default).</summary>
    public const float NormEps = 1e-6f;

    private readonly int _dim, _encoderDim, _heads, _headDim;
    private readonly AttentionLayerConfig _config;
    private AttentionLayer? _attention;
    private Tensor? _qW, _qB, _kvW, _kvB, _projW, _projB, _qNorm, _kNorm;
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

    /// <summary>Reads the eight tensors under <paramref name="prefix"/>; all-or-nothing, and the previous set (if any) is kept on failure.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "attn.")
    {
        ArgumentNullException.ThrowIfNull(weights);
        List<Tensor> owned = [];
        try
        {
            Tensor qW = Expect(LongCatOps.WidenF32(weights, prefix + "q_linear.weight", owned), [_dim, _dim], prefix + "q_linear.weight");
            Tensor qB = Expect(LongCatOps.WidenF32(weights, prefix + "q_linear.bias", owned), [_dim], prefix + "q_linear.bias");
            Tensor qN = Expect(LongCatOps.WidenF32(weights, prefix + "q_norm.weight", owned), [_headDim], prefix + "q_norm.weight");
            Tensor kvW = Expect(LongCatOps.WidenF32(weights, prefix + "kv_linear.weight", owned), [2 * _dim, _encoderDim], prefix + "kv_linear.weight");
            Tensor kvB = Expect(LongCatOps.WidenF32(weights, prefix + "kv_linear.bias", owned), [2 * _dim], prefix + "kv_linear.bias");
            Tensor kN = Expect(LongCatOps.WidenF32(weights, prefix + "k_norm.weight", owned), [_headDim], prefix + "k_norm.weight");
            Tensor pW = Expect(LongCatOps.WidenF32(weights, prefix + "proj.weight", owned), [_dim, _dim], prefix + "proj.weight");
            Tensor pB = Expect(LongCatOps.WidenF32(weights, prefix + "proj.bias", owned), [_dim], prefix + "proj.bias");

            ObjectDisposedException.ThrowIf(_disposed != 0, this);
            foreach (Tensor old in _owned)
                old.Dispose();
            _owned = owned;
            _attention = new AttentionLayer(_config, new AttentionNormWeights(qN, kN, NormEps, NormEps));
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
    public Tensor Forward(IBackend backend, Tensor x, Tensor audio, int frames, int numCond)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(audio);
        if (!_loaded)
            throw new InvalidOperationException("The audio cross-attention has no weights: LoadWeights was never called, or it failed.");
        if (frames < 1)
            throw new ArgumentOutOfRangeException(nameof(frames), frames, "frames must be at least 1.");
        if (numCond < 0 || numCond >= frames)
            throw new ArgumentOutOfRangeException(nameof(numCond), numCond, $"numCond must lie in [0, {frames - 1}] (at least one noisy frame).");
        if (x.Shape.Rank != 2 || x.Shape[1] != _dim || x.Shape[0] % frames != 0)
            throw new ArgumentException($"x must be [frames·S, {_dim}] with frames={frames}, got {x.Shape}.", nameof(x));
        if (audio.Shape.Rank != 2 || audio.Shape[1] != _encoderDim || audio.Shape[0] % frames != 0)
            throw new ArgumentException($"audio must be [frames·ctx, {_encoderDim}] with frames={frames}, got {audio.Shape}.", nameof(audio));

        int tokens = (int)x.Shape[0] / frames, ctx = (int)audio.Shape[0] / frames;
        int noisy = frames - numCond;
        List<Tensor> pieces = [];
        try
        {
            if (numCond > 0)
            {
                // The reference returns zeros for the conditioning frames' tokens (output_cond).
                Tensor zeros = new(new TensorShape(numCond * tokens, _dim), DType.F32);
                backend.Fill(zeros, 0f);
                pieces.Add(zeros);
            }

            for (int f = numCond; f < frames; f++)
            {
                using Tensor frameX = new(new TensorShape(tokens, _dim), DType.F32);
                backend.SliceRows(frameX, x, f * tokens);
                using Tensor frameAudio = new(new TensorShape(ctx, _encoderDim), DType.F32);
                backend.SliceRows(frameAudio, audio, f * ctx);
                pieces.Add(AttendFrame(backend, frameX, frameAudio, tokens, ctx));
            }

            Tensor output = new(new TensorShape(frames * tokens, _dim), DType.F32);
            backend.Concat(output, pieces.ToArray(), 0);
            return output;
        }
        finally
        {
            foreach (Tensor piece in pieces)
                piece.Dispose();
        }
    }

    /// <summary>One frame: q from its video tokens, k and v from its audio tokens, attention, then the output projection.</summary>
    private Tensor AttendFrame(IBackend backend, Tensor frameX, Tensor frameAudio, int tokens, int ctx)
    {
        // The Linear and LinearWeightRows writes use the 4-D layout that the attention layer reads; the element counts match.
        Tensor q = new(new TensorShape(1, tokens, _heads, _headDim), DType.F32);
        Tensor k = new(new TensorShape(1, ctx, _heads, _headDim), DType.F32);
        Tensor v = new(new TensorShape(1, ctx, _heads, _headDim), DType.F32);
        Tensor attended;
        try
        {
            backend.Linear(q, frameX, _qW!, _qB);
            backend.LinearWeightRows(k, frameAudio, _kvW!, _kvB, 0, _dim);
            backend.LinearWeightRows(v, frameAudio, _kvW!, _kvB, _dim, _dim);
        }
        catch
        {
            q.Dispose();
            k.Dispose();
            v.Dispose();
            throw;
        }
        // Attend takes ownership of q, k and v and disposes them.
        attended = _attention!.Attend(backend, q, k, v, new TensorShape(tokens, _dim), default);
        using (attended)
        {
            Tensor projected = new(new TensorShape(tokens, _dim), DType.F32);
            backend.Linear(projected, attended, _projW!, _projB);
            return projected;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Tensor t in _owned)
            t.Dispose();
        _owned = [];
        _loaded = false;
    }

    private static Tensor Expect(Tensor tensor, long[] shape, string key)
    {
        if (tensor.Shape.Rank != shape.Length)
            throw new InvalidDataException($"{key} must have {shape.Length} dimensions, got {tensor.Shape}.");
        for (int i = 0; i < shape.Length; i++)
            if (tensor.Shape[i] != shape[i])
                throw new InvalidDataException($"{key} must be {string.Join('x', shape)}, got {tensor.Shape}.");
        return tensor;
    }
}

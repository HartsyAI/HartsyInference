using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Qwen-Image 2.1 single-stream transformer block (<c>QwenImage21TransformerBlock</c>). One attention and one
/// SwiGLU MLP over the concatenated <c>[text, image]</c> sequence — there is no second (text) stream, no per-block
/// modulation linear, and no affine norm weights anywhere: both LayerNorms are non-affine and are modulated by the
/// transformer's <b>one shared</b> modulation.
///
/// <para>The block is entered twice per generation rather than once per step, because the two halves of the sequence
/// are separable. Text rows take the <c>t = 0</c> modulation and attend causally among themselves, never to image
/// rows, so their K/V are identical at every step: <see cref="ForwardPrefix"/> runs them once and hands back the K/V
/// to cache. <see cref="ForwardTarget"/> then runs image rows alone against <c>[cached K ∥ image K]</c> with the
/// sampled-<c>t</c> modulation. ComfyUI reaches the same place from the other side — it evaluates the whole sequence
/// and caches the prefix afterwards — but factoring it this way means every call has a <b>uniform</b> modulation, so
/// none of the per-row-range scale/gate splitting its <c>_modulated_norm</c>/<c>_gated_residual</c> perform is
/// needed here.</para>
///
/// <para>Modulation arrives pre-biased: the caller passes <c>1 + scale</c>, not <c>scale</c>, and already-tanh'd
/// gates, because the shared modulation is computed once per step and reused by all 32 blocks. The adaLN has no
/// shift term at all (scale-only), which is why <see cref="IBackend.AffineBroadcastLastDim"/> is called with a null
/// shift.</para></summary>
public sealed unsafe class QwenImage21Block : IStreamingBlock
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _mlpDim;
    private readonly float _eps;

    private Tensor? _toQ, _toK, _toV, _toOut;
    private Tensor? _normQ, _normK;
    private Tensor? _gateUp, _mlpOut;

    /// <summary>Creates a 2.1 block. Every projection is bias-free, so only weights are held.</summary>
    /// <param name="mlpDim">SwiGLU inner width (12288 for the released checkpoint); the stored <c>gate_up</c> is
    /// <c>[2·mlpDim, hidden]</c>.</param>
    public QwenImage21Block(int hiddenSize, int numHeads, int headDim, int mlpDim, float eps = 1e-6f)
    {
        if (numHeads * headDim != hiddenSize)
            throw new ArgumentException($"numHeads * headDim ({numHeads} * {headDim}) must equal hiddenSize ({hiddenSize}).");
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = headDim;
        _mlpDim = mlpDim;
        _eps = eps;
    }

    /// <inheritdoc/>
    /// <remarks>Via <see cref="DType.ComputeByteCount"/> rather than <c>ElementCount * SizeInBytes</c>: a
    /// block-quantized dtype reports <c>SizeInBytes == 0</c>, which would total the DiT to weightless and let the
    /// streaming budget admit a model that cannot fit.</remarks>
    public long EstimatedWeightBytes
    {
        get
        {
            long total = 0;
            foreach (Tensor w in EnumerateWeights()) total += w.DType.ComputeByteCount(w.ElementCount);
            return total;
        }
    }

    /// <summary>Loads one block from <c>transformer_blocks.{i}.*</c>. The checkpoint stores gate and up fused as a
    /// single <c>img_mlp.gate_up</c>; they are kept fused so the MLP is one GEMM, and split only after it.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _toQ = weights[$"{prefix}.attn.to_q.weight"];
        _toK = weights[$"{prefix}.attn.to_k.weight"];
        _toV = weights[$"{prefix}.attn.to_v.weight"];
        _toOut = weights[$"{prefix}.attn.to_out.0.weight"];
        _normQ = weights[$"{prefix}.attn.norm_q.weight"];
        _normK = weights[$"{prefix}.attn.norm_k.weight"];
        _gateUp = weights[$"{prefix}.img_mlp.gate_up.weight"];
        _mlpOut = weights[$"{prefix}.img_mlp.out.weight"];

        if ((int)_gateUp.Shape[0] != 2 * _mlpDim)
            throw new ArgumentException($"{prefix}.img_mlp.gate_up.weight has {_gateUp.Shape[0]} rows; expected 2*{_mlpDim} (fused gate+up).");
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? w in new[] { _toQ, _toK, _toV, _toOut, _normQ, _normK, _gateUp, _mlpOut })
            if (w is not null) yield return w;
    }

    /// <summary>Runs the text prefix. Attention is causal within the prefix (ComfyUI gives each text segment a
    /// <c>tril</c> mask), and the post-norm, post-RoPE K/V are handed back for the per-step target pass to attend
    /// over. The caller owns the returned hidden state and both cache tensors.</summary>
    public Tensor ForwardPrefix(IBackend backend, Tensor x, in QwenImage21Modulation mod,
        Tensor ropeCos, Tensor ropeSin, Tensor causalMask, out Tensor keyCache, out Tensor valueCache)
    {
        int seq = (int)x.Shape[x.Shape.Rank - 2];
        (Tensor q, Tensor k, Tensor v) = ProjectQkv(backend, x, mod.Scale1Plus1, seq, ropeCos, ropeSin);
        Tensor attn = Attend(backend, q, k, v, seq, seq, null);
        q.Dispose();
        keyCache = k;
        valueCache = v;
        return FinishBlock(backend, x, attn, mod, seq);
    }

    /// <summary>Runs the target image rows against <c>[prefix K ∥ image K]</c>. Block-causal attention degenerates
    /// to full attention here: every image row attends to the whole prefix and to every image row, so no mask is
    /// built. Pass empty caches for a sequence with no prefix.</summary>
    public Tensor ForwardTarget(IBackend backend, Tensor x, in QwenImage21Modulation mod,
        Tensor ropeCos, Tensor ropeSin, Tensor? prefixKey, Tensor? prefixValue)
    {
        int seq = (int)x.Shape[x.Shape.Rank - 2];
        (Tensor q, Tensor k, Tensor v) = ProjectQkv(backend, x, mod.Scale1Plus1, seq, ropeCos, ropeSin);

        int prefixLen = prefixKey is null ? 0 : (int)prefixKey.Shape[prefixKey.Shape.Rank - 3];
        Tensor keys = k, values = v;
        if (prefixLen > 0)
        {
            TensorShape joint = new TensorShape(1, prefixLen + seq, _numHeads, _headDim);
            keys = new Tensor(joint, k.DType);
            backend.Concat(keys, [prefixKey!, k], 1);
            values = new Tensor(joint, v.DType);
            backend.Concat(values, [prefixValue!, v], 1);
            k.Dispose();
            v.Dispose();
        }

        Tensor attn = Attend(backend, q, keys, values, seq, prefixLen + seq, null);
        q.Dispose();
        keys.Dispose();
        values.Dispose();
        return FinishBlock(backend, x, attn, mod, seq);
    }

    /// <summary>Modulated LayerNorm, then the three projections, per-head QK RMSNorm and in-place interleaved RoPE.
    /// Q/K/V are declared <c>[1, S, H, D]</c> so the RMSNorm's last dim is the head dim and the RoPE kernel — which
    /// reads the same bytes as <c>[S, H·D]</c> — needs no reshape view.</summary>
    private (Tensor Q, Tensor K, Tensor V) ProjectQkv(IBackend backend, Tensor x, Tensor scalePlus1, int seq,
        Tensor ropeCos, Tensor ropeSin)
    {
        DType act = x.DType;
        Tensor modulated = ModulateNorm(backend, x, scalePlus1, seq, act);
        TensorShape heads = new TensorShape(1, seq, _numHeads, _headDim);

        Tensor q = new Tensor(heads, act);
        backend.Linear(q, modulated, _toQ!, null);
        Tensor k = new Tensor(heads, act);
        backend.Linear(k, modulated, _toK!, null);
        Tensor v = new Tensor(heads, act);
        backend.Linear(v, modulated, _toV!, null);
        modulated.Dispose();

        Tensor qn = new Tensor(heads, act);
        backend.RmsNorm(qn, q, _normQ!, _eps);
        q.Dispose();
        Tensor kn = new Tensor(heads, act);
        backend.RmsNorm(kn, k, _normK!, _eps);
        k.Dispose();

        backend.WanRopeInterleaved(qn, ropeCos, ropeSin, seq, _numHeads, _headDim);
        backend.WanRopeInterleaved(kn, ropeCos, ropeSin, seq, _numHeads, _headDim);
        return (qn, kn, v);
    }

    /// <summary>Attention plus the output projection, returning <c>[1, S, hidden]</c>. Uses the token-major kernel
    /// when the backend serves it — Q/K/V already sit in the layout the Linears emitted and <c>to_out</c> wants, so
    /// that path pays no permute on either side.</summary>
    private Tensor Attend(IBackend backend, Tensor q, Tensor k, Tensor v, int qSeq, int kvSeq, Tensor? mask)
    {
        DType act = q.DType;
        float scale = 1.0f / MathF.Sqrt(_headDim);
        // Q and K are RMS-normed per head, so pre-softmax scores are bounded and F16 accumulation is safe.
        Tensor flat = new Tensor(new TensorShape(1, qSeq, _hiddenSize), act);
        if (backend.SupportsTokenMajorAttention)
        {
            backend.ScaledDotProductAttentionTokenMajor(flat, q, k, v, mask, _numHeads, _headDim, scale, allowF16: true);
        }
        else
        {
            Tensor qh = new Tensor(new TensorShape(1, _numHeads, qSeq, _headDim), act);
            backend.Permute0213(qh, q, qSeq, _numHeads, _headDim);
            Tensor kh = new Tensor(new TensorShape(1, _numHeads, kvSeq, _headDim), act);
            backend.Permute0213(kh, k, kvSeq, _numHeads, _headDim);
            Tensor vh = new Tensor(new TensorShape(1, _numHeads, kvSeq, _headDim), act);
            backend.Permute0213(vh, v, kvSeq, _numHeads, _headDim);
            Tensor attn = new Tensor(new TensorShape(1, _numHeads, qSeq, _headDim), act);
            backend.ScaledDotProductAttention(attn, qh, kh, vh, mask, scale, allowF16: true);
            qh.Dispose(); kh.Dispose(); vh.Dispose();
            backend.Permute0213(flat, attn, _numHeads, qSeq, _headDim);
            attn.Dispose();
        }

        Tensor projected = new Tensor(new TensorShape(1, qSeq, _hiddenSize), act);
        backend.Linear(projected, flat, _toOut!, null);
        flat.Dispose();
        return projected;
    }

    /// <summary>The two gated residuals and the MLP between them: <c>x += gate1·attn</c>, then
    /// <c>x += gate2·mlp(norm(x)·scale2)</c>. Consumes <paramref name="attention"/>.</summary>
    private Tensor FinishBlock(IBackend backend, Tensor x, Tensor attention, in QwenImage21Modulation mod, int seq)
    {
        DType act = x.DType;
        TensorShape flat = new TensorShape(1, seq, _hiddenSize);

        Tensor afterAttn = new Tensor(flat, act);
        backend.GatedResidualLastDim(afterAttn, x, attention, mod.Gate1);
        attention.Dispose();

        Tensor mlpIn = ModulateNorm(backend, afterAttn, mod.Scale2Plus1, seq, act);
        Tensor mlpOut = ForwardMlp(backend, mlpIn, seq, act);
        mlpIn.Dispose();

        Tensor result = new Tensor(flat, act);
        backend.GatedResidualLastDim(result, afterAttn, mlpOut, mod.Gate2);
        afterAttn.Dispose();
        mlpOut.Dispose();
        return result;
    }

    /// <summary>Non-affine LayerNorm scaled by the shared modulation. The reference is
    /// <c>LayerNorm(x) * (1 + scale)</c> with no shift; the caller supplies <c>1 + scale</c> already summed, since
    /// one modulation serves every block.</summary>
    private Tensor ModulateNorm(IBackend backend, Tensor x, Tensor scalePlus1, int seq, DType act)
    {
        TensorShape flat = new TensorShape(1, seq, _hiddenSize);
        Tensor normed = new Tensor(flat, act);
        backend.LayerNormNoAffine(normed, x, _eps);
        Tensor scaled = new Tensor(flat, act);
        backend.AffineBroadcastLastDim(scaled, normed, scalePlus1, null);
        normed.Dispose();
        return scaled;
    }

    /// <summary>Fused-<c>gate_up</c> SwiGLU: one <c>[hidden → 2·mlpDim]</c> GEMM, then
    /// <c>silu(gate) * up</c> over the two halves, then the down projection. Gate is the FIRST half — ComfyUI's
    /// <c>_swiglu_eager</c> chunks <c>(gate, up)</c> in that order, and its LoRA key map confirms it by addressing
    /// <c>gate_layer</c> at row 0 and <c>proj</c> at row <c>mlpDim</c> of the same fused matrix.</summary>
    internal Tensor ForwardMlp(IBackend backend, Tensor input, int seq, DType act)
    {
        Tensor fused = new Tensor(new TensorShape(1, seq, 2 * _mlpDim), act);
        backend.Linear(fused, input, _gateUp!, null);

        TensorShape half = new TensorShape(1, seq, _mlpDim);
        Tensor gate = new Tensor(half, act);
        Tensor up = new Tensor(half, act);
        backend.Split([gate, up], fused, 2);
        fused.Dispose();

        Tensor activated = new Tensor(half, act);
        backend.Silu(activated, gate);
        gate.Dispose();
        Tensor gated = new Tensor(half, act);
        backend.Mul(gated, activated, up);
        activated.Dispose();
        up.Dispose();

        Tensor output = new Tensor(new TensorShape(1, seq, _hiddenSize), act);
        backend.Linear(output, gated, _mlpOut!, null);
        gated.Dispose();
        return output;
    }
}

/// <summary>The one modulation every Qwen-Image 2.1 block reads, evaluated once per forward and shared by all of
/// them. Scales arrive as <c>1 + scale</c> and gates already through <c>tanh</c>, so a block does no modulation
/// arithmetic of its own. A pair of these exists per step: one built from <c>t = 0</c> for the text prefix, one from
/// the sampled timestep for the image rows.</summary>
public readonly struct QwenImage21Modulation(Tensor scale1Plus1, Tensor gate1, Tensor scale2Plus1, Tensor gate2) : IDisposable
{
    /// <summary><c>1 + scale</c> for the pre-attention LayerNorm, <c>[1, hidden]</c>.</summary>
    public Tensor Scale1Plus1 { get; } = scale1Plus1;

    /// <summary><c>tanh</c>'d gate on the attention residual, <c>[1, hidden]</c>.</summary>
    public Tensor Gate1 { get; } = gate1;

    /// <summary><c>1 + scale</c> for the pre-MLP LayerNorm, <c>[1, hidden]</c>.</summary>
    public Tensor Scale2Plus1 { get; } = scale2Plus1;

    /// <summary><c>tanh</c>'d gate on the MLP residual, <c>[1, hidden]</c>.</summary>
    public Tensor Gate2 { get; } = gate2;

    public void Dispose()
    {
        Scale1Plus1.Dispose();
        Gate1.Dispose();
        Scale2Plus1.Dispose();
        Gate2.Dispose();
    }
}

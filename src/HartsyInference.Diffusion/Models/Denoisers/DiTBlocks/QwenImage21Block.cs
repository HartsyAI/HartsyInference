using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
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
    private readonly List<Tensor> _owned = new();
    private AttentionLayer? _attention;

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
        // F32 once at load: RmsNorm's fast kernels all require an F32 weight, and the checkpoint ships these
        // BF16, which drops every QK norm onto the cast-operands-on-device fallback. 128 floats per norm.
        _normQ = OwnAsF32(weights[$"{prefix}.attn.norm_q.weight"]);
        _normK = OwnAsF32(weights[$"{prefix}.attn.norm_k.weight"]);
        _gateUp = weights[$"{prefix}.img_mlp.gate_up.weight"];
        _mlpOut = weights[$"{prefix}.img_mlp.out.weight"];

        if ((int)_gateUp.Shape[0] != 2 * _mlpDim)
            throw new ArgumentException($"{prefix}.img_mlp.gate_up.weight has {_gateUp.Shape[0]} rows; expected 2*{_mlpDim} (fused gate+up).");

        // Head-major: the per-head QK RMSNorm needs the head dim last. Q and K are RMS-normed per head, so the call may
        // narrow to F16.
        _attention = new AttentionLayer(
            new AttentionLayerConfig
            {
                Heads = _numHeads, KvHeads = _numHeads, HeadDim = _headDim, QkNorm = AttentionQkNorm.PerHead, AllowF16 = true,
            },
            new AttentionLayerWeights
            {
                Query = _toQ, Key = _toK, Value = _toV, Output = _toOut, QueryNorm = _normQ, KeyNorm = _normK, QueryNormEps = _eps,
                KeyNormEps = _eps,
            });
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? w in new[] { _toQ, _toK, _toV, _toOut, _normQ, _normK, _gateUp, _mlpOut })
            if (w is not null) yield return w;
    }

    /// <summary>Returns <paramref name="source"/> when it is already F32, else an owned F32 copy.</summary>
    private Tensor OwnAsF32(Tensor source)
    {
        if (source.DType == DType.F32)
        {
            return source;
        }
        Tensor converted = TensorCasts.EnsureF32(source);
        if (ReferenceEquals(converted, source))
        {
            return source;
        }
        _owned.Add(converted);
        return converted;
    }

    /// <summary>Releases the F32 norm copies this block created. The checkpoint owns every other weight.</summary>
    internal void DisposeOwned()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }

    /// <summary>Runs the text prefix. Attention is causal within the prefix (ComfyUI gives each text segment a
    /// <c>tril</c> mask), and the post-norm, post-RoPE K/V are handed back for the per-step target pass to attend
    /// over. The caller owns the returned hidden state and both cache tensors.</summary>
    public Tensor ForwardPrefix(IBackend backend, Tensor x, in QwenImage21Modulation mod,
        Tensor ropeCos, Tensor ropeSin, Tensor causalMask, out Tensor keyCache, out Tensor valueCache)
    {
        int seq = (int)x.Shape[x.Shape.Rank - 2];
        Tensor modulated = ModulateNorm(backend, x, mod.Scale1Plus1, seq, x.DType);
        QkRotation rope = new(QkRotationKind.InterleavedPairs, ropeCos, ropeSin, seq);
        CapturedKv cache = new();
        Tensor attn = _attention!.Forward(backend, modulated, modulated, new AttentionCallContext
        {
            QueryRotation = rope, KeyRotation = rope, Bias = causalMask, KvCapture = cache, ReleaseInputs = true,
        });
        keyCache = cache.Key!;
        valueCache = cache.Value!;
        return FinishBlock(backend, x, attn, mod, seq);
    }

    /// <summary>Runs the target image rows against <c>[prefix K ∥ image K]</c>. Block-causal attention degenerates
    /// to full attention here: every image row attends to the whole prefix and to every image row, so no mask is
    /// built. Pass empty caches for a sequence with no prefix.</summary>
    public Tensor ForwardTarget(IBackend backend, Tensor x, in QwenImage21Modulation mod,
        Tensor ropeCos, Tensor ropeSin, Tensor? prefixKey, Tensor? prefixValue)
    {
        int seq = (int)x.Shape[x.Shape.Rank - 2];
        Tensor modulated = ModulateNorm(backend, x, mod.Scale1Plus1, seq, x.DType);
        QkRotation rope = new(QkRotationKind.InterleavedPairs, ropeCos, ropeSin, seq);
        bool prefixed = prefixKey is not null && prefixKey.Shape[prefixKey.Shape.Rank - 3] > 0;
        Tensor attn = _attention!.Forward(backend, modulated, modulated, new AttentionCallContext
        {
            QueryRotation = rope, KeyRotation = rope, ReleaseInputs = true,
            KvRewrite = prefixed ? new PrefixKv(prefixKey!, prefixValue!) : null,
        });
        return FinishBlock(backend, x, attn, mod, seq);
    }

    /// <summary>Prepends the cached prefix K/V to the target's own <c>[1, S, H, D]</c> K/V.</summary>
    private sealed class PrefixKv(Tensor key, Tensor value) : IKvRewrite
    {
        public (Tensor K, Tensor V) Rewrite(IBackend backend, Tensor k, Tensor v)
        {
            TensorShape joint = new(1, key.Shape[1] + k.Shape[1], k.Shape[2], k.Shape[3]);
            Tensor? keys = null, values = null;
            try
            {
                keys = new(joint, k.DType);
                backend.Concat(keys, [key, k], 1);
                values = new(joint, v.DType);
                backend.Concat(values, [value, v], 1);
                return (keys, values);
            }
            catch
            {
                keys?.Dispose();
                values?.Dispose();
                throw;
            }
            finally
            {
                k.Dispose();
                v.Dispose();
            }
        }
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
        if (act == DType.F16)
        {
            // The reference clips the block output to F16's range for exactly this dtype
            // (qwen_image21/model.py, end of QwenImage21TransformerBlock.forward): the SwiGLU intermediate can
            // momentarily exceed 65504, and an Inf in the residual stream poisons every later block.
            backend.Clamp(result, result, -65504f, 65504f);
        }
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
/// arithmetic of its own. Always <b>F32</b>, independent of the activation dtype, because the backend's 16-bit
/// DiT recipe is 16-bit activations with an F32 scale/gate. A pair of these exists per step: one built from <c>t = 0</c> for the text prefix, one from
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

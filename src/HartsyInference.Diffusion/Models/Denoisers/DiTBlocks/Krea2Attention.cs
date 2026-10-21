using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Krea 2 self-attention (<c>Krea2Attention</c> + <c>Krea2AttnProcessor</c>). Grouped-query attention with
/// per-head zero-centered RMSNorm on Q and K, optional 3-axis RoPE, and a learned <b>sigmoid output gate</b>:
/// <c>out = to_out( sdpa(q, k, v) · sigmoid(to_gate(x)) )</c>. Shared by the main transformer blocks (with RoPE, GQA)
/// and the text-fusion blocks (no RoPE, full MHA).
///
/// <para>The zero-centered RMSNorm scales (<c>weight + 1</c>) are folded to plain RMSNorm weights at load time (the
/// caller adds 1.0), so the per-head norm runs through <see cref="IBackend.RmsNorm"/> unchanged.</para></summary>
public sealed class Krea2Attention
{
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly float _eps;

    private Tensor? _toQ, _toK, _toV, _toGate, _toOut, _normQ, _normK;
    private AttentionLayer? _layer;

    /// <summary>Creates an attention module. <paramref name="numKvHeads"/> &lt; <paramref name="numHeads"/> enables GQA.</summary>
    public Krea2Attention(int hidden, int numHeads, int numKvHeads, float eps = 1e-5f)
    {
        if (hidden % numHeads != 0)
            throw new ArgumentException($"hidden {hidden} must be divisible by numHeads {numHeads}.");
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException($"numHeads {numHeads} must be divisible by numKvHeads {numKvHeads}.");
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = hidden / numHeads;
        _eps = eps;
    }

    /// <summary>Loads <c>{prefix}.to_q/to_k/to_v/to_gate.weight</c>, <c>{prefix}.to_out.0.weight</c> and the per-head <c>{prefix}.norm_q/norm_k.weight</c> (zero-centered: 1.0 is added at load).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _toQ = w[$"{p}.to_q.weight"];
        _toK = w[$"{p}.to_k.weight"];
        _toV = w[$"{p}.to_v.weight"];
        _toGate = w[$"{p}.to_gate.weight"];
        _toOut = w[$"{p}.to_out.0.weight"];
        _normQ = Krea2Norm.LoadZeroCentered(w[$"{p}.norm_q.weight"]);
        _normK = Krea2Norm.LoadZeroCentered(w[$"{p}.norm_k.weight"]);
        // Q and K are RMS-normed per head, so the scores are bounded and the call may narrow to F16. The grouped token-major
        // entry serves the text-fusion blocks' equal heads too.
        _layer = new AttentionLayer(
            new AttentionLayerConfig
            {
                Heads = _numHeads, KvHeads = _numKvHeads, HeadDim = _headDim, QkNorm = AttentionQkNorm.PerHead,
                Gate = AttentionGate.Sigmoid, TokenMajorRoute = AttentionRoute.TokenMajorGqa, AllowF16 = true,
            },
            new AttentionLayerWeights
            {
                Query = _toQ, Key = _toK, Value = _toV, Gate = _toGate, Output = _toOut, QueryNorm = _normQ, KeyNorm = _normK,
                QueryNormEps = _eps, KeyNormEps = _eps,
            });
    }

    /// <summary>Enumerates weight tensors for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _toQ, _toK, _toV, _toGate, _toOut, _normQ, _normK })
            if (t is not null) yield return t;
    }

    /// <summary>Runs attention over <paramref name="x"/> <c>[B, S, hidden]</c>. Pass <paramref name="rope"/> (already <c>Precompute</c>d for this sequence) for the main blocks; pass null for the text-fusion blocks.</summary>
    /// <param name="vRowScale">SwarmUI's <c>attn1_token_weight_patch</c> value-row scale, already expanded to
    /// <c>v</c>'s own shape and dtype so the multiply is a plain elementwise op. Null on every path but Krea 2's
    /// weighted joint attention.</param>
    public Tensor Forward(IBackend backend, Tensor x, FluxRope? rope, int batch, int seqLen, Tensor? attnBias = null,
        Tensor? vRowScale = null)
    {
        AttentionLayer layer = _layer ?? throw new InvalidOperationException("Krea2Attention.LoadWeights has not run.");
        // The rope rotates on the device for one batch item; a batched call rotates on the host after the head permute.
        AttentionCallContext context = new()
        {
            Rotation = batch == 1 ? rope : null, HeadMajorRotation = batch == 1 ? null : rope, Bias = attnBias,
            ValueRowScale = vRowScale,
        };
        return layer.Forward(backend, x, x, in context);
    }
}

/// <summary>Helper for Krea 2's zero-centered RMSNorm scales (<c>F.rms_norm(x, weight = weight + 1)</c>): loads a norm weight as F32 and folds the <c>+1</c> so the runtime can use plain <see cref="IBackend.RmsNorm"/>.</summary>
public static unsafe class Krea2Norm
{
    /// <summary>Returns an F32 copy of <paramref name="raw"/> with 1.0 added to every element.</summary>
    public static Tensor LoadZeroCentered(Tensor raw)
    {
        Tensor f32 = raw.DType == DType.F32 ? CopyF32(raw) : raw.CastTo(DType.F32);
        float* p = (float*)f32.DataPointer;
        long n = f32.Shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] += 1.0f;
        return f32;
    }

    private static Tensor CopyF32(Tensor raw)
    {
        Tensor copy = new Tensor(raw.Shape, DType.F32);
        long bytes = raw.Shape.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)raw.DataPointer, (void*)copy.DataPointer, bytes, bytes);
        return copy;
    }
}

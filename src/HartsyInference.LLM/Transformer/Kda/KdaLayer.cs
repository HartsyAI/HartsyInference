using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer.Kda;

/// <summary>Shape of a Kimi Delta Attention (KDA) layer: a linear-attention block with a short causal conv on q, k and v, a
/// per-key-channel forget gate, a delta-rule recurrence, and a sigmoid output gate. Heads share one key/value width, as in Kimi.</summary>
/// <param name="Hidden">Model width (the layer's input and output).</param>
/// <param name="Heads">Heads; each keeps one <c>[HeadDim, HeadDim]</c> recurrent state.</param>
/// <param name="HeadDim">Key and value width per head.</param>
/// <param name="ConvKernel">Causal depthwise conv taps on q, k and v.</param>
/// <param name="ForgetRank">Rank of the low-rank forget projection.</param>
/// <param name="GateLowerBound">Forget gate lower bound; the log-decay lies in (bound, 0).</param>
public readonly record struct KdaSettings(int Hidden, int Heads, int HeadDim, int ConvKernel, int ForgetRank, float GateLowerBound)
{
    /// <summary>Width of q, k or v together: <c>Heads · HeadDim</c>.</summary>
    public int KeyDim => Heads * HeadDim;

    /// <summary>Width of the fused q/k/v projection and its conv: <c>3 · KeyDim</c>.</summary>
    public int ConvDim => 3 * KeyDim;
}

/// <summary>Weights of one KDA layer, all F32 and in the <c>[out, in]</c> layout <see cref="IBackend.Linear"/> takes. The forget
/// gate is stored folded (see <see cref="FoldForgetGate"/>), so the layer needs no per-channel broadcast.</summary>
/// <param name="QkvProj"><c>[3·KeyDim, Hidden]</c>: q, k and v projections stacked.</param>
/// <param name="ConvTaps"><c>[ConvDim, ConvKernel]</c> depthwise taps.</param>
/// <param name="ForgetDown"><c>[ForgetRank, Hidden]</c>.</param>
/// <param name="ForgetUp"><c>[KeyDim, ForgetRank]</c>, the folded <c>diag(exp A)·W</c>.</param>
/// <param name="ForgetBias"><c>[KeyDim]</c>, the folded <c>exp(A) ⊙ dt_bias</c>.</param>
/// <param name="BetaProj"><c>[Heads, Hidden]</c>: write-strength projection, pre-sigmoid.</param>
/// <param name="OutGateProj"><c>[KeyDim, Hidden]</c>.</param>
/// <param name="OutNorm"><c>[HeadDim]</c>, the per-head RMSNorm weight.</param>
/// <param name="OutProj"><c>[Hidden, KeyDim]</c>.</param>
public sealed record KdaWeights(Tensor QkvProj, Tensor ConvTaps, Tensor ForgetDown, Tensor ForgetUp, Tensor ForgetBias,
    Tensor BetaProj, Tensor OutGateProj, Tensor OutNorm, Tensor OutProj);

/// <summary>The carried state of one KDA layer across calls: the conv history and the recurrent state.</summary>
public sealed class KdaState : IDisposable
{
    private bool _disposed;

    /// <summary>Creates a zeroed state for <paramref name="settings"/>.</summary>
    public KdaState(KdaSettings settings)
    {
        History = new Tensor(new TensorShape(settings.ConvKernel - 1, settings.ConvDim), DType.F32);
        Recurrent = new Tensor(new TensorShape(settings.Heads, settings.HeadDim, settings.HeadDim), DType.F32);
        History.AsSpan<float>().Clear();
        Recurrent.AsSpan<float>().Clear();
    }

    /// <summary>Pre-conv rows of the last <c>ConvKernel − 1</c> tokens, <c>[ConvKernel−1, ConvDim]</c>.</summary>
    public Tensor History { get; }

    /// <summary>Delta-rule state, <c>[Heads, HeadDim, HeadDim]</c>.</summary>
    public Tensor Recurrent { get; }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        History.Dispose();
        Recurrent.Dispose();
    }
}

/// <summary>The KDA layer forward pass, composed from shared backend ops.</summary>
public static class KdaLayer
{
    /// <summary>Folds the forget gate's per-channel <c>exp(A)</c> and <c>dt_bias</c> into the low-rank up projection, so the gate
    /// <c>−bound · sigmoid(exp(A)·(W·h + dt_bias))</c> is one <see cref="IBackend.Linear"/> and a sigmoid. Returns the folded
    /// <c>(ForgetUp, ForgetBias)</c> as new host-side F32 arrays in <c>[KeyDim, ForgetRank]</c> and <c>[KeyDim]</c>.</summary>
    /// <param name="forgetUp">F32 <c>[KeyDim, ForgetRank]</c>, the checkpoint's <c>f_b_proj</c>.</param>
    /// <param name="aLog">F32 <c>[KeyDim]</c>, the checkpoint's <c>A_log</c> exponentiated per channel.</param>
    /// <param name="dtBias">F32 <c>[KeyDim]</c>.</param>
    public static (float[] Up, float[] Bias) FoldForgetGate(ReadOnlySpan<float> forgetUp, ReadOnlySpan<float> aLog, ReadOnlySpan<float> dtBias, int keyDim, int forgetRank)
    {
        if (forgetUp.Length != keyDim * forgetRank || aLog.Length != keyDim || dtBias.Length != keyDim)
            throw new ArgumentException($"FoldForgetGate needs forgetUp [{keyDim}, {forgetRank}], aLog and dtBias [{keyDim}].");
        float[] up = new float[forgetUp.Length];
        float[] bias = new float[keyDim];
        for (int c = 0; c < keyDim; c++)
        {
            float scale = MathF.Exp(aLog[c]);
            for (int r = 0; r < forgetRank; r++) up[c * forgetRank + r] = scale * forgetUp[c * forgetRank + r];
            bias[c] = scale * dtBias[c];
        }
        return (up, bias);
    }

    /// <summary>Runs the layer over <paramref name="tokens"/> tokens of <paramref name="input"/> (F32 <c>[tokens, Hidden]</c>) into
    /// <paramref name="output"/>, advancing <paramref name="state"/>.</summary>
    public static void Forward(IBackend backend, KdaSettings s, KdaWeights w, Tensor input, Tensor output, KdaState state, float eps = 1e-6f)
    {
        ArgumentNullException.ThrowIfNull(backend);
        int tokens = checked((int)(input.ElementCount / s.Hidden));
        int keyDim = s.KeyDim;
        float qScale = 1f / MathF.Sqrt(s.HeadDim);

        using Tensor qkv = Alloc(tokens, s.ConvDim);
        backend.Linear(qkv, input, w.QkvProj, null);
        using Tensor q = Alloc(tokens, keyDim);
        using Tensor k = Alloc(tokens, keyDim);
        using Tensor v = Alloc(tokens, keyDim);
        backend.SsmConvSplitPrefill(q, k, v, qkv, state.History, w.ConvTaps, s.ConvKernel, s.Heads, s.HeadDim, s.Heads, s.HeadDim, eps, qScale);

        using Tensor forgetLow = Alloc(tokens, s.ForgetRank);
        backend.Linear(forgetLow, input, w.ForgetDown, null);
        using Tensor forgetPre = Alloc(tokens, keyDim);
        backend.Linear(forgetPre, forgetLow, w.ForgetUp, w.ForgetBias);
        using Tensor forgetSig = Alloc(tokens, keyDim);
        backend.Sigmoid(forgetSig, forgetPre);
        using Tensor logDecay = Alloc(tokens, keyDim);
        backend.Scale(logDecay, forgetSig, s.GateLowerBound);

        using Tensor beta = Alloc(tokens, s.Heads);
        backend.Linear(beta, input, w.BetaProj, null);

        // The recurrence writes [tokens·Heads, HeadDim] rows, the same memory as [tokens, KeyDim], so the per-head norm reads it directly.
        using Tensor core = Alloc(tokens * s.Heads, s.HeadDim);
        backend.SsmChannelDecayPrefill(core, state.Recurrent, q, k, v, logDecay, beta, s.Heads, s.HeadDim, s.HeadDim, 1);
        using Tensor normed = Alloc(tokens * s.Heads, s.HeadDim);
        backend.RmsNorm(normed, core, w.OutNorm, eps);

        using Tensor gatePre = Alloc(tokens, keyDim);
        backend.Linear(gatePre, input, w.OutGateProj, null);
        using Tensor gate = Alloc(tokens, keyDim);
        backend.Sigmoid(gate, gatePre);
        using Tensor gated = Alloc(tokens, keyDim);
        backend.Mul(gated, normed, gate);
        backend.Linear(output, gated, w.OutProj, null);
    }

    private static Tensor Alloc(long rows, long cols) => new(new TensorShape(rows, cols), DType.F32);
}

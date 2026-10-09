namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>One DSpark draft stage (upstream <c>DSparkBlock</c>): hyper-connection mixing around the draft attention and a feed-forward layer over the draft experts,
/// the same block structure as a backbone layer.</summary>
internal sealed class DeepSeekV41DSparkStage
{
    private readonly int _dim;
    private readonly int _hc;
    private readonly float _normEps;
    private readonly DeepSeekV41HyperConnection _hcAttn;
    private readonly DeepSeekV41HyperConnection _hcFfn;
    private readonly float[] _attnNorm;
    private readonly float[] _ffnNorm;
    private readonly DeepSeekV41DSparkAttention _attention;
    private readonly DeepSeekV41MoeLayer _ffn;

    /// <param name="dim">Residual width.</param>
    /// <param name="hc">Residual copies.</param>
    /// <param name="normEps">Config <c>rms_norm_eps</c>.</param>
    /// <param name="hcAttn">Mixing around attention.</param>
    /// <param name="hcFfn">Mixing around the feed-forward layer.</param>
    /// <param name="attnNorm">Norm weight before attention, <c>[dim]</c>.</param>
    /// <param name="ffnNorm">Norm weight before the feed-forward layer, <c>[dim]</c>.</param>
    /// <param name="attention">The draft attention.</param>
    /// <param name="ffn">The draft feed-forward layer.</param>
    public DeepSeekV41DSparkStage(int dim, int hc, float normEps, DeepSeekV41HyperConnection hcAttn, DeepSeekV41HyperConnection hcFfn, float[] attnNorm,
        float[] ffnNorm, DeepSeekV41DSparkAttention attention, DeepSeekV41MoeLayer ffn)
    {
        ArgumentNullException.ThrowIfNull(hcAttn);
        ArgumentNullException.ThrowIfNull(hcFfn);
        ArgumentNullException.ThrowIfNull(attnNorm);
        ArgumentNullException.ThrowIfNull(ffnNorm);
        ArgumentNullException.ThrowIfNull(attention);
        ArgumentNullException.ThrowIfNull(ffn);
        if (attnNorm.Length != dim || ffnNorm.Length != dim) throw new ArgumentException("Norm weights must hold dim values.");
        if (hcAttn.Dim != dim || hcFfn.Dim != dim || hcAttn.Hc != hc || hcFfn.Hc != hc) throw new ArgumentException("Hyper-connections must match dim and hc.");
        if (attention.Settings.Dim != dim) throw new ArgumentException("The attention must match dim.", nameof(attention));
        _dim = dim;
        _hc = hc;
        _normEps = normEps;
        _hcAttn = hcAttn;
        _hcFfn = hcFfn;
        _attnNorm = attnNorm;
        _ffnNorm = ffnNorm;
        _attention = attention;
        _ffn = ffn;
    }

    /// <summary>The attention settings, used to size the stage's window.</summary>
    public DeepSeekV41AttentionSettings AttentionSettings => _attention.Settings;

    /// <summary>Seeds the window from the target's latents of the prompt; <paramref name="mainX"/> is <c>[tokens, Dim]</c>.</summary>
    public void Seed(ReadOnlySpan<float> mainX, int tokens, DeepSeekV41AttentionState state) => _attention.Seed(mainX, tokens, state);

    /// <summary>Runs one draft block through the stage. <paramref name="x"/> is <c>[block, hc, Dim]</c> and is replaced by the output; <paramref name="preMix"/> is the
    /// collapse coefficients from the previous stage <c>[block, hc]</c>; <paramref name="nextPreMix"/> receives this stage's feed-forward coefficients.</summary>
    public void Draft(Span<float> x, int block, ReadOnlySpan<float> preMix, Span<float> nextPreMix, int startPos, ReadOnlySpan<float> mainX,
        DeepSeekV41AttentionState state, Action<string, float[]>? tap = null)
    {
        if (x.Length != (long)block * _hc * _dim) throw new ArgumentException("x must hold block x hc x Dim values.", nameof(x));
        if (preMix.Length != block * _hc || nextPreMix.Length != block * _hc) throw new ArgumentException("Pre-mix arrays must hold block x hc values.");

        float[] aPre = new float[block * _hc], aPost = new float[block * _hc], aComb = new float[block * _hc * _hc];
        float[] fPre = new float[block * _hc], fPost = new float[block * _hc], fComb = new float[block * _hc * _hc];
        float[] sublayerIn = new float[block * _dim], sublayerOut = new float[block * _dim], mid = new float[x.Length];

        _hcAttn.Mixes(x, block, aPre, aPost, aComb);
        // as in the backbone block, the feed-forward layer collapses with the attention coefficients, and the next stage takes the feed-forward ones (upstream's deferred hc_pre)
        _hcAttn.Collapse(x, preMix, block, sublayerIn);
        DeepSeekV41HostMath.RmsNormRows(sublayerIn, _attnNorm, _dim, _normEps);
        _attention.Draft(sublayerIn, block, mainX, startPos, state, sublayerOut);
        tap?.Invoke("attn", sublayerOut.ToArray());
        _hcAttn.Expand(sublayerOut, x, aPost, aComb, block, mid);

        _hcFfn.Mixes(mid, block, fPre, fPost, fComb);
        _hcFfn.Collapse(mid, aPre, block, sublayerIn);
        DeepSeekV41HostMath.RmsNormRows(sublayerIn, _ffnNorm, _dim, _normEps);
        _ffn.Forward(sublayerIn, block, ReadOnlySpan<byte>.Empty, sublayerOut);
        tap?.Invoke("ffn", sublayerOut.ToArray());
        _hcFfn.Expand(sublayerOut, mid, fPost, fComb, block, x);
        tap?.Invoke("x", x.ToArray());

        fPre.CopyTo(nextPreMix);
    }

    /// <summary>Collapses the stage's residual copies with <paramref name="pre"/>, as upstream's <c>hc_pre</c> does before the head.</summary>
    public void Collapse(ReadOnlySpan<float> x, ReadOnlySpan<float> pre, int block, Span<float> y) => _hcFfn.Collapse(x, pre, block, y);
}

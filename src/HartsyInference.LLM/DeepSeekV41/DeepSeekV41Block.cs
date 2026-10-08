using HartsyInference.LLM.DeepSeekV41.Engram;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>One V4.1 transformer block on the host reference path: optional Engram, then attention and the feed-forward layer, each wrapped in hyper-connection mixing.</summary>
/// <remarks>Matches upstream <c>Block.forward</c>. The coefficients a sublayer derives are used by the <i>next</i> sublayer's collapse: attention collapses the stream with the
/// previous block's feed-forward coefficients, and the feed-forward layer collapses with the ones attention just derived. The block hands its own feed-forward coefficients back
/// for the following block.</remarks>
public sealed class DeepSeekV41Block
{
    private readonly int _dim;
    private readonly int _hc;
    private readonly float _normEps;
    private readonly DeepSeekV41HyperConnection _hcAttn;
    private readonly DeepSeekV41HyperConnection _hcFfn;
    private readonly float[] _attnNorm;
    private readonly float[] _ffnNorm;
    private readonly DeepSeekV41Attention _attention;
    private readonly DeepSeekV41MoeLayer _ffn;
    private readonly DeepSeekV41EngramModule? _engram;
    private readonly int _engramSlot;

    /// <summary>This block's attention settings, used to size its sequence state.</summary>
    public DeepSeekV41AttentionSettings AttentionSettings => _attention.Settings;

    /// <summary>Whether this block applies an Engram lookup before its sublayers.</summary>
    public bool HasEngram => _engram is not null;

    /// <param name="dim">Residual width.</param>
    /// <param name="hc">Residual copies.</param>
    /// <param name="normEps">Config <c>rms_norm_eps</c>.</param>
    /// <param name="hcAttn">Mixing around attention.</param>
    /// <param name="hcFfn">Mixing around the feed-forward layer.</param>
    /// <param name="attnNorm">Norm weight before attention, <c>[dim]</c>.</param>
    /// <param name="ffnNorm">Norm weight before the feed-forward layer, <c>[dim]</c>.</param>
    /// <param name="attention">The attention layer.</param>
    /// <param name="ffn">The routed feed-forward layer.</param>
    /// <param name="engram">The Engram module, or null on a layer without one.</param>
    /// <param name="engramSlot">This layer's position in <c>engram_layer_ids</c>, which selects its hash ids.</param>
    public DeepSeekV41Block(int dim, int hc, float normEps, DeepSeekV41HyperConnection hcAttn, DeepSeekV41HyperConnection hcFfn, float[] attnNorm,
        float[] ffnNorm, DeepSeekV41Attention attention, DeepSeekV41MoeLayer ffn, DeepSeekV41EngramModule? engram = null, int engramSlot = 0)
    {
        ArgumentNullException.ThrowIfNull(hcAttn);
        ArgumentNullException.ThrowIfNull(hcFfn);
        ArgumentNullException.ThrowIfNull(attnNorm);
        ArgumentNullException.ThrowIfNull(ffnNorm);
        ArgumentNullException.ThrowIfNull(attention);
        ArgumentNullException.ThrowIfNull(ffn);
        if (attnNorm.Length != dim || ffnNorm.Length != dim) throw new ArgumentException("Norm weights must hold dim values.");
        if (hcAttn.Dim != dim || hcFfn.Dim != dim || hcAttn.Hc != hc || hcFfn.Hc != hc) throw new ArgumentException("Hyper-connections must match dim and hc.");
        if (attention.Settings.Dim != dim) throw new ArgumentException("The attention layer must match dim.", nameof(attention));
        if (engram is not null && (engram.Dim != dim || engram.HcMult != hc)) throw new ArgumentException("The Engram module must match dim and hc.", nameof(engram));
        _dim = dim;
        _hc = hc;
        _normEps = normEps;
        _hcAttn = hcAttn;
        _hcFfn = hcFfn;
        _attnNorm = attnNorm;
        _ffnNorm = ffnNorm;
        _attention = attention;
        _ffn = ffn;
        _engram = engram;
        _engramSlot = engramSlot;
    }

    private Action<string, float[]>? _probe;

    /// <summary>Diagnostic tap for oracle comparisons: called with a stage name (<c>attn_in</c>, <c>attn_out</c>, <c>ffn_in</c>, <c>ffn_out</c>, <c>out</c>, and <c>route</c> with the chosen expert ids as values) and a copy of that stage's values. Null in normal use.</summary>
    /// <remarks>Setting a probe replaces any earlier one. The routing stage reports through this block's MoE layer, which holds one probe.</remarks>
    public Action<string, float[]>? Probe
    {
        get => _probe;
        set
        {
            _probe = value;
            _ffn.RouteProbe = value is null ? null : ids => value("route", Array.ConvertAll(ids, id => (float)id));
        }
    }

    /// <summary>Collapses a stream into one input with already-derived coefficients; the model uses this for the final head.</summary>
    public void Collapse(ReadOnlySpan<float> x, ReadOnlySpan<float> pre, int tokens, Span<float> y) => _hcFfn.Collapse(x, pre, tokens, y);

    /// <summary>Runs the block without the DSpark target tap; see the overload that takes <paramref name="streamMean"/>.</summary>
    public void Forward(Span<float> x, int tokens, int startPos, ReadOnlySpan<float> preMix, Span<float> nextPreMix, DeepSeekV41AttentionState state,
        DeepSeekV41SharedAttention shared, ReadOnlySpan<long> hashIds, int hashLayers, ReadOnlySpan<bool> tokenMask, ReadOnlySpan<byte> imageTokens)
        => Forward(x, tokens, startPos, preMix, nextPreMix, state, shared, hashIds, hashLayers, tokenMask, imageTokens, default);

    /// <summary>Runs the block over <paramref name="tokens"/> positions, updating the stream in place.</summary>
    /// <param name="x">Residual stream, <c>[tokens, hc, dim]</c>; replaced by the block's output.</param>
    /// <param name="tokens">Position count.</param>
    /// <param name="startPos">Absolute position of the first row.</param>
    /// <param name="preMix">Collapse coefficients for attention, from the previous block, <c>[tokens, hc]</c>.</param>
    /// <param name="nextPreMix">Receives this block's feed-forward coefficients for the next block, <c>[tokens, hc]</c>.</param>
    /// <param name="state">This layer's attention cache for the sequence.</param>
    /// <param name="shared">Slots handed between attention layers.</param>
    /// <param name="hashIds">Engram hash ids from the hasher, <c>[tokens, hashLayers, columns]</c>; empty when no layer has Engram.</param>
    /// <param name="hashLayers">Engram layers in <paramref name="hashIds"/>.</param>
    /// <param name="tokenMask">False shuts the Engram gate for that position; empty means every position is live.</param>
    /// <param name="imageTokens">Nonzero inside an image span, selecting the vision routing bias; empty when none.</param>
    /// <param name="streamMean">Receives the hc-mean of <paramref name="x"/>, <c>[tokens, dim]</c>, taken after this block's Engram step and before its sublayers, which is where the
    /// DSpark draft reads its target layers; empty when not wanted.</param>
    public void Forward(Span<float> x, int tokens, int startPos, ReadOnlySpan<float> preMix, Span<float> nextPreMix, DeepSeekV41AttentionState state,
        DeepSeekV41SharedAttention shared, ReadOnlySpan<long> hashIds, int hashLayers, ReadOnlySpan<bool> tokenMask, ReadOnlySpan<byte> imageTokens, Span<float> streamMean)
    {
        if (x.Length != (long)tokens * _hc * _dim) throw new ArgumentException("x must hold tokens x hc x dim values.", nameof(x));
        if (preMix.Length != tokens * _hc || nextPreMix.Length != tokens * _hc) throw new ArgumentException("Pre-mix arrays must hold tokens x hc values.");

        if (_engram is not null)
        {
            int columns = _engram.Columns;
            if (hashIds.Length != (long)tokens * hashLayers * columns || _engramSlot >= hashLayers)
                throw new ArgumentException("hashIds must hold tokens x hashLayers x columns values and cover this layer's slot.", nameof(hashIds));
            long[] mine = new long[tokens * columns];
            for (int t = 0; t < tokens; t++) hashIds.Slice((t * hashLayers + _engramSlot) * columns, columns).CopyTo(mine.AsSpan(t * columns, columns));
            _engram.Apply(x, tokens, mine, tokenMask);
        }

        if (!streamMean.IsEmpty)
        {
            if (streamMean.Length != tokens * _dim) throw new ArgumentException("streamMean must hold tokens x dim values.", nameof(streamMean));
            // upstream takes h.mean(dim=2) of the stream entering the layer, after Engram and before any sublayer
            for (int t = 0; t < tokens; t++)
            {
                for (int d = 0; d < _dim; d++)
                {
                    float sum = 0f;
                    for (int c = 0; c < _hc; c++) sum += x[(t * _hc + c) * _dim + d];
                    streamMean[t * _dim + d] = sum / _hc;
                }
            }
        }

        float[] aPre = new float[tokens * _hc], aPost = new float[tokens * _hc], aComb = new float[tokens * _hc * _hc];
        float[] fPre = new float[tokens * _hc], fPost = new float[tokens * _hc], fComb = new float[tokens * _hc * _hc];
        float[] sublayerIn = new float[tokens * _dim], sublayerOut = new float[tokens * _dim], mid = new float[x.Length];

        _hcAttn.Mixes(x, tokens, aPre, aPost, aComb);
        _hcAttn.Collapse(x, preMix, tokens, sublayerIn);
        DeepSeekV41HostMath.RmsNormRows(sublayerIn, _attnNorm, _dim, _normEps);
        Probe?.Invoke("attn_in", sublayerIn.ToArray());
        _attention.Forward(sublayerIn, tokens, startPos, state, shared, sublayerOut);
        Probe?.Invoke("attn_out", sublayerOut.ToArray());
        _hcAttn.Expand(sublayerOut, x, aPost, aComb, tokens, mid);

        _hcFfn.Mixes(mid, tokens, fPre, fPost, fComb);
        _hcFfn.Collapse(mid, aPre, tokens, sublayerIn);
        DeepSeekV41HostMath.RmsNormRows(sublayerIn, _ffnNorm, _dim, _normEps);
        Probe?.Invoke("ffn_in", sublayerIn.ToArray());
        _ffn.Forward(sublayerIn, tokens, imageTokens, sublayerOut);
        Probe?.Invoke("ffn_out", sublayerOut.ToArray());
        _hcFfn.Expand(sublayerOut, mid, fPost, fComb, tokens, x);
        Probe?.Invoke("out", x.ToArray());

        fPre.CopyTo(nextPreMix);
    }
}

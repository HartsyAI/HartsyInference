using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;

/// <summary>Spatial transformer block for UNet cross-attention. Wraps N BasicTransformerBlocks (each: SelfAttn→CrossAttn→FFN). SD1.5 uses 1 block, SDXL uses [1,2,10] per level.</summary>
public sealed class CrossAttentionBlock
{
    private readonly int _channels;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _crossAttentionDim;
    private readonly int _numTransformerBlocks;

    /// <summary>How many cross-attention layers this block contains. Equal to the BasicTransformerBlock count — each transformer block has one cross-attention sub-layer that an IP-Adapter injects K_ip/V_ip into.</summary>
    public int NumTransformerBlocks => _numTransformerBlocks;

    // Input projection: GroupNorm + linear proj_in (or conv 1x1)
    private Tensor? _normWeight;
    private Tensor? _normBias;
    private Tensor? _projInWeight;
    private Tensor? _projInBias;

    // Multiple transformer blocks (SDXL can have up to 10 per spatial transformer)
    private readonly TransformerSubBlock[] _selfAttns;
    private readonly TransformerSubBlock[] _crossAttns;
    private readonly FeedForwardBlock[] _ffns;

    // Output projection
    private Tensor? _projOutWeight;
    private Tensor? _projOutBias;

    /// <summary>Creates a cross-attention block with N transformer layers.</summary>
    /// <param name="channels">Number of input/output channels.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="crossAttentionDim">Dimension of the cross-attention context (text encoder hidden size).</param>
    /// <param name="numTransformerBlocks">Number of BasicTransformerBlocks. SD1.5=1, SDXL=[1,2,10] per level.</param>
    /// <param name="normGroups">Number of groups for the input GroupNorm.</param>
    /// <param name="normEps">GroupNorm epsilon.</param>
    public CrossAttentionBlock(int channels, int numHeads, int crossAttentionDim, int numTransformerBlocks = 1, int normGroups = 32, float normEps = 1e-5f)
    {
        _channels = channels;
        _numHeads = numHeads;
        _headDim = channels / numHeads;
        _crossAttentionDim = crossAttentionDim;
        _numTransformerBlocks = numTransformerBlocks;

        _selfAttns = new TransformerSubBlock[numTransformerBlocks];
        _crossAttns = new TransformerSubBlock[numTransformerBlocks];
        _ffns = new FeedForwardBlock[numTransformerBlocks];

        for (int i = 0; i < numTransformerBlocks; i++)
        {
            _selfAttns[i] = new TransformerSubBlock(channels, numHeads, channels);
            _crossAttns[i] = new TransformerSubBlock(channels, numHeads, crossAttentionDim);
            _ffns[i] = new FeedForwardBlock(channels);
        }
    }

    /// <summary>Loads weights from named tensors.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _normWeight = weights[$"{prefix}.norm.weight"];
        _normBias = weights[$"{prefix}.norm.bias"];
        _projInWeight = weights[$"{prefix}.proj_in.weight"];
        _projInBias = weights[$"{prefix}.proj_in.bias"];

        for (int i = 0; i < _numTransformerBlocks; i++)
        {
            _selfAttns[i].LoadWeights(weights, $"{prefix}.transformer_blocks.{i}.attn1", $"{prefix}.transformer_blocks.{i}.norm1");
            _crossAttns[i].LoadWeights(weights, $"{prefix}.transformer_blocks.{i}.attn2", $"{prefix}.transformer_blocks.{i}.norm2");
            _ffns[i].LoadWeights(weights, $"{prefix}.transformer_blocks.{i}.ff", $"{prefix}.transformer_blocks.{i}.norm3");
        }

        _projOutWeight = weights[$"{prefix}.proj_out.weight"];
        _projOutBias = weights[$"{prefix}.proj_out.bias"];
    }

    /// <summary>Enumerates all weight tensors held by this block and its sub-blocks.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_normWeight is not null) yield return _normWeight;
        if (_normBias is not null) yield return _normBias;
        if (_projInWeight is not null) yield return _projInWeight;
        if (_projInBias is not null) yield return _projInBias;
        for (int i = 0; i < _numTransformerBlocks; i++)
        {
            foreach (Tensor w in _selfAttns[i].EnumerateWeights()) yield return w;
            foreach (Tensor w in _crossAttns[i].EnumerateWeights()) yield return w;
            foreach (Tensor w in _ffns[i].EnumerateWeights()) yield return w;
        }
        if (_projOutWeight is not null) yield return _projOutWeight;
        if (_projOutBias is not null) yield return _projOutBias;
    }

    /// <summary>Forward pass: input [B, C, H, W] + context [B, seqLen, crossDim] → output [B, C, H, W]. No IPA injection.</summary>
    public Tensor Forward(IBackend backend, Tensor input, Tensor context)
    {
        return Forward(backend, input, context, ipaImageTokens: null, ipaToKIpAll: null, ipaToVIpAll: null, ipaStartIndex: 0, ipaScalePerLayer: null);
    }

    /// <summary>Forward with optional IP-Adapter image-attention injection. The shared image-prompt tokens (<paramref name="ipaImageTokens"/>) are injected into each cross-attention sub-layer with its own per-layer <c>to_k_ip</c> / <c>to_v_ip</c> projection, indexed from the flat checkpoint list starting at <paramref name="ipaStartIndex"/>. The block consumes <see cref="NumTransformerBlocks"/> entries from the K/V lists. Per-layer IPA strength comes from <paramref name="ipaScalePerLayer"/> at index <c>ipaStartIndex + i</c> for sub-block <c>i</c>; a scale of 0 short-circuits the image attention entirely (used for step-fraction gating and weight-type schedules that zero out specific blocks).</summary>
    public Tensor Forward(IBackend backend, Tensor input, Tensor context,
        Tensor? ipaImageTokens,
        IReadOnlyList<Tensor>? ipaToKIpAll,
        IReadOnlyList<Tensor>? ipaToVIpAll,
        int ipaStartIndex,
        IReadOnlyList<float>? ipaScalePerLayer)
    {
        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];
        int spatial = height * width;

        DType dtype = input.DType;
        bool hasIpa = ipaImageTokens is not null && ipaToKIpAll is not null && ipaToVIpAll is not null && ipaScalePerLayer is not null;

        // 1. GroupNorm on spatial input
        TensorShape spatialShape = new TensorShape(batch, channels, height, width);
        Tensor normed = new Tensor(spatialShape, dtype);
        backend.GroupNorm(normed, input, _normWeight!, _normBias!, 32, 1e-6f);

        // 2. [B, C, H, W] → [B, spatial, C] via batched transpose and project in. Transpose2D takes
        // explicit dims and reads the buffer flat, so normed is passed directly — a Reshape view here
        // would touch DataPointer, forcing a GPU→host sync + re-upload of the activation (the profiled
        // dominant SDXL step cost: ~457 multi-MB H2D misses per step across the block stack).
        TensorShape seqShape = new TensorShape(batch, spatial, channels);
        Tensor hidden = new Tensor(seqShape, dtype);
        backend.Transpose2D(hidden, normed, channels, spatial);
        normed.Dispose();

        // proj_in: [B, spatial, C] → [B, spatial, C]
        // When SDXL (useLinearProjection=true), weights are [C, C] (linear)
        // When SD1.5, weights are [C, C, 1, 1] (1x1 conv) but math is identical for proj_in/proj_out
        Tensor projected = new Tensor(seqShape, dtype);
        backend.Linear(projected, hidden, _projInWeight!, _projInBias!);
        hidden.Dispose();
        hidden = projected;

        // 3. Run N transformer blocks (SD1.5: 1, SDXL level 2: 10).
        //    IPA, when supplied, picks per-cross-attn-layer K/V from the flat checkpoint list
        //    starting at ipaStartIndex; the i-th transformer block uses entries
        //    ipaToK/VIpAll[ipaStartIndex + i].
        for (int i = 0; i < _numTransformerBlocks; i++)
        {
            // Self-attention (context = self) — never gets IPA injection.
            Tensor selfOut = _selfAttns[i].Forward(backend, hidden, hidden);
            hidden.Dispose();
            hidden = selfOut;

            // Cross-attention (context = text embeddings) — IPA hooks here.
            Tensor? perLayerKIp = hasIpa ? ipaToKIpAll![ipaStartIndex + i] : null;
            Tensor? perLayerVIp = hasIpa ? ipaToVIpAll![ipaStartIndex + i] : null;
            float perLayerScale = hasIpa ? ipaScalePerLayer![ipaStartIndex + i] : 0f;
            Tensor crossOut = _crossAttns[i].Forward(backend, hidden, context,
                ipaImageTokens, perLayerKIp, perLayerVIp, perLayerScale);
            hidden.Dispose();
            hidden = crossOut;

            // Feed-forward
            Tensor ffnOut = _ffns[i].Forward(backend, hidden);
            hidden.Dispose();
            hidden = ffnOut;
        }

        // 4. proj_out: [B, spatial, C] → [B, spatial, C]
        Tensor projOut = new Tensor(seqShape, dtype);
        backend.Linear(projOut, hidden, _projOutWeight!, _projOutBias!);
        hidden.Dispose();

        // 5. Transpose [B, spatial, C] → [B, C, H, W] and add residual. The output tensor is allocated
        // with the final 4-D shape directly (shape is metadata; Transpose2D writes the buffer flat) —
        // no Reshape, no device→host round-trip.
        Tensor output = new Tensor(spatialShape, dtype);
        backend.Transpose2D(output, projOut, spatial, channels);
        projOut.Dispose();

        // Add residual from input
        Tensor result = new Tensor(spatialShape, dtype);
        backend.Add(result, output, input);
        output.Dispose();

        return result;
    }

}

/// <summary>Attention sub-block: LayerNorm → MultiHeadAttention → Residual. Used for both self-attention and cross-attention.</summary>
internal sealed unsafe class TransformerSubBlock
{
    private readonly int _channels;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _contextDim;

    // LayerNorm
    private Tensor? _normWeight;
    private Tensor? _normBias;

    // Q from hidden, K/V from context
    private Tensor? _toQWeight;
    private Tensor? _toQBias;
    private Tensor? _toKWeight;
    private Tensor? _toKBias;
    private Tensor? _toVWeight;
    private Tensor? _toVBias;
    private Tensor? _toOutWeight;
    private Tensor? _toOutBias;

    // Cross-attention K/V cache: the projections of the text conditioning are step-invariant (the
    // context tensor is fixed across the denoise loop), so the k/v Linears + head-split permutes run
    // once per generation instead of once per step (~140 GEMMs + 140 permutes/step on SDXL). Keyed on
    // the context tensor reference — the fused loop passes the same pre-cast conditioning object every
    // step; per-step slicing callers (legacy CFG loop) miss every call and behave exactly as before.
    private Tensor? _crossKvContextRef;
    private Tensor? _cachedCrossKeyMh;
    private Tensor? _cachedCrossValueMh;

    public TransformerSubBlock(int channels, int numHeads, int contextDim)
    {
        _channels = channels;
        _numHeads = numHeads;
        _headDim = channels / numHeads;
        _contextDim = contextDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string attnPrefix, string normPrefix)
    {
        _normWeight = weights[$"{normPrefix}.weight"];
        _normBias = weights[$"{normPrefix}.bias"];
        _toQWeight = weights[$"{attnPrefix}.to_q.weight"];
        _toKWeight = weights[$"{attnPrefix}.to_k.weight"];
        _toVWeight = weights[$"{attnPrefix}.to_v.weight"];
        _toOutWeight = weights[$"{attnPrefix}.to_out.0.weight"];
        _toOutBias = weights[$"{attnPrefix}.to_out.0.bias"];

        // Q bias and K/V bias may not exist in some models — use zero if absent
        weights.TryGetValue($"{attnPrefix}.to_q.bias", out _toQBias);
        weights.TryGetValue($"{attnPrefix}.to_k.bias", out _toKBias);
        weights.TryGetValue($"{attnPrefix}.to_v.bias", out _toVBias);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_normWeight is not null) yield return _normWeight;
        if (_normBias is not null) yield return _normBias;
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toQBias is not null) yield return _toQBias;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toKBias is not null) yield return _toKBias;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_toVBias is not null) yield return _toVBias;
        if (_toOutWeight is not null) yield return _toOutWeight;
        if (_toOutBias is not null) yield return _toOutBias;
    }

    /// <summary>Forward: hidden [B, seqLen, C] + context [B, ctxLen, ctxDim] → output [B, seqLen, C] with residual. No IPA injection.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor context)
    {
        return Forward(backend, hidden, context, ipaImageTokens: null, toKIp: null, toVIp: null, ipaScale: 0f);
    }

    /// <summary>Forward with optional IP-Adapter image attention injection. When <paramref name="ipaImageTokens"/> + <paramref name="toKIp"/> + <paramref name="toVIp"/> are all non-null, a parallel attention pass is computed with the same Q (from <paramref name="hidden"/>) but K/V derived from the projected image tokens. The image attention output is added to the text attention output (scaled by <paramref name="ipaScale"/>) before the shared <c>to_out</c> projection — matching diffusers' <c>IPAdapterAttnProcessor.__call__</c> exactly.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor context,
        Tensor? ipaImageTokens, Tensor? toKIp, Tensor? toVIp, float ipaScale)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];
        int ctxLen = (int)context.Shape[1];

        DType dtype = hidden.DType;
        bool hasIpa = ipaImageTokens is not null && toKIp is not null && toVIp is not null && ipaScale != 0f;

        // LayerNorm
        TensorShape hidShape = new TensorShape(batch, seqLen, _channels);
        Tensor normed = new Tensor(hidShape, dtype);
        backend.LayerNorm(normed, hidden, _normWeight!, _normBias!, 1e-5f);

        // Q from normed hidden; K/V from normed hidden (self-attn) or raw context (cross-attn)
        bool isSelfAttn = ReferenceEquals(hidden, context);
        Tensor kvSource = isSelfAttn ? normed : context;
        int kvSeqLen = isSelfAttn ? seqLen : ctxLen;

        Tensor query = new Tensor(hidShape, dtype);
        backend.Linear(query, normed, _toQWeight!, _toQBias);

        // Multi-head split shapes. Permute0213 takes the (seq, heads, headDim) split explicitly and
        // reads the buffer flat, so the projections are passed directly — Reshape views would each
        // force a GPU→host sync + re-upload of a multi-MB activation (see CrossAttentionBlock.Forward).
        TensorShape qMhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);
        TensorShape kvMhShape = new TensorShape(batch, _numHeads, kvSeqLen, _headDim);

        // Cross-attention K/V: step-invariant per generation (see cache fields) — reuse when the same
        // context tensor comes back; the cache owns the reused tensors' lifetime.
        bool kvFromCache = !isSelfAttn && ReferenceEquals(_crossKvContextRef, context)
            && _cachedCrossKeyMh is not null && _cachedCrossValueMh is not null;
        Tensor keyMh, valueMh;
        if (kvFromCache)
        {
            keyMh = _cachedCrossKeyMh!;
            valueMh = _cachedCrossValueMh!;
        }
        else
        {
            TensorShape kvOutShape = new TensorShape(batch, kvSeqLen, _channels);
            Tensor key = new Tensor(kvOutShape, dtype);
            backend.Linear(key, kvSource, _toKWeight!, _toKBias);
            Tensor value = new Tensor(kvOutShape, dtype);
            backend.Linear(value, kvSource, _toVWeight!, _toVBias);

            keyMh = new Tensor(kvMhShape, dtype);
            backend.Permute0213(keyMh, key, kvSeqLen, _numHeads, _headDim);
            key.Dispose();
            valueMh = new Tensor(kvMhShape, dtype);
            backend.Permute0213(valueMh, value, kvSeqLen, _numHeads, _headDim);
            value.Dispose();

            if (!isSelfAttn)
            {
                _cachedCrossKeyMh?.Dispose();
                _cachedCrossValueMh?.Dispose();
                _cachedCrossKeyMh = keyMh;
                _cachedCrossValueMh = valueMh;
                _crossKvContextRef = context;
                kvFromCache = true;
            }
        }
        normed.Dispose();

        Tensor queryMh = new Tensor(qMhShape, dtype);
        backend.Permute0213(queryMh, query, seqLen, _numHeads, _headDim);
        query.Dispose();

        // SDPA — text branch. allowF16: SDXL/SD1.5 attention is universally run in fp16 by the reference
        // engines (the checkpoints themselves ship F16); when this UNet loads F32 the flag lets the
        // cuDNN fused path engage via the F32↔F16 cast route instead of materializing [H,Sq,Skv] scores.
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(qMhShape, dtype);
        backend.ScaledDotProductAttention(attnOut, queryMh, keyMh, valueMh, null, scale, allowF16: true);
        if (!kvFromCache)
        {
            keyMh.Dispose();
            valueMh.Dispose();
        }

        // SDPA — image branch (IP-Adapter). Reuses queryMh; produces attention against
        // image-prompt tokens projected through to_k_ip / to_v_ip. Result is added on top of
        // the text attention before to_out, scaled by ipaScale (the user's "IP-Adapter strength").
        if (hasIpa)
        {
            int imgSeqLen = (int)ipaImageTokens!.Shape[1];
            // Project to_k_ip and to_v_ip — shape [_channels, contextDim], no bias.
            Tensor ipKeyFlat = new Tensor(new TensorShape(batch, imgSeqLen, _channels), dtype);
            backend.Linear(ipKeyFlat, ipaImageTokens, toKIp!, null);
            Tensor ipValueFlat = new Tensor(new TensorShape(batch, imgSeqLen, _channels), dtype);
            backend.Linear(ipValueFlat, ipaImageTokens, toVIp!, null);

            // Multi-head split: [B, imgSeqLen, H*D] → [B, H, imgSeqLen, D] (direct, no Reshape views)
            TensorShape ipMhShape = new TensorShape(batch, _numHeads, imgSeqLen, _headDim);
            Tensor ipKeyMh = new Tensor(ipMhShape, dtype);
            backend.Permute0213(ipKeyMh, ipKeyFlat, imgSeqLen, _numHeads, _headDim);
            ipKeyFlat.Dispose();

            Tensor ipValueMh = new Tensor(ipMhShape, dtype);
            backend.Permute0213(ipValueMh, ipValueFlat, imgSeqLen, _numHeads, _headDim);
            ipValueFlat.Dispose();

            // Image attention against the same Q
            Tensor imgAttnOut = new Tensor(qMhShape, dtype);
            backend.ScaledDotProductAttention(imgAttnOut, queryMh, ipKeyMh, ipValueMh, null, scale, allowF16: true);
            ipKeyMh.Dispose();
            ipValueMh.Dispose();

            // attnOut += ipaScale * imgAttnOut
            AccumulateScaled(attnOut, imgAttnOut, ipaScale);
            imgAttnOut.Dispose();
        }
        queryMh.Dispose();

        // Permute back — allocated as the final [B, seqLen, C] merged shape directly (no Reshape).
        Tensor merged = new Tensor(hidShape, dtype);
        backend.Permute0213(merged, attnOut, _numHeads, seqLen, _headDim);
        attnOut.Dispose();

        // Output projection
        Tensor projected = new Tensor(hidShape, dtype);
        backend.Linear(projected, merged, _toOutWeight!, _toOutBias);
        merged.Dispose();

        // Residual
        Tensor output = new Tensor(hidShape, dtype);
        backend.Add(output, hidden, projected);
        projected.Dispose();

        return output;
    }

    /// <summary>In-place accumulate: <c>target += scale * addend</c>. Both tensors must have identical shapes and dtypes. Used by IPA to fold image-attention output into the text-attention output before the shared <c>to_out</c> projection. Handled in F32 / F16 directly to avoid an extra backend op for what amounts to a fused-multiply-add over the attention output buffer.</summary>
    private static void AccumulateScaled(Tensor target, Tensor addend, float scale)
    {
        long count = target.ElementCount;
        if (target.DType == DType.F32)
        {
            float* tp = (float*)target.DataPointer;
            float* ap = (float*)addend.DataPointer;
            for (long i = 0; i < count; i++) tp[i] += scale * ap[i];
        }
        else if (target.DType == DType.F16)
        {
            // F16 stored as ushort half-precision; round-trip through F32 for the accumulate
            // to avoid catastrophic precision loss on small contributions.
            ushort* tp = (ushort*)target.DataPointer;
            ushort* ap = (ushort*)addend.DataPointer;
            for (long i = 0; i < count; i++)
            {
                float t = (float)BitConverter.UInt16BitsToHalf(tp[i]);
                float a = (float)BitConverter.UInt16BitsToHalf(ap[i]);
                tp[i] = BitConverter.HalfToUInt16Bits((Half)(t + scale * a));
            }
        }
        else
        {
            throw new NotSupportedException($"AccumulateScaled doesn't support dtype {target.DType}.");
        }
    }

}

/// <summary>Feed-forward network: LayerNorm → Linear → GEGLU → Linear → Residual.</summary>
internal sealed class FeedForwardBlock
{
    private readonly int _channels;

    // LayerNorm
    private Tensor? _normWeight;
    private Tensor? _normBias;

    // GEGLU: net.0.proj (projects to 2 * innerDim for gating)
    private Tensor? _geGluProjWeight;
    private Tensor? _geGluProjBias;

    // Output linear: net.2
    private Tensor? _outLinearWeight;
    private Tensor? _outLinearBias;

    public FeedForwardBlock(int channels)
    {
        _channels = channels;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string ffPrefix, string normPrefix)
    {
        _normWeight = weights[$"{normPrefix}.weight"];
        _normBias = weights[$"{normPrefix}.bias"];
        _geGluProjWeight = weights[$"{ffPrefix}.net.0.proj.weight"];
        _geGluProjBias = weights[$"{ffPrefix}.net.0.proj.bias"];
        _outLinearWeight = weights[$"{ffPrefix}.net.2.weight"];
        _outLinearBias = weights[$"{ffPrefix}.net.2.bias"];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_normWeight is not null) yield return _normWeight;
        if (_normBias is not null) yield return _normBias;
        if (_geGluProjWeight is not null) yield return _geGluProjWeight;
        if (_geGluProjBias is not null) yield return _geGluProjBias;
        if (_outLinearWeight is not null) yield return _outLinearWeight;
        if (_outLinearBias is not null) yield return _outLinearBias;
    }

    /// <summary>Forward: hidden [B, seqLen, C] → output [B, seqLen, C] with residual.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];

        DType dtype = hidden.DType;

        // LayerNorm
        TensorShape shape = new TensorShape(batch, seqLen, _channels);
        Tensor normed = new Tensor(shape, dtype);
        backend.LayerNorm(normed, hidden, _normWeight!, _normBias!, 1e-5f);

        // GEGLU: project to 2*innerDim, split, gate
        // innerDim = channels * 4 (diffusers default mult=4)
        int innerDim = _channels * 4;
        int geGluOutDim = innerDim * 2;

        TensorShape geGluShape = new TensorShape(batch, seqLen, geGluOutDim);
        Tensor geGluOut = new Tensor(geGluShape, dtype);
        backend.Linear(geGluOut, normed, _geGluProjWeight!, _geGluProjBias!);
        normed.Dispose();

        // Split and apply GELU gate via backend: output = x[:innerDim] * GELU(x[innerDim:])
        TensorShape innerShape = new TensorShape(batch, seqLen, innerDim);
        Tensor gated = new Tensor(innerShape, dtype);
        backend.GeGlu(gated, geGluOut);
        geGluOut.Dispose();

        // Output linear: [B, seqLen, innerDim] → [B, seqLen, channels]
        Tensor outLinear = new Tensor(shape, dtype);
        backend.Linear(outLinear, gated, _outLinearWeight!, _outLinearBias!);
        gated.Dispose();

        // Residual
        Tensor output = new Tensor(shape, dtype);
        backend.Add(output, hidden, outLinear);
        outLinear.Dispose();

        return output;
    }

}

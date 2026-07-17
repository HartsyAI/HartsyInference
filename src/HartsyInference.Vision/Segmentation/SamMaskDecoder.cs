using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Segmentation.Sam2;

namespace HartsyInference.Vision.Segmentation;

/// <summary>SAM / SAM 2 mask decoder: a two-way transformer that mixes learned output tokens (one IoU token +
/// <c>N+1</c> mask tokens) with the sparse prompt tokens and the image embedding, then produces low-resolution
/// mask logits and per-mask IoU (quality) scores.
///
/// <para>Flow (matching Meta's <c>MaskDecoder.predict_masks</c>): prepend the IoU + mask tokens to the sparse
/// prompt embeddings; add the dense prompt embedding to the image embedding as the transformer's image input;
/// run <see cref="Sam2Config.DecoderDepth"/> <see cref="TwoWayAttentionBlock"/> layers plus a final
/// token→image attention; upscale the transformed image embedding 4× with two transposed convs; and combine a
/// per-mask hypernetwork MLP output with the upscaled embedding to form each mask. Keys follow
/// <c>sam_mask_decoder.*</c> (<c>iou_token</c>, <c>mask_tokens</c>, <c>transformer.*</c>,
/// <c>output_upscaling.{0,1,3}</c>, <c>output_hypernetworks_mlps.{i}.*</c>, <c>iou_prediction_head.*</c>).</para></summary>
public sealed unsafe class SamMaskDecoder
{
    private const float LnEps = 1e-6f;

    private readonly int _transformerDim;
    private readonly int _numMaskTokens;
    private readonly bool _iouUseSigmoid;

    private readonly TwoWayAttentionBlock[] _layers;
    private readonly DecoderAttention _finalAttn;
    private readonly SamMlpHead[] _hyperMlps;
    private readonly SamMlpHead _iouHead;

    private Tensor? _iouToken;      // [1, dim]
    private Tensor? _maskTokens;    // [numMaskTokens, dim]
    private Tensor? _objScoreToken; // [1, dim]  (SAM 2.1 pred_obj_scores; prepended before the iou token)
    private Tensor? _normFinalW, _normFinalB;
    private Tensor? _upConv0W, _upConv0B;   // ConvTranspose2d [dim, dim/4, 2, 2]
    private Tensor? _upLnW, _upLnB;         // LayerNorm2d over dim/4 channels
    private Tensor? _upConv1W, _upConv1B;   // ConvTranspose2d [dim/4, dim/8, 2, 2]
    private Tensor? _convS0W, _convS0B;     // 1×1 conv on the stride-4 FPN level → dim/8 ch (high-res feat_s0)
    private Tensor? _convS1W, _convS1B;     // 1×1 conv on the stride-8 FPN level → dim/4 ch (high-res feat_s1)

    /// <summary>Number of learned mask tokens (<see cref="Sam2Config.NumMultimaskOutputs"/> + 1 ambiguity-free token).</summary>
    public int NumMaskTokens => _numMaskTokens;

    /// <summary>Builds the decoder structure from the config; weights load via <see cref="LoadWeights"/>.</summary>
    public SamMaskDecoder(Sam2Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _transformerDim = config.PromptEmbedDim;
        _numMaskTokens = config.NumMultimaskOutputs + 1;
        _iouUseSigmoid = config.IouPredictionUseSigmoid;

        _layers = new TwoWayAttentionBlock[config.DecoderDepth];
        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i] = new TwoWayAttentionBlock(config.DecoderNumHeads, skipFirstLayerPe: i == 0, LnEps);
        }
        _finalAttn = new DecoderAttention(config.DecoderNumHeads);
        _hyperMlps = new SamMlpHead[_numMaskTokens];
        for (int i = 0; i < _numMaskTokens; i++) _hyperMlps[i] = new SamMlpHead();
        _iouHead = new SamMlpHead();
    }

    /// <summary>Loads the decoder weights (tokens, transformer, upscaling head, hypernetworks) under <paramref name="prefix"/>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        ArgumentNullException.ThrowIfNull(weights);
        string p = prefix ?? string.Empty;

        _iouToken = F32(weights, $"{p}iou_token.weight");
        _maskTokens = F32(weights, $"{p}mask_tokens.weight");
        // SAM 2.1 prepends an object-score token (pred_obj_scores); optional for older/synthetic checkpoints.
        if (weights.TryGetValue($"{p}obj_score_token.weight", out Tensor? ost))
            _objScoreToken = ost.DType != DType.F32 ? ost.CastTo(DType.F32) : ost;
        // High-res feature convs (use_high_res_features_in_sam); optional.
        if (weights.TryGetValue($"{p}conv_s0.weight", out Tensor? cs0w))
        {
            _convS0W = cs0w.DType != DType.F32 ? cs0w.CastTo(DType.F32) : cs0w;
            _convS0B = F32(weights, $"{p}conv_s0.bias");
            _convS1W = F32(weights, $"{p}conv_s1.weight");
            _convS1B = F32(weights, $"{p}conv_s1.bias");
        }

        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].LoadWeights(weights, $"{p}transformer.layers.{i}.");
        }
        _finalAttn.LoadWeights(weights, $"{p}transformer.final_attn_token_to_image.");
        _normFinalW = F32(weights, $"{p}transformer.norm_final_attn.weight");
        _normFinalB = F32(weights, $"{p}transformer.norm_final_attn.bias");

        _upConv0W = F32(weights, $"{p}output_upscaling.0.weight");
        _upConv0B = F32(weights, $"{p}output_upscaling.0.bias");
        _upLnW = F32(weights, $"{p}output_upscaling.1.weight");
        _upLnB = F32(weights, $"{p}output_upscaling.1.bias");
        _upConv1W = F32(weights, $"{p}output_upscaling.3.weight");
        _upConv1B = F32(weights, $"{p}output_upscaling.3.bias");

        for (int i = 0; i < _numMaskTokens; i++)
        {
            _hyperMlps[i].LoadWeights(weights, $"{p}output_hypernetworks_mlps.{i}.");
        }
        _iouHead.LoadWeights(weights, $"{p}iou_prediction_head.");
    }

    /// <summary>Yields every loaded weight tensor.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_iouToken is not null) yield return _iouToken;
        if (_maskTokens is not null) yield return _maskTokens;
        if (_objScoreToken is not null) yield return _objScoreToken;
        foreach (TwoWayAttentionBlock l in _layers)
            foreach (Tensor t in l.EnumerateWeights()) yield return t;
        foreach (Tensor t in _finalAttn.EnumerateWeights()) yield return t;
        foreach (Tensor? t in new[] { _normFinalW, _normFinalB, _upConv0W, _upConv0B, _upLnW, _upLnB, _upConv1W, _upConv1B,
                                      _convS0W, _convS0B, _convS1W, _convS1B })
            if (t is not null) yield return t;
        foreach (SamMlpHead h in _hyperMlps)
            foreach (Tensor t in h.EnumerateWeights()) yield return t;
        foreach (Tensor t in _iouHead.EnumerateWeights()) yield return t;
    }

    /// <summary>Predicts masks + IoU scores. <paramref name="imageEmbedding"/>/<paramref name="imagePe"/> are
    /// <c>[1, dim, H, W]</c>, <paramref name="sparsePromptEmbeddings"/> is <c>[1, N, dim]</c>, and the optional
    /// <paramref name="denseEmbeddings"/> is <c>[1, dim, H, W]</c>. Returns mask logits <c>[1, numMasks, 4H, 4W]</c>
    /// and IoU scores <c>[1, numMasks]</c>, where <paramref name="multimaskOutput"/> selects the 3 multimask
    /// tokens (else the single token).</summary>
    public (Tensor maskLogits, Tensor iouScores) Forward(IBackend backend, Tensor imageEmbedding, Tensor imagePe,
        Tensor sparsePromptEmbeddings, Tensor? denseEmbeddings, bool multimaskOutput,
        Tensor? highResStride4 = null, Tensor? highResStride8 = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (_iouToken is null || _maskTokens is null)
            throw new InvalidOperationException("SamMaskDecoder weights not loaded.");
        if (imageEmbedding.Shape.Rank != 4 || imageEmbedding.Shape[0] != 1)
            throw new ArgumentException($"imageEmbedding must be [1,dim,H,W]; got {imageEmbedding.Shape}.", nameof(imageEmbedding));

        int dim = _transformerDim;
        int h = (int)imageEmbedding.Shape[2];
        int w = (int)imageEmbedding.Shape[3];

        // tokens = cat([obj_score?, iou_token, mask_tokens, sparse_prompt], dim=1) → [1, T+N, dim].
        // SAM 2.1 prepends an object-score token, shifting the iou token to index 1.
        int iouIdx = _objScoreToken is not null ? 1 : 0;
        Tensor iou3 = _iouToken!.Reshape(new TensorShape(1, 1, dim));
        Tensor mask3 = _maskTokens!.Reshape(new TensorShape(1, _numMaskTokens, dim));
        int n = (int)sparsePromptEmbeddings.Shape[1];
        int t = iouIdx + 1 + _numMaskTokens;
        Tensor tokens = new Tensor(new TensorShape(1, t + n, dim), DType.F32);
        if (_objScoreToken is not null)
        {
            Tensor obj3 = _objScoreToken.Reshape(new TensorShape(1, 1, dim));
            backend.Concat(tokens, [obj3, iou3, mask3, sparsePromptEmbeddings], 1);
        }
        else
        {
            backend.Concat(tokens, [iou3, mask3, sparsePromptEmbeddings], 1);
        }

        // src = image_embedding (+ dense), flattened to a sequence [1, HW, dim].
        Tensor src = imageEmbedding;
        if (denseEmbeddings is not null)
        {
            Tensor summed = new Tensor(imageEmbedding.Shape, DType.F32);
            backend.Add(summed, imageEmbedding, denseEmbeddings);
            src = summed;
        }
        Tensor keys = NchwToSeq(backend, src, dim, h, w);
        Tensor peSeq = NchwToSeq(backend, imagePe, dim, h, w);

        Tensor queries = tokens;
        for (int i = 0; i < _layers.Length; i++)
        {
            (queries, keys) = _layers[i].Forward(backend, queries, keys, tokens, peSeq);
        }

        // Final token→image attention.
        Tensor qf = SamDecoderOps.AddNew(backend, queries, tokens);
        Tensor kf = SamDecoderOps.AddNew(backend, keys, peSeq);
        Tensor finalOut = _finalAttn.Forward(backend, qf, kf, keys);
        queries = SamDecoderOps.AddNew(backend, queries, finalOut);
        queries = SamDecoderOps.LayerNormNew(backend, queries, _normFinalW!, _normFinalB!, LnEps);

        // Upscale the transformed image embedding 4× (fusing high-res FPN features when available).
        Tensor srcNchw = SeqToNchw(backend, keys, dim, h, w);
        Tensor upscaled = UpscaleEmbedding(backend, srcNchw, h, w, highResStride4, highResStride8);
        int hyperDim = (int)upscaled.Shape[1];
        int uH = (int)upscaled.Shape[2];
        int uW = (int)upscaled.Shape[3];

        // Per-mask hypernetwork MLPs on each mask token → [1, numMaskTokens, hyperDim].
        Tensor hyper = new Tensor(new TensorShape(1, _numMaskTokens, hyperDim), DType.F32);
        float* hyperPtr = (float*)hyper.DataPointer;
        for (int i = 0; i < _numMaskTokens; i++)
        {
            Tensor tok = SliceToken(queries, iouIdx + 1 + i, dim);
            Tensor hv = _hyperMlps[i].Forward(backend, tok);
            float* hvPtr = (float*)hv.DataPointer;
            for (int d = 0; d < hyperDim; d++) hyperPtr[i * hyperDim + d] = hvPtr[d];
        }

        // masks = hyper @ upscaled.view(1, hyperDim, uH*uW) → [1, numMaskTokens, uH, uW].
        Tensor upFlat = upscaled.Reshape(new TensorShape(1, hyperDim, uH * uW));
        Tensor masksFull = new Tensor(new TensorShape(1, _numMaskTokens, uH * uW), DType.F32);
        backend.BatchedMatMul(masksFull, hyper, upFlat);
        masksFull = masksFull.Reshape(new TensorShape(1, _numMaskTokens, uH, uW));

        // IoU scores from the IoU token (SAM 2.1 applies a sigmoid).
        Tensor iouTok = SliceToken(queries, iouIdx, dim);
        Tensor iouFull = _iouHead.Forward(backend, iouTok); // [1, numMaskTokens]
        if (_iouUseSigmoid)
            backend.Sigmoid(iouFull, iouFull);

        int start = multimaskOutput ? 1 : 0;
        int count = multimaskOutput ? _numMaskTokens - 1 : 1;
        Tensor maskLogits = SliceMasks(masksFull, start, count, uH, uW);
        Tensor iouScores = SliceIou(iouFull, start, count);
        return (maskLogits, iouScores);
    }

    /// <summary>Two transposed-conv 2× upsamples with a channel LayerNorm + GELU between and a GELU after.
    /// When the high-res FPN levels are supplied (SAM 2.1 <c>use_high_res_features_in_sam</c>), the stride-8
    /// level (via <c>conv_s1</c>) is added after the first transposed conv and the stride-4 level (via
    /// <c>conv_s0</c>) after the second, matching <c>act(dc(x) + feat)</c> in the reference upscaler.</summary>
    private Tensor UpscaleEmbedding(IBackend backend, Tensor srcNchw, int h, int w,
        Tensor? highResStride4, Tensor? highResStride8)
    {
        bool useHighRes = _convS0W is not null && highResStride4 is not null && highResStride8 is not null;

        int mid = (int)_upConv0W!.Shape[1];
        Tensor up0 = new Tensor(new TensorShape(1, mid, 2 * h, 2 * w), DType.F32);
        backend.ConvTranspose2d(up0, srcNchw, _upConv0W!, _upConv0B, 2, 2, 0, 0);
        if (useHighRes)
        {
            Tensor featS1 = new Tensor(new TensorShape(1, mid, 2 * h, 2 * w), DType.F32);
            backend.Conv2D(featS1, highResStride8!, _convS1W!, _convS1B!, 1, 1, 0, 0);
            backend.Add(up0, up0, featS1);
        }
        Tensor up0n = LayerNorm2d(backend, up0, _upLnW!, _upLnB!, mid, 2 * h, 2 * w);
        backend.Gelu(up0n, up0n);

        int outCh = (int)_upConv1W!.Shape[1];
        Tensor up1 = new Tensor(new TensorShape(1, outCh, 4 * h, 4 * w), DType.F32);
        backend.ConvTranspose2d(up1, up0n, _upConv1W!, _upConv1B, 2, 2, 0, 0);
        if (useHighRes)
        {
            Tensor featS0 = new Tensor(new TensorShape(1, outCh, 4 * h, 4 * w), DType.F32);
            backend.Conv2D(featS0, highResStride4!, _convS0W!, _convS0B!, 1, 1, 0, 0);
            backend.Add(up1, up1, featS0);
        }
        backend.Gelu(up1, up1);
        return up1;
    }

    /// <summary>Channel-wise LayerNorm2d on <c>[1,C,H,W]</c> (normalize over C per spatial location) via a
    /// transpose to last-dim, the affine LayerNorm, and a transpose back.</summary>
    private static Tensor LayerNorm2d(IBackend backend, Tensor x, Tensor weight, Tensor bias, int c, int h, int w)
    {
        int hw = h * w;
        Tensor v = x.Reshape(new TensorShape(1, c, hw));
        Tensor t = new Tensor(new TensorShape(1, hw, c), DType.F32);
        backend.Transpose2D(t, v, c, hw);
        Tensor normed = new Tensor(new TensorShape(1, hw, c), DType.F32);
        backend.LayerNorm(normed, t, weight, bias, LnEps);
        Tensor back = new Tensor(new TensorShape(1, c, hw), DType.F32);
        backend.Transpose2D(back, normed, hw, c);
        return back.Reshape(new TensorShape(1, c, h, w));
    }

    /// <summary>[1,C,H,W] → [1,H·W,C].</summary>
    private static Tensor NchwToSeq(IBackend backend, Tensor x, int c, int h, int w)
    {
        Tensor v = x.Reshape(new TensorShape(1, c, h * w));
        Tensor outT = new Tensor(new TensorShape(1, h * w, c), DType.F32);
        backend.Transpose2D(outT, v, c, h * w);
        return outT;
    }

    /// <summary>[1,H·W,C] → [1,C,H,W].</summary>
    private static Tensor SeqToNchw(IBackend backend, Tensor x, int c, int h, int w)
    {
        Tensor t = new Tensor(new TensorShape(1, c, h * w), DType.F32);
        backend.Transpose2D(t, x, h * w, c);
        return t.Reshape(new TensorShape(1, c, h, w));
    }

    private static Tensor SliceToken(Tensor queries, int tokenIdx, int dim)
    {
        Tensor outT = new Tensor(new TensorShape(1, dim), DType.F32);
        float* src = (float*)queries.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int d = 0; d < dim; d++) dst[d] = src[(long)tokenIdx * dim + d];
        return outT;
    }

    private static Tensor SliceMasks(Tensor masksFull, int start, int count, int h, int w)
    {
        Tensor outT = new Tensor(new TensorShape(1, count, h, w), DType.F32);
        float* src = (float*)masksFull.DataPointer;
        float* dst = (float*)outT.DataPointer;
        long plane = (long)h * w;
        for (int m = 0; m < count; m++)
        {
            long sOff = (start + m) * plane;
            long dOff = m * plane;
            for (long i = 0; i < plane; i++) dst[dOff + i] = src[sOff + i];
        }
        return outT;
    }

    private static Tensor SliceIou(Tensor iouFull, int start, int count)
    {
        Tensor outT = new Tensor(new TensorShape(1, count), DType.F32);
        float* src = (float*)iouFull.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int m = 0; m < count; m++) dst[m] = src[start + m];
        return outT;
    }

    private static Tensor F32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        if (!weights.TryGetValue(key, out Tensor? t))
            throw new KeyNotFoundException($"SamMaskDecoder: missing weight '{key}'.");
        return t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
    }
}

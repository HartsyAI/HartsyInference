using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>RT-DETR transformer decoder: <c>NumDecoderLayers</c> layers of [query self-attention →
/// multi-scale deformable cross-attention → FFN] with per-layer iterative box refinement, over the
/// object queries + reference points selected by the two-stage query selection in
/// <see cref="RtDetrModel"/>. Reads each layer's class + box heads; the final layer's outputs are the
/// detection logits and boxes (cxcywh, normalized).
/// <para><b>Deformable attention</b> predicts, per query and head, <c>NumLevels·NumPoints</c> sampling
/// offsets + softmax attention weights, then bilinearly samples the multi-level value maps at
/// reference-point + offset locations (<see cref="BilinearSampleZeroPad"/>, zero padding, grid
/// align_corners=False) and weight-sums them — a host loop over queries×heads×levels×points.</para></summary>
public sealed class RtDetrDecoder
{
    private readonly RtDetrConfig _config;
    private readonly DecoderLayer[] _layers;
    private readonly RtDetrLinear[] _classEmbed;
    private readonly RtDetrMlp[] _bboxEmbed;
    private readonly RtDetrMlp _queryPosHead;

    /// <summary>Builds the decoder from a config.</summary>
    public RtDetrDecoder(RtDetrConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _layers = new DecoderLayer[config.NumDecoderLayers];
        _classEmbed = new RtDetrLinear[config.NumDecoderLayers];
        _bboxEmbed = new RtDetrMlp[config.NumDecoderLayers];
        for (int i = 0; i < config.NumDecoderLayers; i++)
        {
            _layers[i] = new DecoderLayer(config);
            _classEmbed[i] = new RtDetrLinear(config.NumClasses);
            _bboxEmbed[i] = new RtDetrMlp([config.HiddenDim, config.HiddenDim, 4]);
        }
        _queryPosHead = new RtDetrMlp([2 * config.HiddenDim, config.HiddenDim]);
    }

    /// <summary>Loads decoder weights. <paramref name="prefix"/> is <c>model.decoder</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].LoadWeights(weights, $"{prefix}.layers.{i}");
            _classEmbed[i].LoadWeights(weights, $"{prefix}.class_embed.{i}");
            _bboxEmbed[i].LoadWeights(weights, $"{prefix}.bbox_embed.{i}");
        }
        _queryPosHead.LoadWeights(weights, $"{prefix}.query_pos_head");
    }

    /// <summary>Yields every decoder weight for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < _layers.Length; i++)
        {
            foreach (Tensor t in _layers[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _classEmbed[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _bboxEmbed[i].EnumerateWeights()) yield return t;
        }
        foreach (Tensor t in _queryPosHead.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the decoder. <paramref name="target"/> is the initial query embeddings
    /// <c>[1,Q,d]</c>; <paramref name="initReferenceUnact"/> the pre-sigmoid reference points
    /// <c>[1,Q,4]</c>; <paramref name="sourceFlatten"/> the encoder memory <c>[1,S,d]</c>;
    /// <paramref name="levelShapes"/> and <paramref name="levelStart"/> describe the 3 feature levels.
    /// Returns final class logits <c>[1,Q,NumClasses]</c> and boxes <c>[1,Q,4]</c> (cxcywh in [0,1]).
    /// Caller owns both returned tensors.</summary>
    public Tensor Forward(IBackend backend, Tensor target, Tensor initReferenceUnact, Tensor sourceFlatten,
        (int H, int W)[] levelShapes, int[] levelStart, out Tensor boxes)
    {
        int q = _config.NumQueries;
        Tensor referencePoints = new Tensor(new TensorShape(1, q, 4), DType.F32);
        Sigmoid(initReferenceUnact.AsSpan<float>(), referencePoints.AsSpan<float>());

        Tensor hidden = new Tensor(target.Shape, DType.F32);
        target.AsSpan<float>().CopyTo(hidden.AsSpan<float>());

        Tensor? finalLogits = null;
        Tensor? finalBoxes = null;
        for (int idx = 0; idx < _layers.Length; idx++)
        {
            Tensor queryPos = _queryPosHead.Forward(backend, referencePoints);
            Tensor next = _layers[idx].Forward(backend, hidden, queryPos, referencePoints, sourceFlatten, levelShapes, levelStart);
            queryPos.Dispose();
            hidden.Dispose();
            hidden = next;

            Tensor predictedCorners = _bboxEmbed[idx].Forward(backend, hidden);
            Tensor newRef = new Tensor(new TensorShape(1, q, 4), DType.F32);
            RefineReference(predictedCorners.AsSpan<float>(), referencePoints.AsSpan<float>(), newRef.AsSpan<float>());
            predictedCorners.Dispose();
            referencePoints.Dispose();
            referencePoints = newRef;

            Tensor logits = _classEmbed[idx].Forward(backend, hidden);
            if (idx == _layers.Length - 1)
            {
                finalLogits = logits;
                finalBoxes = new Tensor(new TensorShape(1, q, 4), DType.F32);
                referencePoints.AsSpan<float>().CopyTo(finalBoxes.AsSpan<float>());
            }
            else
            {
                logits.Dispose();
            }
        }
        hidden.Dispose();
        referencePoints.Dispose();
        boxes = finalBoxes!;
        return finalLogits!;
    }

    private static void Sigmoid(ReadOnlySpan<float> src, Span<float> dst)
    {
        for (int i = 0; i < src.Length; i++)
            dst[i] = 1f / (1f + MathF.Exp(-src[i]));
    }

    /// <summary>new_ref = sigmoid(predicted_corners + inverse_sigmoid(reference_points)).</summary>
    private static void RefineReference(ReadOnlySpan<float> corners, ReadOnlySpan<float> reference, Span<float> outRef)
    {
        const float eps = 1e-5f;
        for (int i = 0; i < corners.Length; i++)
        {
            float x = reference[i];
            if (x < 0f) x = 0f; else if (x > 1f) x = 1f;
            float x1 = MathF.Max(x, eps);
            float x2 = MathF.Max(1f - x, eps);
            float inv = MathF.Log(x1 / x2);
            float v = corners[i] + inv;
            outRef[i] = 1f / (1f + MathF.Exp(-v));
        }
    }

    /// <summary>Bilinear sample of <paramref name="channels"/> consecutive channels at pixel
    /// coordinate (<paramref name="px"/>, <paramref name="py"/>) from a flattened value buffer, with
    /// zero padding outside the <paramref name="height"/>×<paramref name="width"/> grid. A grid cell
    /// (x, y) starts at <c>tokenBase + (y·width + x)·rowStride + channelOffset</c>.</summary>
    public static void BilinearSampleZeroPad(ReadOnlySpan<float> grid, int tokenBase, int height, int width,
        int rowStride, int channelOffset, int channels, float px, float py, Span<float> dst)
    {
        for (int c = 0; c < channels; c++)
            dst[c] = 0f;
        int x0 = (int)MathF.Floor(px), y0 = (int)MathF.Floor(py);
        int x1 = x0 + 1, y1 = y0 + 1;
        float tx = px - x0, ty = py - y0;
        AddCorner(grid, tokenBase, height, width, rowStride, channelOffset, channels, x0, y0, (1 - tx) * (1 - ty), dst);
        AddCorner(grid, tokenBase, height, width, rowStride, channelOffset, channels, x1, y0, tx * (1 - ty), dst);
        AddCorner(grid, tokenBase, height, width, rowStride, channelOffset, channels, x0, y1, (1 - tx) * ty, dst);
        AddCorner(grid, tokenBase, height, width, rowStride, channelOffset, channels, x1, y1, tx * ty, dst);
    }

    private static void AddCorner(ReadOnlySpan<float> grid, int tokenBase, int height, int width, int rowStride,
        int channelOffset, int channels, int x, int y, float weight, Span<float> dst)
    {
        if (weight == 0f || x < 0 || x >= width || y < 0 || y >= height)
            return;
        int cellBase = tokenBase + (y * width + x) * rowStride + channelOffset;
        for (int c = 0; c < channels; c++)
            dst[c] += weight * grid[cellBase + c];
    }

    /// <summary>One RT-DETR decoder layer: query self-attention (pos added to Q/K), multi-scale
    /// deformable cross-attention, and a ReLU feed-forward — each with a residual + post-LayerNorm.</summary>
    private sealed class DecoderLayer
    {
        private readonly RtDetrConfig _config;
        private readonly RtDetrLinear _qProj, _kProj, _vProj, _outProj;
        private readonly RtDetrLayerNorm _selfAttnLayerNorm;
        private readonly RtDetrLinear _samplingOffsets, _attentionWeights, _valueProj, _outputProj;
        private readonly RtDetrLayerNorm _encoderAttnLayerNorm;
        private readonly RtDetrLinear _fc1, _fc2;
        private readonly RtDetrLayerNorm _finalLayerNorm;

        public DecoderLayer(RtDetrConfig config)
        {
            _config = config;
            int d = config.HiddenDim;
            _qProj = new RtDetrLinear(d);
            _kProj = new RtDetrLinear(d);
            _vProj = new RtDetrLinear(d);
            _outProj = new RtDetrLinear(d);
            _selfAttnLayerNorm = new RtDetrLayerNorm(d, config.LayerNormEps);
            _samplingOffsets = new RtDetrLinear(config.NumHeads * config.NumLevels * config.NumPoints * 2);
            _attentionWeights = new RtDetrLinear(config.NumHeads * config.NumLevels * config.NumPoints);
            _valueProj = new RtDetrLinear(d);
            _outputProj = new RtDetrLinear(d);
            _encoderAttnLayerNorm = new RtDetrLayerNorm(d, config.LayerNormEps);
            _fc1 = new RtDetrLinear(config.FeedForwardDim);
            _fc2 = new RtDetrLinear(d);
            _finalLayerNorm = new RtDetrLayerNorm(d, config.LayerNormEps);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
        {
            _qProj.LoadWeights(weights, $"{prefix}.self_attn.q_proj");
            _kProj.LoadWeights(weights, $"{prefix}.self_attn.k_proj");
            _vProj.LoadWeights(weights, $"{prefix}.self_attn.v_proj");
            _outProj.LoadWeights(weights, $"{prefix}.self_attn.out_proj");
            _selfAttnLayerNorm.LoadWeights(weights, $"{prefix}.self_attn_layer_norm");
            _samplingOffsets.LoadWeights(weights, $"{prefix}.encoder_attn.sampling_offsets");
            _attentionWeights.LoadWeights(weights, $"{prefix}.encoder_attn.attention_weights");
            _valueProj.LoadWeights(weights, $"{prefix}.encoder_attn.value_proj");
            _outputProj.LoadWeights(weights, $"{prefix}.encoder_attn.output_proj");
            _encoderAttnLayerNorm.LoadWeights(weights, $"{prefix}.encoder_attn_layer_norm");
            _fc1.LoadWeights(weights, $"{prefix}.fc1");
            _fc2.LoadWeights(weights, $"{prefix}.fc2");
            _finalLayerNorm.LoadWeights(weights, $"{prefix}.final_layer_norm");
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (RtDetrLinear l in new[] { _qProj, _kProj, _vProj, _outProj, _samplingOffsets, _attentionWeights, _valueProj, _outputProj, _fc1, _fc2 })
                foreach (Tensor t in l.EnumerateWeights()) yield return t;
            foreach (RtDetrLayerNorm ln in new[] { _selfAttnLayerNorm, _encoderAttnLayerNorm, _finalLayerNorm })
                foreach (Tensor t in ln.EnumerateWeights()) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor hidden, Tensor queryPos, Tensor referencePoints,
            Tensor sourceFlatten, (int H, int W)[] levelShapes, int[] levelStart)
        {
            // Self-attention: pos added to queries/keys, not values.
            Tensor qkIn = Add(hidden, queryPos);
            Tensor query = _qProj.Forward(backend, qkIn);
            Tensor key = _kProj.Forward(backend, qkIn);
            qkIn.Dispose();
            Tensor value = _vProj.Forward(backend, hidden);
            Tensor attn = RtDetrAttention.MultiHead(backend, query, key, value, _config.NumHeads, _config.HeadDim);
            query.Dispose(); key.Dispose(); value.Dispose();
            Tensor attnOut = _outProj.Forward(backend, attn);
            attn.Dispose();
            Tensor selfSum = Add(hidden, attnOut);
            attnOut.Dispose();
            Tensor h1 = _selfAttnLayerNorm.Forward(backend, selfSum);
            selfSum.Dispose();

            // Deformable cross-attention.
            Tensor crossIn = Add(h1, queryPos);
            Tensor cross = DeformableAttention(backend, crossIn, sourceFlatten, referencePoints, levelShapes, levelStart);
            crossIn.Dispose();
            Tensor crossSum = Add(h1, cross);
            cross.Dispose();
            h1.Dispose();
            Tensor h2 = _encoderAttnLayerNorm.Forward(backend, crossSum);
            crossSum.Dispose();

            // Feed-forward (ReLU).
            Tensor ff1 = _fc1.Forward(backend, h2);
            ReluInPlace(ff1);
            Tensor ff2 = _fc2.Forward(backend, ff1);
            ff1.Dispose();
            Tensor ffSum = Add(h2, ff2);
            ff2.Dispose();
            h2.Dispose();
            Tensor result = _finalLayerNorm.Forward(backend, ffSum);
            ffSum.Dispose();
            return result;
        }

        private unsafe Tensor DeformableAttention(IBackend backend, Tensor queries, Tensor sourceFlatten,
            Tensor referencePoints, (int H, int W)[] levelShapes, int[] levelStart)
        {
            int q = _config.NumQueries;
            int heads = _config.NumHeads;
            int levels = _config.NumLevels;
            int points = _config.NumPoints;
            int d = _config.HiddenDim;

            Tensor value = _valueProj.Forward(backend, sourceFlatten);   // [1, S, d]
            Tensor offsetsT = _samplingOffsets.Forward(backend, queries); // [1, Q, heads*levels*points*2]
            Tensor weightsT = _attentionWeights.Forward(backend, queries); // [1, Q, heads*levels*points] (raw logits)

            // Flatten level shapes to [levels*2] (h,w); softmax over levels*points is folded into the backend op.
            Span<int> shapesFlat = stackalloc int[levels * 2];
            for (int l = 0; l < levels; l++) { shapesFlat[l * 2] = levelShapes[l].H; shapesFlat[l * 2 + 1] = levelShapes[l].W; }

            // RT-DETR shares one [q,4] reference across all levels: refQueryStride=4 (coords), refLevelStride=0.
            Tensor output = new Tensor(new TensorShape(1, q, d), DType.F32);
            backend.DeformableAttention(output, value, offsetsT, weightsT, referencePoints, shapesFlat, levelStart,
                heads, levels, points, 4, 4, 0);

            value.Dispose();
            offsetsT.Dispose();
            weightsT.Dispose();
            Tensor projected = _outputProj.Forward(backend, output);
            output.Dispose();
            return projected;
        }

        private static Tensor Add(Tensor a, Tensor b)
        {
            Tensor outp = new Tensor(a.Shape, DType.F32);
            Span<float> o = outp.AsSpan<float>();
            ReadOnlySpan<float> sa = a.AsSpan<float>(), sb = b.AsSpan<float>();
            for (int i = 0; i < o.Length; i++)
                o[i] = sa[i] + sb[i];
            return outp;
        }

        private static void ReluInPlace(Tensor t)
        {
            Span<float> s = t.AsSpan<float>();
            for (int i = 0; i < s.Length; i++)
                if (s[i] < 0f) s[i] = 0f;
        }
    }
}

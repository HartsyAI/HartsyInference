using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>RT-DETR transformer decoder — the NMS-free query decoder and the net-new piece of this
/// port. It (1) selects <c>NumQueries</c> object queries from the encoder memory by top-k
/// objectness, (2) runs <c>NumDecoderLayers</c> layers of [query self-attention → multi-scale
/// deformable cross-attention → FFN] with per-layer iterative box refinement, and (3) reads the
/// last layer's class + box heads.
/// <para><b>Deformable attention</b> predicts, per query and head, <c>NumLevels·NumPoints</c>
/// sampling offsets + attention weights, then bilinearly samples the multi-level value maps at
/// reference-point + offset locations and weight-sums them. The sampling is a straightforward host
/// loop over queries×heads×levels×points (<see cref="BilinearSampleZeroPad"/>) — the
/// perf-naive-but-correct path; a fused kernel is future work. Feature sizes in the structural test
/// config are tiny, so the host loop is acceptable there.</para>
/// <para>Structural, parity-unverified: exact PaddleDetection/Ultralytics weight names, the RepC3
/// query-selection MLP widths, and whether value carries the query position embedding.</para></summary>
public sealed class RtDetrDecoder
{
    private readonly RtDetrConfig _config;

    // Query selection.
    private readonly RtDetrLinear _encOutput;      // hidden -> hidden
    private readonly RtDetrLayerNorm _encOutputNorm;
    private readonly RtDetrLinear _encScore;       // hidden -> numClasses
    private readonly RtDetrLinear _encBbox0;       // hidden -> hidden (relu)
    private readonly RtDetrLinear _encBbox1;       // hidden -> hidden (relu)
    private readonly RtDetrLinear _encBbox2;       // hidden -> 4

    // Query position embedding MLP: 4 -> 2*hidden (relu) -> hidden.
    private readonly RtDetrLinear _queryPos0;
    private readonly RtDetrLinear _queryPos1;

    // Per-layer modules.
    private readonly RtDetrLinear[] _selfQkv;      // hidden -> 3*hidden
    private readonly RtDetrLinear[] _selfProj;     // hidden -> hidden
    private readonly RtDetrLayerNorm[] _norm1;
    private readonly RtDetrLinear[] _samplingOffsets;   // hidden -> heads*levels*points*2
    private readonly RtDetrLinear[] _attentionWeights;  // hidden -> heads*levels*points
    private readonly RtDetrLinear[] _valueProj;    // hidden -> hidden
    private readonly RtDetrLinear[] _outputProj;   // hidden -> hidden
    private readonly RtDetrLayerNorm[] _norm2;
    private readonly RtDetrLinear[] _ffnFc1;       // hidden -> ffn
    private readonly RtDetrLinear[] _ffnFc2;       // ffn -> hidden
    private readonly RtDetrLayerNorm[] _norm3;

    // Per-layer prediction heads (refinement + final read-out).
    private readonly RtDetrLinear[] _decScore;     // hidden -> numClasses
    private readonly RtDetrLinear[] _decBbox0;     // hidden -> hidden (relu)
    private readonly RtDetrLinear[] _decBbox1;     // hidden -> hidden (relu)
    private readonly RtDetrLinear[] _decBbox2;     // hidden -> 4

    /// <summary>Constructs the decoder module tree from a config.</summary>
    public RtDetrDecoder(RtDetrConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        int h = config.HiddenDim;
        int nc = config.NumClasses;
        int ffn = config.FeedForwardDim;
        int layers = config.NumDecoderLayers;
        int deformDim = config.NumHeads * config.NumLevels * config.NumPoints;

        _encOutput = new RtDetrLinear(h);
        _encOutputNorm = new RtDetrLayerNorm(h, config.LayerNormEps);
        _encScore = new RtDetrLinear(nc);
        _encBbox0 = new RtDetrLinear(h);
        _encBbox1 = new RtDetrLinear(h);
        _encBbox2 = new RtDetrLinear(4);

        _queryPos0 = new RtDetrLinear(2 * h);
        _queryPos1 = new RtDetrLinear(h);

        _selfQkv = new RtDetrLinear[layers];
        _selfProj = new RtDetrLinear[layers];
        _norm1 = new RtDetrLayerNorm[layers];
        _samplingOffsets = new RtDetrLinear[layers];
        _attentionWeights = new RtDetrLinear[layers];
        _valueProj = new RtDetrLinear[layers];
        _outputProj = new RtDetrLinear[layers];
        _norm2 = new RtDetrLayerNorm[layers];
        _ffnFc1 = new RtDetrLinear[layers];
        _ffnFc2 = new RtDetrLinear[layers];
        _norm3 = new RtDetrLayerNorm[layers];
        _decScore = new RtDetrLinear[layers];
        _decBbox0 = new RtDetrLinear[layers];
        _decBbox1 = new RtDetrLinear[layers];
        _decBbox2 = new RtDetrLinear[layers];
        for (int i = 0; i < layers; i++)
        {
            _selfQkv[i] = new RtDetrLinear(3 * h);
            _selfProj[i] = new RtDetrLinear(h);
            _norm1[i] = new RtDetrLayerNorm(h, config.LayerNormEps);
            _samplingOffsets[i] = new RtDetrLinear(deformDim * 2);
            _attentionWeights[i] = new RtDetrLinear(deformDim);
            _valueProj[i] = new RtDetrLinear(h);
            _outputProj[i] = new RtDetrLinear(h);
            _norm2[i] = new RtDetrLayerNorm(h, config.LayerNormEps);
            _ffnFc1[i] = new RtDetrLinear(ffn);
            _ffnFc2[i] = new RtDetrLinear(h);
            _norm3[i] = new RtDetrLayerNorm(h, config.LayerNormEps);
            _decScore[i] = new RtDetrLinear(nc);
            _decBbox0[i] = new RtDetrLinear(h);
            _decBbox1[i] = new RtDetrLinear(h);
            _decBbox2[i] = new RtDetrLinear(4);
        }
    }

    /// <summary>Loads decoder weights under <paramref name="prefix"/> (default <c>"decoder"</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _encOutput.LoadWeights(weights, $"{prefix}.enc_output.linear");
        _encOutputNorm.LoadWeights(weights, $"{prefix}.enc_output.norm");
        _encScore.LoadWeights(weights, $"{prefix}.enc_score");
        _encBbox0.LoadWeights(weights, $"{prefix}.enc_bbox.0");
        _encBbox1.LoadWeights(weights, $"{prefix}.enc_bbox.1");
        _encBbox2.LoadWeights(weights, $"{prefix}.enc_bbox.2");
        _queryPos0.LoadWeights(weights, $"{prefix}.query_pos_head.0");
        _queryPos1.LoadWeights(weights, $"{prefix}.query_pos_head.1");

        for (int i = 0; i < _config.NumDecoderLayers; i++)
        {
            string b = $"{prefix}.layers.{i}";
            _selfQkv[i].LoadWeights(weights, $"{b}.self_attn.qkv");
            _selfProj[i].LoadWeights(weights, $"{b}.self_attn.proj");
            _norm1[i].LoadWeights(weights, $"{b}.norm1");
            _samplingOffsets[i].LoadWeights(weights, $"{b}.cross_attn.sampling_offsets");
            _attentionWeights[i].LoadWeights(weights, $"{b}.cross_attn.attention_weights");
            _valueProj[i].LoadWeights(weights, $"{b}.cross_attn.value_proj");
            _outputProj[i].LoadWeights(weights, $"{b}.cross_attn.output_proj");
            _norm2[i].LoadWeights(weights, $"{b}.norm2");
            _ffnFc1[i].LoadWeights(weights, $"{b}.ffn.fc1");
            _ffnFc2[i].LoadWeights(weights, $"{b}.ffn.fc2");
            _norm3[i].LoadWeights(weights, $"{b}.norm3");
            _decScore[i].LoadWeights(weights, $"{prefix}.dec_score.{i}");
            _decBbox0[i].LoadWeights(weights, $"{prefix}.dec_bbox.{i}.0");
            _decBbox1[i].LoadWeights(weights, $"{prefix}.dec_bbox.{i}.1");
            _decBbox2[i].LoadWeights(weights, $"{prefix}.dec_bbox.{i}.2");
        }
    }

    /// <summary>Yields every decoder weight for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _encOutput.EnumerateWeights()) yield return t;
        foreach (Tensor t in _encOutputNorm.EnumerateWeights()) yield return t;
        foreach (Tensor t in _encScore.EnumerateWeights()) yield return t;
        foreach (Tensor t in _encBbox0.EnumerateWeights()) yield return t;
        foreach (Tensor t in _encBbox1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _encBbox2.EnumerateWeights()) yield return t;
        foreach (Tensor t in _queryPos0.EnumerateWeights()) yield return t;
        foreach (Tensor t in _queryPos1.EnumerateWeights()) yield return t;
        for (int i = 0; i < _config.NumDecoderLayers; i++)
        {
            foreach (Tensor t in _selfQkv[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _selfProj[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _norm1[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _samplingOffsets[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _attentionWeights[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _valueProj[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _outputProj[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _norm2[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _ffnFc1[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _ffnFc2[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _norm3[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _decScore[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _decBbox0[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _decBbox1[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _decBbox2[i].EnumerateWeights()) yield return t;
        }
    }

    /// <summary>Runs query selection + the deformable-attention decoder stack over the three encoder
    /// memory maps. Returns class logits <c>[1, NumQueries, NumClasses]</c> (pre-sigmoid), boxes
    /// <c>[1, NumQueries, 4]</c> (cxcywh in [0,1]), and the final decoder hidden state
    /// <c>[1, NumQueries, hidden]</c>. Caller owns and disposes all three.</summary>
    public (Tensor ClassLogits, Tensor Boxes, Tensor Hidden) Forward(IBackend backend, IReadOnlyList<(Tensor Map, int H, int W)> memoryMaps)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (memoryMaps.Count != _config.NumLevels)
            throw new ArgumentException($"Expected {_config.NumLevels} memory maps; got {memoryMaps.Count}.", nameof(memoryMaps));

        int hidden = _config.HiddenDim;
        LevelShape[] shapes = new LevelShape[_config.NumLevels];
        int start = 0;
        for (int l = 0; l < _config.NumLevels; l++)
        {
            shapes[l] = new LevelShape(memoryMaps[l].H, memoryMaps[l].W, start);
            start += memoryMaps[l].H * memoryMaps[l].W;
        }
        int totalTokens = start;

        Tensor memoryFlat = FlattenMemory(backend, memoryMaps, totalTokens, hidden);

        // ── Query selection ────────────────────────────────────────────────
        Tensor encOutLin = _encOutput.Forward(backend, memoryFlat);
        Tensor encOut = _encOutputNorm.Forward(backend, encOutLin);
        encOutLin.Dispose();
        Tensor encClass = _encScore.Forward(backend, encOut);
        Tensor encRefBoxes = ComputeEncoderRefBoxes(backend, encOut, shapes, totalTokens);

        int[] topk = SelectTopK(encClass, totalTokens, _config.NumClasses, _config.NumQueries);
        encClass.Dispose();

        Tensor target = GatherRows(encOut, topk, hidden);
        Tensor refPoints = GatherRows(encRefBoxes, topk, 4);
        encOut.Dispose();
        encRefBoxes.Dispose();

        // ── Decoder layers ─────────────────────────────────────────────────
        for (int layer = 0; layer < _config.NumDecoderLayers; layer++)
        {
            Tensor queryPos = QueryPositionEmbedding(backend, refPoints);
            Tensor afterSelf = SelfAttention(backend, layer, target, queryPos);
            target.Dispose();

            Tensor crossQuery = new Tensor(afterSelf.Shape, DType.F32);
            backend.Add(crossQuery, afterSelf, queryPos);
            queryPos.Dispose();
            Tensor afterCross = DeformableCrossAttention(backend, layer, crossQuery, refPoints, memoryFlat, shapes);
            crossQuery.Dispose();
            Tensor cross = new Tensor(afterSelf.Shape, DType.F32);
            backend.Add(cross, afterSelf, afterCross);
            afterSelf.Dispose(); afterCross.Dispose();
            Tensor normCross = _norm2[layer].Forward(backend, cross);
            cross.Dispose();

            Tensor ffnOut = FeedForward(backend, layer, normCross);
            target = ApplyResidualNorm(backend, _norm3[layer], normCross, ffnOut);
            normCross.Dispose(); ffnOut.Dispose();

            RefineReferencePoints(backend, layer, target, ref refPoints);
        }

        Tensor classLogits = _decScore[_config.NumDecoderLayers - 1].Forward(backend, target);
        memoryFlat.Dispose();
        return (classLogits, refPoints, target);
    }

    // ── Deformable cross-attention ──────────────────────────────────────────

    private Tensor DeformableCrossAttention(IBackend backend, int layer, Tensor query, Tensor refPoints, Tensor memoryFlat, ReadOnlySpan<LevelShape> shapes)
    {
        int nq = _config.NumQueries;
        int hidden = _config.HiddenDim;
        int heads = _config.NumHeads;
        int headDim = _config.HeadDim;
        int levels = _config.NumLevels;
        int points = _config.NumPoints;

        Tensor value = _valueProj[layer].Forward(backend, memoryFlat);
        Tensor offsets = _samplingOffsets[layer].Forward(backend, query);
        Tensor attnLogits = _attentionWeights[layer].Forward(backend, query);

        ReadOnlySpan<float> valueSpan = value.AsSpan<float>();
        ReadOnlySpan<float> offSpan = offsets.AsSpan<float>();
        Span<float> attnSpan = attnLogits.AsSpan<float>();
        ReadOnlySpan<float> refSpan = refPoints.AsSpan<float>();

        // Softmax attention weights over (levels*points) per (query, head).
        int perHead = levels * points;
        for (int q = 0; q < nq; q++)
        {
            for (int hh = 0; hh < heads; hh++)
            {
                int baseIdx = (q * heads + hh) * perHead;
                float max = float.NegativeInfinity;
                for (int j = 0; j < perHead; j++)
                    max = MathF.Max(max, attnSpan[baseIdx + j]);
                float sum = 0f;
                for (int j = 0; j < perHead; j++)
                {
                    float e = MathF.Exp(attnSpan[baseIdx + j] - max);
                    attnSpan[baseIdx + j] = e;
                    sum += e;
                }
                float inv = 1f / sum;
                for (int j = 0; j < perHead; j++)
                    attnSpan[baseIdx + j] *= inv;
            }
        }

        Tensor outAcc = new Tensor(new TensorShape(1, nq, hidden), DType.F32);
        Span<float> outSpan = outAcc.AsSpan<float>();
        outSpan.Clear();
        Span<float> sample = stackalloc float[headDim];

        for (int q = 0; q < nq; q++)
        {
            float refCx = refSpan[q * 4 + 0];
            float refCy = refSpan[q * 4 + 1];
            float refW = refSpan[q * 4 + 2];
            float refH = refSpan[q * 4 + 3];
            for (int hh = 0; hh < heads; hh++)
            {
                int channelOffset = hh * headDim;
                int outBase = q * hidden + channelOffset;
                for (int l = 0; l < levels; l++)
                {
                    LevelShape s = shapes[l];
                    for (int pt = 0; pt < points; pt++)
                    {
                        int flat = ((q * heads + hh) * levels + l) * points + pt;
                        float offX = offSpan[flat * 2 + 0];
                        float offY = offSpan[flat * 2 + 1];
                        // 4-d reference-point branch (Deformable-DETR): offset normalized by points, scaled by half the box size.
                        float locX = refCx + offX / points * refW * 0.5f;
                        float locY = refCy + offY / points * refH * 0.5f;
                        // grid_sample(align_corners=False) pixel mapping: px = loc*size - 0.5.
                        float px = locX * s.Width - 0.5f;
                        float py = locY * s.Height - 0.5f;
                        BilinearSampleZeroPad(valueSpan, s.Start, s.Height, s.Width, hidden, channelOffset, headDim, px, py, sample);
                        float w = attnSpan[(q * heads + hh) * perHead + l * points + pt];
                        for (int d = 0; d < headDim; d++)
                            outSpan[outBase + d] += w * sample[d];
                    }
                }
            }
        }

        value.Dispose(); offsets.Dispose(); attnLogits.Dispose();
        Tensor projected = _outputProj[layer].Forward(backend, outAcc);
        outAcc.Dispose();
        return projected;
    }

    /// <summary>Bilinear-samples <paramref name="channels"/> features at pixel <c>(px, py)</c> from a
    /// feature grid, zero-padding out-of-bounds neighbours (grid_sample <c>align_corners=False</c>,
    /// zeros padding). The grid is laid <c>[token, rowStride]</c> with the level's tokens starting at
    /// <paramref name="tokenBase"/> in row-major <c>(y·width + x)</c> order, and this call reads the
    /// window <c>[channelOffset, channelOffset+channels)</c> of each row into <paramref name="dst"/>.</summary>
    internal static void BilinearSampleZeroPad(ReadOnlySpan<float> grid, int tokenBase, int height, int width, int rowStride, int channelOffset, int channels, float px, float py, Span<float> dst)
    {
        int x0 = (int)MathF.Floor(px);
        int y0 = (int)MathF.Floor(py);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float tx = px - x0;
        float ty = py - y0;
        float w00 = (1f - tx) * (1f - ty);
        float w01 = tx * (1f - ty);
        float w10 = (1f - tx) * ty;
        float w11 = tx * ty;

        for (int c = 0; c < channels; c++)
            dst[c] = 0f;

        Accumulate(grid, tokenBase, height, width, rowStride, channelOffset, channels, x0, y0, w00, dst);
        Accumulate(grid, tokenBase, height, width, rowStride, channelOffset, channels, x1, y0, w01, dst);
        Accumulate(grid, tokenBase, height, width, rowStride, channelOffset, channels, x0, y1, w10, dst);
        Accumulate(grid, tokenBase, height, width, rowStride, channelOffset, channels, x1, y1, w11, dst);
    }

    private static void Accumulate(ReadOnlySpan<float> grid, int tokenBase, int height, int width, int rowStride, int channelOffset, int channels, int x, int y, float weight, Span<float> dst)
    {
        if (weight == 0f || x < 0 || x >= width || y < 0 || y >= height)
            return;
        long rowStart = (long)(tokenBase + y * width + x) * rowStride + channelOffset;
        for (int c = 0; c < channels; c++)
            dst[c] += weight * grid[(int)(rowStart + c)];
    }

    // ── Decoder sub-steps ───────────────────────────────────────────────────

    private Tensor SelfAttention(IBackend backend, int layer, Tensor target, Tensor queryPos)
    {
        int nq = _config.NumQueries;
        int hidden = _config.HiddenDim;
        Tensor qkvInput = new Tensor(target.Shape, DType.F32);
        backend.Add(qkvInput, target, queryPos);
        Tensor qkv = _selfQkv[layer].Forward(backend, qkvInput);
        qkvInput.Dispose();

        Tensor q = new Tensor(new TensorShape(1, nq, hidden), DType.F32);
        Tensor k = new Tensor(new TensorShape(1, nq, hidden), DType.F32);
        Tensor v = new Tensor(new TensorShape(1, nq, hidden), DType.F32);
        backend.Split([q, k, v], qkv, dim: 2);
        qkv.Dispose();

        Tensor attn = RtDetrAttention.MultiHead(backend, q, k, v, _config.NumHeads, _config.HeadDim);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor proj = _selfProj[layer].Forward(backend, attn);
        attn.Dispose();

        Tensor normed = ApplyResidualNorm(backend, _norm1[layer], target, proj);
        proj.Dispose();
        return normed;
    }

    private Tensor FeedForward(IBackend backend, int layer, Tensor input)
    {
        Tensor fc1 = _ffnFc1[layer].Forward(backend, input);
        backend.Gelu(fc1, fc1);
        Tensor fc2 = _ffnFc2[layer].Forward(backend, fc1);
        fc1.Dispose();
        return fc2;
    }

    private Tensor QueryPositionEmbedding(IBackend backend, Tensor refPoints)
    {
        Tensor h0 = _queryPos0.Forward(backend, refPoints);
        backend.Clamp(h0, h0, 0f, float.MaxValue);   // ReLU
        Tensor h1 = _queryPos1.Forward(backend, h0);
        h0.Dispose();
        return h1;
    }

    private static Tensor ApplyResidualNorm(IBackend backend, RtDetrLayerNorm norm, Tensor residual, Tensor delta)
    {
        Tensor sum = new Tensor(residual.Shape, DType.F32);
        backend.Add(sum, residual, delta);
        Tensor normed = norm.Forward(backend, sum);
        sum.Dispose();
        return normed;
    }

    private void RefineReferencePoints(IBackend backend, int layer, Tensor target, ref Tensor refPoints)
    {
        Tensor b0 = _decBbox0[layer].Forward(backend, target);
        backend.Clamp(b0, b0, 0f, float.MaxValue);
        Tensor b1 = _decBbox1[layer].Forward(backend, b0);
        backend.Clamp(b1, b1, 0f, float.MaxValue);
        Tensor delta = _decBbox2[layer].Forward(backend, b1);
        b0.Dispose(); b1.Dispose();

        int nq = _config.NumQueries;
        Tensor refined = new Tensor(new TensorShape(1, nq, 4), DType.F32);
        ReadOnlySpan<float> refSpan = refPoints.AsSpan<float>();
        ReadOnlySpan<float> deltaSpan = delta.AsSpan<float>();
        Span<float> outSpan = refined.AsSpan<float>();
        for (int i = 0; i < nq * 4; i++)
            outSpan[i] = Sigmoid(deltaSpan[i] + Logit(refSpan[i]));
        delta.Dispose();
        refPoints.Dispose();
        refPoints = refined;
    }

    // ── Query selection helpers ─────────────────────────────────────────────

    private Tensor FlattenMemory(IBackend backend, IReadOnlyList<(Tensor Map, int H, int W)> maps, int totalTokens, int hidden)
    {
        Tensor[] perLevel = new Tensor[maps.Count];
        for (int l = 0; l < maps.Count; l++)
        {
            int tokens = maps[l].H * maps[l].W;
            perLevel[l] = new Tensor(new TensorShape(1, tokens, hidden), DType.F32);
            backend.Transpose2D(perLevel[l], maps[l].Map, hidden, tokens);
        }
        Tensor flat = new Tensor(new TensorShape(1, totalTokens, hidden), DType.F32);
        backend.Concat(flat, perLevel, dim: 1);
        for (int l = 0; l < perLevel.Length; l++)
            perLevel[l].Dispose();
        return flat;
    }

    /// <summary>Encoder box head: 3-layer MLP delta added (in logit space) to per-token grid anchors, then sigmoid → cxcywh in [0,1].</summary>
    private Tensor ComputeEncoderRefBoxes(IBackend backend, Tensor encOut, ReadOnlySpan<LevelShape> shapes, int totalTokens)
    {
        Tensor b0 = _encBbox0.Forward(backend, encOut);
        backend.Clamp(b0, b0, 0f, float.MaxValue);
        Tensor b1 = _encBbox1.Forward(backend, b0);
        backend.Clamp(b1, b1, 0f, float.MaxValue);
        Tensor delta = _encBbox2.Forward(backend, b1);
        b0.Dispose(); b1.Dispose();

        Tensor boxes = new Tensor(new TensorShape(1, totalTokens, 4), DType.F32);
        ReadOnlySpan<float> deltaSpan = delta.AsSpan<float>();
        Span<float> outSpan = boxes.AsSpan<float>();

        int token = 0;
        for (int l = 0; l < shapes.Length; l++)
        {
            LevelShape s = shapes[l];
            float wh = 0.05f * (1 << l);   // RT-DETR anchor size grows by 2^level
            for (int y = 0; y < s.Height; y++)
            {
                for (int x = 0; x < s.Width; x++)
                {
                    float cx = (x + 0.5f) / s.Width;
                    float cy = (y + 0.5f) / s.Height;
                    int o = token * 4;
                    outSpan[o + 0] = Sigmoid(deltaSpan[o + 0] + Logit(cx));
                    outSpan[o + 1] = Sigmoid(deltaSpan[o + 1] + Logit(cy));
                    outSpan[o + 2] = Sigmoid(deltaSpan[o + 2] + Logit(wh));
                    outSpan[o + 3] = Sigmoid(deltaSpan[o + 3] + Logit(wh));
                    token++;
                }
            }
        }
        delta.Dispose();
        return boxes;
    }

    private static int[] SelectTopK(Tensor classLogits, int totalTokens, int numClasses, int k)
    {
        ReadOnlySpan<float> span = classLogits.AsSpan<float>();
        float[] score = new float[totalTokens];
        for (int t = 0; t < totalTokens; t++)
        {
            float max = float.NegativeInfinity;
            int baseIdx = t * numClasses;
            for (int c = 0; c < numClasses; c++)
                max = MathF.Max(max, span[baseIdx + c]);
            score[t] = max;
        }
        int[] idx = new int[totalTokens];
        for (int t = 0; t < totalTokens; t++)
            idx[t] = t;
        // Partial selection of the top-k by score (descending). totalTokens is small in practice.
        Array.Sort(idx, (a, b) => score[b].CompareTo(score[a]));
        int take = Math.Min(k, totalTokens);
        int[] topk = new int[k];
        for (int i = 0; i < k; i++)
            topk[i] = idx[Math.Min(i, take - 1)];   // pad by repeating the last if fewer tokens than queries
        return topk;
    }

    private static Tensor GatherRows(Tensor source, int[] rowIndices, int rowWidth)
    {
        int rows = rowIndices.Length;
        Tensor gathered = new Tensor(new TensorShape(1, rows, rowWidth), DType.F32);
        ReadOnlySpan<float> src = source.AsSpan<float>();
        Span<float> dst = gathered.AsSpan<float>();
        for (int i = 0; i < rows; i++)
        {
            int srcOff = rowIndices[i] * rowWidth;
            int dstOff = i * rowWidth;
            for (int c = 0; c < rowWidth; c++)
                dst[dstOff + c] = src[srcOff + c];
        }
        return gathered;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float Logit(float p)
    {
        float clamped = Math.Clamp(p, 1e-6f, 1f - 1e-6f);
        return MathF.Log(clamped / (1f - clamped));
    }

    /// <summary>Per-level spatial size + flattened token start offset.</summary>
    private readonly record struct LevelShape(int Height, int Width, int Start);
}

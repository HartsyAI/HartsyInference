using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>Two-stage query selection + decoder + detection heads for Grounding DINO. Generates per-pixel proposals
/// from the encoder vision output, scores them contrastively against the encoder text, picks the top
/// <c>num_queries</c>, runs the decoder with iterative box refinement, then applies the per-layer contrastive class
/// head and box head to produce the final <c>logits [1, Nq, max_text_len]</c> and <c>pred_boxes [1, Nq, 4]</c>
/// (cxcywh in [0,1]).</summary>
public sealed unsafe class GroundingDinoDetector : IDisposable
{
    private readonly GroundingDinoConfig _cfg;
    private readonly GroundingDinoDecoder _decoder;
    private readonly GroundingDinoMlp _encBboxEmbed;
    private readonly GroundingDinoMlp _bboxEmbed;
    private Tensor? _encOutW, _encOutB, _encNormW, _encNormB, _queryPosEmb;
    private int _disposed;

    public GroundingDinoDetector(GroundingDinoConfig cfg)
    {
        _cfg = cfg;
        _decoder = new GroundingDinoDecoder(cfg);
        _encBboxEmbed = new GroundingDinoMlp(3, [cfg.DModel, cfg.DModel, 4]);
        _bboxEmbed = new GroundingDinoMlp(3, [cfg.DModel, cfg.DModel, 4]);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model")
    {
        _decoder.LoadWeights(w, $"{prefix}.decoder");
        _encBboxEmbed.LoadWeights(w, $"{prefix}.encoder_output_bbox_embed");
        _bboxEmbed.LoadWeights(w, "bbox_embed.0");   // shared decoder + head box MLP
        _encOutW = w[$"{prefix}.enc_output.weight"]; _encOutB = w[$"{prefix}.enc_output.bias"];
        _encNormW = w[$"{prefix}.enc_output_norm.weight"]; _encNormB = w[$"{prefix}.enc_output_norm.bias"];
        _queryPosEmb = w[$"{prefix}.query_position_embeddings.weight"];
    }

    public sealed class Output
    {
        public required Tensor Logits { get; init; }       // [1, Nq, max_text_len]
        public required Tensor PredBoxes { get; init; }    // [1, Nq, 4] cxcywh
    }

    public Output Forward(IBackend backend, Tensor encoderVision, Tensor encoderText, int textLen,
        int[][] spatialShapes, int[] levelStart)
    {
        int nimg = (int)encoderVision.Shape[1], d = _cfg.DModel, nq = _cfg.NumQueries;
        int maxText = _cfg.MaxTextLen;

        // ── two-stage proposals ──
        float[] proposalLogits = new float[(long)nimg * 4];
        bool[] valid = new bool[nimg];
        BuildProposals(spatialShapes, levelStart, proposalLogits, valid);

        // object_query = enc_output_norm(enc_output(encoder_vision with invalid tokens zeroed))
        Tensor masked = new(encoderVision.Shape, DType.F32);
        float* ev = (float*)encoderVision.DataPointer;
        float* mp = (float*)masked.DataPointer;
        for (int i = 0; i < nimg; i++)
        {
            bool keep = valid[i];
            for (int c = 0; c < d; c++) mp[(long)i * d + c] = keep ? ev[(long)i * d + c] : 0f;
        }
        Tensor encOut = new(masked.Shape, DType.F32);
        backend.Linear(encOut, masked, _encOutW!, _encOutB);
        masked.Dispose();
        Tensor objQuery = new(encOut.Shape, DType.F32);
        backend.LayerNorm(objQuery, encOut, _encNormW!, _encNormB!, _cfg.LayerNormEps);
        encOut.Dispose();

        // enc_outputs_class = contrastive(objQuery, encoderText) -> per-token max
        // enc_outputs_coord_logits = enc_bbox(objQuery) + proposalLogits
        Tensor delta = _encBboxEmbed.Forward(backend, objQuery, nimg);
        float* dp = (float*)delta.DataPointer;
        float[] topkScore = new float[nimg];
        float* oq = (float*)objQuery.DataPointer;
        float* et = (float*)encoderText.DataPointer;
        for (int i = 0; i < nimg; i++)
        {
            float best = float.NegativeInfinity;
            for (int j = 0; j < textLen; j++)
            {
                float dot = 0f;
                for (int c = 0; c < d; c++) dot += oq[(long)i * d + c] * et[(long)j * d + c];
                if (dot > best) best = dot;
            }
            topkScore[i] = best;
        }

        // top-k proposals (descending by score)
        int[] order = new int[nimg];
        for (int i = 0; i < nimg; i++) order[i] = i;
        Array.Sort(order, (a, b) => topkScore[b].CompareTo(topkScore[a]));
        int[] topk = new int[nq];
        Array.Copy(order, topk, nq);

        // reference_points = sigmoid(gathered coord_logits)
        float[] referencePoints = new float[(long)nq * 4];
        for (int q = 0; q < nq; q++)
        {
            int src = topk[q];
            for (int c = 0; c < 4; c++)
            {
                float coordLogit = dp[(long)src * 4 + c] + proposalLogits[(long)src * 4 + c];
                referencePoints[(long)q * 4 + c] = GdMath.Sigmoid(coordLogit);
            }
        }
        delta.Dispose();
        objQuery.Dispose();

        // target = query_position_embeddings (embedding_init_target = true)
        Tensor target = new(new TensorShape(1, nq, d), DType.F32);
        long tbytes = (long)nq * d * 4;
        Buffer.MemoryCopy((void*)_queryPosEmb!.DataPointer, (void*)target.DataPointer, tbytes, tbytes);

        // ── decoder ──
        GroundingDinoDecoder.Result dec = _decoder.Forward(backend, target, referencePoints, encoderVision,
            encoderText, _bboxEmbed, spatialShapes, levelStart);
        target.Dispose();

        // ── heads (final layer only) ──
        int last = _cfg.DecoderLayers - 1;
        Tensor finalHidden = dec.IntermediateHidden[last];
        float[] finalRef = last == 0 ? referencePoints : dec.IntermediateReference[last - 1];

        Tensor logits = new(new TensorShape(1, nq, maxText), DType.F32);
        float* lp = (float*)logits.DataPointer;
        float* fh = (float*)finalHidden.DataPointer;
        for (int q = 0; q < nq; q++)
            for (int j = 0; j < maxText; j++)
            {
                float v = float.NegativeInfinity;
                if (j < textLen)
                {
                    float dot = 0f;
                    for (int c = 0; c < d; c++) dot += fh[(long)q * d + c] * et[(long)j * d + c];
                    v = dot;
                }
                lp[(long)q * maxText + j] = v;
            }

        Tensor deltaFinal = _bboxEmbed.Forward(backend, finalHidden, nq);
        float* dfp = (float*)deltaFinal.DataPointer;
        Tensor boxes = new(new TensorShape(1, nq, 4), DType.F32);
        float* bp = (float*)boxes.DataPointer;
        for (int q = 0; q < nq; q++)
            for (int c = 0; c < 4; c++)
            {
                float reference = GdMath.InverseSigmoid(finalRef[(long)q * 4 + c]);
                bp[(long)q * 4 + c] = GdMath.Sigmoid(dfp[(long)q * 4 + c] + reference);
            }
        deltaFinal.Dispose();
        foreach (Tensor t in dec.IntermediateHidden) t.Dispose();

        return new Output { Logits = logits, PredBoxes = boxes };
    }

    private void BuildProposals(int[][] shapes, int[] levelStart, float[] proposalLogits, bool[] valid)
    {
        int levels = _cfg.NumFeatureLevels;
        for (int l = 0; l < levels; l++)
        {
            int h = shapes[l][0], w = shapes[l][1], start = levelStart[l];
            float wh = 0.05f * MathF.Pow(2f, l);
            for (int row = 0; row < h; row++)
                for (int col = 0; col < w; col++)
                {
                    int i = start + row * w + col;
                    float gx = (col + 0.5f) / w;
                    float gy = (row + 0.5f) / h;
                    float[] p = [gx, gy, wh, wh];
                    bool v = true;
                    for (int c = 0; c < 4; c++)
                    {
                        v &= p[c] > 0.01f && p[c] < 0.99f;
                        proposalLogits[(long)i * 4 + c] = MathF.Log(p[c] / (1f - p[c]));
                    }
                    valid[i] = v;
                    if (!v)
                        for (int c = 0; c < 4; c++) proposalLogits[(long)i * 4 + c] = float.PositiveInfinity;
                }
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _decoder.Dispose();
        _encBboxEmbed.Dispose();
        _bboxEmbed.Dispose();
        GC.SuppressFinalize(this);
    }
}

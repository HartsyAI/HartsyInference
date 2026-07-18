using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>Grounding DINO decoder: <c>config.decoder_layers</c> layers of query self-attention, text cross-attention
/// and multi-scale deformable image cross-attention, with iterative box refinement between layers using the shared
/// box head. Returns each layer's LayerNormed hidden state and the refined reference points.</summary>
public sealed unsafe class GroundingDinoDecoder : IDisposable
{
    private readonly GroundingDinoConfig _cfg;
    private readonly DecoderLayer[] _layers;
    private readonly GroundingDinoMlp _refPointsHead;
    private Tensor? _lnW, _lnB;
    private int _disposed;

    public GroundingDinoDecoder(GroundingDinoConfig cfg)
    {
        _cfg = cfg;
        _layers = new DecoderLayer[cfg.DecoderLayers];
        for (int i = 0; i < cfg.DecoderLayers; i++) _layers[i] = new DecoderLayer(cfg);
        _refPointsHead = new GroundingDinoMlp(2, [cfg.DModel, cfg.DModel]);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model.decoder")
    {
        for (int i = 0; i < _cfg.DecoderLayers; i++)
            _layers[i].LoadWeights(w, $"{prefix}.layers.{i}");
        _refPointsHead.LoadWeights(w, $"{prefix}.reference_points_head");
        _lnW = w[$"{prefix}.layer_norm.weight"]; _lnB = w[$"{prefix}.layer_norm.bias"];
    }

    /// <summary>Result: per-layer LayerNormed hidden states and refined reference points, both stacked over layers.</summary>
    public sealed class Result
    {
        public required Tensor[] IntermediateHidden { get; init; }   // each [1, Nq, d]
        public required float[][] IntermediateReference { get; init; } // each [Nq*4]
    }

    /// <summary>Runs the decoder. <paramref name="target"/> is <c>[1, Nq, d]</c> (initial queries),
    /// <paramref name="referencePoints"/> is <c>[Nq*4]</c> (sigmoid space), <paramref name="bboxEmbed"/> is the shared
    /// 3-layer box head used for refinement.</summary>
    public Result Forward(IBackend backend, Tensor target, float[] referencePoints, Tensor encoderVision,
        Tensor encoderText, GroundingDinoMlp bboxEmbed, int[][] spatialShapes, int[] levelStart)
    {
        int nq = (int)target.Shape[1], d = _cfg.DModel, levels = _cfg.NumFeatureLevels;
        Tensor hidden = Copy(target);
        float[] refPoints = (float[])referencePoints.Clone();

        Tensor[] interHidden = new Tensor[_cfg.DecoderLayers];
        float[][] interRef = new float[_cfg.DecoderLayers][];

        for (int idx = 0; idx < _cfg.DecoderLayers; idx++)
        {
            // reference_points_input broadcast across levels (valid_ratios = 1)
            float[] refInput = new float[(long)nq * levels * 4];
            for (int q = 0; q < nq; q++)
                for (int l = 0; l < levels; l++)
                    for (int c = 0; c < 4; c++)
                        refInput[((long)q * levels + l) * 4 + c] = refPoints[(long)q * 4 + c];

            // query_pos = reference_points_head( sinusoid(reference_points[q], 128) )
            Tensor sinePos = new(new TensorShape(1, nq, _cfg.QueryDim / 2 * d), DType.F32);
            float* spd = (float*)sinePos.DataPointer;
            int half = _cfg.DModel / 2;   // 128
            for (int q = 0; q < nq; q++)
            {
                ReadOnlySpan<float> coord = new ReadOnlySpan<float>(refPoints, q * 4, 4);
                float[] enc = GdMath.EncodeSinusoid(coord, half);
                for (int c = 0; c < enc.Length; c++) spd[(long)q * enc.Length + c] = enc[c];
            }
            Tensor queryPos = _refPointsHead.Forward(backend, sinePos, nq);
            sinePos.Dispose();

            Tensor next = _layers[idx].Forward(backend, hidden, queryPos, encoderVision, encoderText,
                refInput, spatialShapes, levelStart);
            hidden.Dispose();
            hidden = next;
            queryPos.Dispose();

            // box refinement: new_ref = sigmoid(bbox_embed(hidden) + logit(reference_points))
            Tensor delta = bboxEmbed.Forward(backend, hidden, nq);
            float* dp = (float*)delta.DataPointer;
            float[] newRef = new float[(long)nq * 4];
            for (int q = 0; q < nq; q++)
                for (int c = 0; c < 4; c++)
                {
                    float logit = GdMath.InverseSigmoid(refPoints[(long)q * 4 + c]);
                    newRef[(long)q * 4 + c] = GdMath.Sigmoid(dp[(long)q * 4 + c] + logit);
                }
            delta.Dispose();
            refPoints = newRef;

            // intermediate hidden = layer_norm(hidden)
            Tensor ln = new(hidden.Shape, DType.F32);
            backend.LayerNorm(ln, hidden, _lnW!, _lnB!, _cfg.LayerNormEps);
            interHidden[idx] = ln;
            interRef[idx] = (float[])refPoints.Clone();
        }
        hidden.Dispose();
        return new Result { IntermediateHidden = interHidden, IntermediateReference = interRef };
    }

    private static Tensor Copy(Tensor src)
    {
        Tensor dst = new(src.Shape, DType.F32);
        long bytes = src.ElementCount * 4;
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)dst.DataPointer, bytes, bytes);
        return dst;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (DecoderLayer l in _layers) l.Dispose();
        _refPointsHead.Dispose();
        GC.SuppressFinalize(this);
    }

    private sealed class DecoderLayer : IDisposable
    {
        private readonly GroundingDinoConfig _cfg;
        private readonly GroundingDinoMultiheadAttention _selfAttn;
        private readonly GroundingDinoMultiheadAttention _textCrossAttn;
        private readonly MultiscaleDeformableAttention _deformAttn;
        private Tensor? _selfLnW, _selfLnB, _textLnW, _textLnB, _crossLnW, _crossLnB;
        private Tensor? _fc1W, _fc1B, _fc2W, _fc2B, _finalLnW, _finalLnB;

        public DecoderLayer(GroundingDinoConfig cfg)
        {
            _cfg = cfg;
            _selfAttn = new GroundingDinoMultiheadAttention(cfg.DModel, cfg.DecoderAttentionHeads);
            _textCrossAttn = new GroundingDinoMultiheadAttention(cfg.DModel, cfg.DecoderAttentionHeads);
            _deformAttn = new MultiscaleDeformableAttention(cfg.DModel, cfg.DecoderAttentionHeads, cfg.NumFeatureLevels, cfg.DecoderNPoints);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            _selfAttn.LoadWeights(w, $"{prefix}.self_attn");
            _textCrossAttn.LoadWeights(w, $"{prefix}.encoder_attn_text");
            _deformAttn.LoadWeights(w, $"{prefix}.encoder_attn");
            _selfLnW = w[$"{prefix}.self_attn_layer_norm.weight"]; _selfLnB = w[$"{prefix}.self_attn_layer_norm.bias"];
            _textLnW = w[$"{prefix}.encoder_attn_text_layer_norm.weight"]; _textLnB = w[$"{prefix}.encoder_attn_text_layer_norm.bias"];
            _crossLnW = w[$"{prefix}.encoder_attn_layer_norm.weight"]; _crossLnB = w[$"{prefix}.encoder_attn_layer_norm.bias"];
            _fc1W = w[$"{prefix}.fc1.weight"]; _fc1B = w[$"{prefix}.fc1.bias"];
            _fc2W = w[$"{prefix}.fc2.weight"]; _fc2B = w[$"{prefix}.fc2.bias"];
            _finalLnW = w[$"{prefix}.final_layer_norm.weight"]; _finalLnB = w[$"{prefix}.final_layer_norm.bias"];
        }

        public Tensor Forward(IBackend backend, Tensor hiddenIn, Tensor queryPos, Tensor encoderVision, Tensor encoderText,
            float[] refInput, int[][] shapes, int[] levelStart)
        {
            int nq = (int)hiddenIn.Shape[1], d = _cfg.DModel;

            // self-attention (residual add on the GPU into a fresh tensor — never alias out==in on CUDA)
            Tensor qk = new(hiddenIn.Shape, DType.F32);
            backend.Add(qk, hiddenIn, queryPos);
            Tensor sa = _selfAttn.Forward(backend, qk, qk, hiddenIn, null);
            qk.Dispose();
            Tensor saRes = new(sa.Shape, DType.F32);
            backend.Add(saRes, sa, hiddenIn);
            sa.Dispose();
            Tensor h1 = new(saRes.Shape, DType.F32);
            backend.LayerNorm(h1, saRes, _selfLnW!, _selfLnB!, _cfg.LayerNormEps);
            saRes.Dispose();

            // text cross-attention
            Tensor q2 = new(h1.Shape, DType.F32);
            backend.Add(q2, h1, queryPos);
            Tensor ta = _textCrossAttn.Forward(backend, q2, encoderText, encoderText, null);
            q2.Dispose();
            Tensor taRes = new(ta.Shape, DType.F32);
            backend.Add(taRes, ta, h1);
            ta.Dispose();
            Tensor h2 = new(taRes.Shape, DType.F32);
            backend.LayerNorm(h2, taRes, _textLnW!, _textLnB!, _cfg.LayerNormEps);
            taRes.Dispose();
            h1.Dispose();

            // deformable image cross-attention
            Tensor q3 = new(h2.Shape, DType.F32);
            backend.Add(q3, h2, queryPos);
            Tensor da = _deformAttn.Forward(backend, q3, encoderVision, refInput, 4, shapes, levelStart);
            q3.Dispose();
            Tensor daRes = new(da.Shape, DType.F32);
            backend.Add(daRes, da, h2);
            da.Dispose();
            Tensor h3 = new(daRes.Shape, DType.F32);
            backend.LayerNorm(h3, daRes, _crossLnW!, _crossLnB!, _cfg.LayerNormEps);
            daRes.Dispose();
            h2.Dispose();

            // FFN (GPU ReLU via in-place Clamp)
            Tensor fc1 = new(new TensorShape(1, nq, _cfg.DecoderFfnDim), DType.F32);
            backend.Linear(fc1, h3, _fc1W!, _fc1B);
            backend.Clamp(fc1, fc1, 0f, float.MaxValue);
            Tensor fc2 = new(new TensorShape(1, nq, d), DType.F32);
            backend.Linear(fc2, fc1, _fc2W!, _fc2B);
            fc1.Dispose();
            Tensor fc2Res = new(fc2.Shape, DType.F32);
            backend.Add(fc2Res, fc2, h3);
            fc2.Dispose();
            h3.Dispose();
            Tensor outT = new(fc2Res.Shape, DType.F32);
            backend.LayerNorm(outT, fc2Res, _finalLnW!, _finalLnB!, _cfg.LayerNormEps);
            fc2Res.Dispose();
            return outT;
        }

        public void Dispose()
        {
            _selfAttn.Dispose();
            _textCrossAttn.Dispose();
            _deformAttn.Dispose();
        }
    }
}

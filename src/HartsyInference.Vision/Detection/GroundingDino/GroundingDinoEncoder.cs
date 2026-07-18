using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>Grounding DINO cross-modality feature enhancer (encoder): <c>config.encoder_layers</c> layers, each a
/// bi-directional image↔text fusion attention, then a text self-attention enhancer, then a deformable image
/// self-attention — in that order. Assumes no vision/text padding (the shipped detection path pads neither); the
/// only active mask is the special-token block mask in the text enhancer.</summary>
public sealed unsafe class GroundingDinoEncoder : IDisposable
{
    private readonly GroundingDinoConfig _cfg;
    private readonly EncoderLayer[] _layers;
    private int _disposed;

    public GroundingDinoEncoder(GroundingDinoConfig cfg)
    {
        _cfg = cfg;
        _layers = new EncoderLayer[cfg.EncoderLayers];
        for (int i = 0; i < cfg.EncoderLayers; i++) _layers[i] = new EncoderLayer(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model.encoder")
    {
        for (int i = 0; i < _cfg.EncoderLayers; i++)
            _layers[i].LoadWeights(w, $"{prefix}.layers.{i}");
    }

    /// <summary>Runs the enhancer. <paramref name="vision"/> is <c>[1, Nimg, d]</c> (source_flatten),
    /// <paramref name="visionPos"/> is <c>[1, Nimg, d]</c>, <paramref name="text"/> is <c>[1, T, d]</c>.
    /// <paramref name="attendMask"/> is the <c>[T*T]</c> text block mask, <paramref name="positionIds"/> the reset ids.
    /// Returns the enhanced (vision, text). Caller owns both.</summary>
    public (Tensor Vision, Tensor Text) Forward(IBackend backend, Tensor vision, Tensor visionPos, Tensor text,
        bool[] attendMask, int[] positionIds, int[][] spatialShapes, int[] levelStart)
    {
        int nimg = (int)vision.Shape[1];
        float[] refPoints = BuildReferencePoints(spatialShapes, levelStart, nimg);
        int t = (int)text.Shape[1];
        Tensor textPos = BuildTextPositionEmbedding(positionIds, t);
        Tensor textMask = BuildTextAdditiveMask(attendMask, t, _cfg.EncoderAttentionHeads / 2);

        Tensor v = Copy(vision);
        Tensor tx = Copy(text);
        for (int i = 0; i < _layers.Length; i++)
        {
            (Tensor nv, Tensor nt) = _layers[i].Forward(backend, v, visionPos, tx, textPos, textMask,
                refPoints, spatialShapes, levelStart);
            v.Dispose(); tx.Dispose();
            v = nv; tx = nt;
        }
        textPos.Dispose();
        textMask.Dispose();
        return (v, tx);
    }

    /// <summary>Encoder reference points: each token's own normalized cell center, broadcast across all levels
    /// (valid_ratios are 1 for unpadded input). Flattened <c>[Nimg * nLevels * 2]</c>.</summary>
    private float[] BuildReferencePoints(int[][] shapes, int[] levelStart, int nimg)
    {
        int levels = _cfg.NumFeatureLevels;
        float[] refp = new float[(long)nimg * levels * 2];
        for (int l = 0; l < levels; l++)
        {
            int h = shapes[l][0], w = shapes[l][1], start = levelStart[l];
            for (int row = 0; row < h; row++)
                for (int col = 0; col < w; col++)
                {
                    int q = start + row * w + col;
                    float rx = (col + 0.5f) / w;
                    float ry = (row + 0.5f) / h;
                    for (int lvl = 0; lvl < levels; lvl++)
                    {
                        long b = ((long)q * levels + lvl) * 2;
                        refp[b] = rx;
                        refp[b + 1] = ry;
                    }
                }
        }
        return refp;
    }

    /// <summary>Text position embedding via the DETR sinusoidal encoding of the reset position ids (num_pos_feats =
    /// d_model, temperature 10000, single coordinate). Returns <c>[1, T, d_model]</c>.</summary>
    private Tensor BuildTextPositionEmbedding(int[] positionIds, int t)
    {
        int d = _cfg.DModel;
        float scale = 2f * MathF.PI;
        float[] dimT = new float[d];
        for (int k = 0; k < d; k++)
            dimT[k] = MathF.Pow(10000f, 2f * (k / 2) / d);
        Tensor outT = new(new TensorShape(1, t, d), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int i = 0; i < t; i++)
        {
            float pos = positionIds[i];
            long b = (long)i * d;
            for (int m = 0; m < d; m += 2)
            {
                op[b + m] = MathF.Sin(pos * scale / dimT[m]);
                op[b + m + 1] = MathF.Cos(pos * scale / dimT[m + 1]);
            }
        }
        return outT;
    }

    private static Tensor BuildTextAdditiveMask(bool[] attend, int t, int heads)
    {
        Tensor mask = new(new TensorShape(1, heads, t, t), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < t; i++)
                for (int j = 0; j < t; j++)
                    mp[((long)h * t + i) * t + j] = attend[(long)i * t + j] ? 0f : -1e30f;
        return mask;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (EncoderLayer l in _layers) l.Dispose();
        GC.SuppressFinalize(this);
    }

    /// <summary>One enhancer layer: fusion (bi-attention) → text enhancer → deformable image self-attention.</summary>
    private sealed class EncoderLayer : IDisposable
    {
        private readonly GroundingDinoConfig _cfg;
        private readonly BiAttentionFusion _fusion;
        private readonly TextEnhancer _textEnhancer;
        private readonly DeformableImageLayer _deformable;

        public EncoderLayer(GroundingDinoConfig cfg)
        {
            _cfg = cfg;
            _fusion = new BiAttentionFusion(cfg);
            _textEnhancer = new TextEnhancer(cfg);
            _deformable = new DeformableImageLayer(cfg);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            _fusion.LoadWeights(w, $"{prefix}.fusion_layer");
            _textEnhancer.LoadWeights(w, $"{prefix}.text_enhancer_layer");
            _deformable.LoadWeights(w, $"{prefix}.deformable_layer");
        }

        public (Tensor Vision, Tensor Text) Forward(IBackend backend, Tensor vision, Tensor visionPos, Tensor text,
            Tensor textPos, Tensor textMask, float[] refPoints, int[][] shapes, int[] levelStart)
        {
            (Tensor fv, Tensor ft) = _fusion.Forward(backend, vision, text);
            Tensor et = _textEnhancer.Forward(backend, ft, textPos, textMask);
            ft.Dispose();
            Tensor ev = _deformable.Forward(backend, fv, visionPos, refPoints, shapes, levelStart);
            fv.Dispose();
            return (ev, et);
        }

        public void Dispose()
        {
            _fusion.Dispose();
            _textEnhancer.Dispose();
            _deformable.Dispose();
        }
    }

    /// <summary>Bi-directional image↔text cross-attention with pre-LN and per-channel residual scaling.</summary>
    private sealed class BiAttentionFusion : IDisposable
    {
        private readonly GroundingDinoConfig _cfg;
        private readonly int _embed, _heads, _hd;
        private Tensor? _lnVW, _lnVB, _lnTW, _lnTB;
        private Tensor? _vProjW, _vProjB, _tProjW, _tProjB, _vvW, _vvB, _tvW, _tvB;
        // Output projections pre-scaled by the per-channel vision_param / text_param so the residual is a plain add.
        private Tensor? _ovWs, _ovBs, _otWs, _otBs;

        public BiAttentionFusion(GroundingDinoConfig cfg)
        {
            _cfg = cfg;
            _embed = cfg.EncoderFfnDim / 2;
            _heads = cfg.EncoderAttentionHeads / 2;
            _hd = _embed / _heads;
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            _lnVW = w[$"{prefix}.layer_norm_vision.weight"]; _lnVB = w[$"{prefix}.layer_norm_vision.bias"];
            _lnTW = w[$"{prefix}.layer_norm_text.weight"]; _lnTB = w[$"{prefix}.layer_norm_text.bias"];
            string a = $"{prefix}.attn";
            _vProjW = w[$"{a}.vision_proj.weight"]; _vProjB = w[$"{a}.vision_proj.bias"];
            _tProjW = w[$"{a}.text_proj.weight"]; _tProjB = w[$"{a}.text_proj.bias"];
            _vvW = w[$"{a}.values_vision_proj.weight"]; _vvB = w[$"{a}.values_vision_proj.bias"];
            _tvW = w[$"{a}.values_text_proj.weight"]; _tvB = w[$"{a}.values_text_proj.bias"];
            // Fold vision_param/text_param (per-output-channel residual gates) into the output projections:
            // vision += param ⊙ (Wx+b)  ==  vision += (param⊙W)x + (param⊙b), so it becomes a plain GPU add.
            _ovWs = ScaleRows(w[$"{a}.out_vision_proj.weight"], w[$"{prefix}.vision_param"]);
            _ovBs = ScaleVec(w[$"{a}.out_vision_proj.bias"], w[$"{prefix}.vision_param"]);
            _otWs = ScaleRows(w[$"{a}.out_text_proj.weight"], w[$"{prefix}.text_param"]);
            _otBs = ScaleVec(w[$"{a}.out_text_proj.bias"], w[$"{prefix}.text_param"]);
        }

        public (Tensor Vision, Tensor Text) Forward(IBackend backend, Tensor visionIn, Tensor textIn)
        {
            int d = _cfg.DModel, nimg = (int)visionIn.Shape[1], t = (int)textIn.Shape[1];
            int embed = _embed, heads = _heads, hd = _hd;

            Tensor vision = new(visionIn.Shape, DType.F32);
            backend.LayerNorm(vision, visionIn, _lnVW!, _lnVB!, _cfg.LayerNormEps);
            Tensor text = new(textIn.Shape, DType.F32);
            backend.LayerNorm(text, textIn, _lnTW!, _lnTB!, _cfg.LayerNormEps);

            Tensor vq = Lin(backend, vision, _vProjW!, _vProjB!, nimg, embed);
            Tensor tk = Lin(backend, text, _tProjW!, _tProjB!, t, embed);
            Tensor vv = Lin(backend, vision, _vvW!, _vvB!, nimg, embed);
            Tensor tv = Lin(backend, text, _tvW!, _tvB!, t, embed);

            // Bi-directional cross-attention as two GPU SDPA passes over the shared vq·tkᵀ score matrix:
            //   vision_out = softmax_text(vq·tkᵀ)·tv  = attn(Q=vq, K=tk, V=tv)
            //   text_out   = softmax_vision(vq·tkᵀ)·vv = attn(Q=tk, K=vq, V=vv)
            Tensor visionAttn = RtDetrAttention.MultiHead(backend, vq, tk, tv, heads, hd);  // [1, nimg, embed]
            Tensor textAttn = RtDetrAttention.MultiHead(backend, tk, vq, vv, heads, hd);    // [1, t, embed]
            vq.Dispose(); tk.Dispose(); vv.Dispose(); tv.Dispose();

            Tensor deltaV = Lin(backend, visionAttn, _ovWs!, _ovBs!, nimg, d);  // already param-scaled
            Tensor deltaT = Lin(backend, textAttn, _otWs!, _otBs!, t, d);
            visionAttn.Dispose(); textAttn.Dispose();

            Tensor visionOut = new(vision.Shape, DType.F32);
            backend.Add(visionOut, vision, deltaV);
            Tensor textOut = new(text.Shape, DType.F32);
            backend.Add(textOut, text, deltaT);
            vision.Dispose(); text.Dispose(); deltaV.Dispose(); deltaT.Dispose();
            return (visionOut, textOut);
        }

        /// <summary>Returns a copy of the <c>[rows, cols]</c> weight with each output row r multiplied by <paramref name="scale"/>[r].</summary>
        private static Tensor ScaleRows(Tensor weight, Tensor scale)
        {
            int rows = (int)weight.Shape[0], cols = (int)weight.Shape[1];
            Tensor o = new(weight.Shape, DType.F32);
            float* wp = (float*)weight.DataPointer, sp = (float*)scale.DataPointer, op = (float*)o.DataPointer;
            for (int r = 0; r < rows; r++)
            {
                float s = sp[r];
                long baseOff = (long)r * cols;
                for (int c = 0; c < cols; c++) op[baseOff + c] = wp[baseOff + c] * s;
            }
            return o;
        }

        /// <summary>Returns a copy of the length-n vector with each element i multiplied by <paramref name="scale"/>[i].</summary>
        private static Tensor ScaleVec(Tensor vec, Tensor scale)
        {
            int n = (int)vec.Shape[0];
            Tensor o = new(vec.Shape, DType.F32);
            float* vp = (float*)vec.DataPointer, sp = (float*)scale.DataPointer, op = (float*)o.DataPointer;
            for (int i = 0; i < n; i++) op[i] = vp[i] * sp[i];
            return o;
        }

        private static Tensor Lin(IBackend backend, Tensor input, Tensor w, Tensor? b, int rows, int outDim)
        {
            Tensor o = new(new TensorShape(1, rows, outDim), DType.F32);
            backend.Linear(o, input, w, b);
            return o;
        }

        public void Dispose()
        {
            _ovWs?.Dispose(); _ovBs?.Dispose(); _otWs?.Dispose(); _otBs?.Dispose();
        }
    }

    /// <summary>Text self-attention enhancer: masked MHA (4 heads) + ReLU FFN, pre/post LayerNorm residuals.</summary>
    private sealed class TextEnhancer : IDisposable
    {
        private readonly GroundingDinoConfig _cfg;
        private readonly GroundingDinoMultiheadAttention _attn;
        private Tensor? _fc1W, _fc1B, _fc2W, _fc2B, _lnBeforeW, _lnBeforeB, _lnAfterW, _lnAfterB;

        public TextEnhancer(GroundingDinoConfig cfg)
        {
            _cfg = cfg;
            _attn = new GroundingDinoMultiheadAttention(cfg.DModel, cfg.EncoderAttentionHeads / 2);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            _attn.LoadWeights(w, $"{prefix}.self_attn");
            _fc1W = w[$"{prefix}.fc1.weight"]; _fc1B = w[$"{prefix}.fc1.bias"];
            _fc2W = w[$"{prefix}.fc2.weight"]; _fc2B = w[$"{prefix}.fc2.bias"];
            _lnBeforeW = w[$"{prefix}.layer_norm_before.weight"]; _lnBeforeB = w[$"{prefix}.layer_norm_before.bias"];
            _lnAfterW = w[$"{prefix}.layer_norm_after.weight"]; _lnAfterB = w[$"{prefix}.layer_norm_after.bias"];
        }

        public Tensor Forward(IBackend backend, Tensor text, Tensor textPos, Tensor mask)
        {
            int t = (int)text.Shape[1], d = _cfg.DModel;
            Tensor qk = new(text.Shape, DType.F32);
            backend.Add(qk, text, textPos);   // queries = keys = text + pos; values = text
            Tensor attn = _attn.Forward(backend, qk, qk, text, mask);
            qk.Dispose();
            Tensor res1 = new(attn.Shape, DType.F32);
            backend.Add(res1, attn, text);
            attn.Dispose();
            Tensor h = new(res1.Shape, DType.F32);
            backend.LayerNorm(h, res1, _lnBeforeW!, _lnBeforeB!, _cfg.LayerNormEps);
            res1.Dispose();

            int inner = _cfg.EncoderFfnDim / 2;
            Tensor fc1 = new(new TensorShape(1, t, inner), DType.F32);
            backend.Linear(fc1, h, _fc1W!, _fc1B);
            backend.Clamp(fc1, fc1, 0f, float.MaxValue);   // ReLU on GPU
            Tensor fc2 = new(new TensorShape(1, t, d), DType.F32);
            backend.Linear(fc2, fc1, _fc2W!, _fc2B);
            fc1.Dispose();
            Tensor res2 = new(fc2.Shape, DType.F32);
            backend.Add(res2, fc2, h);
            fc2.Dispose();
            h.Dispose();
            Tensor outT = new(res2.Shape, DType.F32);
            backend.LayerNorm(outT, res2, _lnAfterW!, _lnAfterB!, _cfg.LayerNormEps);
            res2.Dispose();
            return outT;
        }

        public void Dispose() => _attn.Dispose();
    }

    /// <summary>Deformable image self-attention layer: MS-deformable attention + ReLU FFN.</summary>
    private sealed class DeformableImageLayer : IDisposable
    {
        private readonly GroundingDinoConfig _cfg;
        private readonly MultiscaleDeformableAttention _attn;
        private Tensor? _lnW, _lnB, _fc1W, _fc1B, _fc2W, _fc2B, _finalLnW, _finalLnB;

        public DeformableImageLayer(GroundingDinoConfig cfg)
        {
            _cfg = cfg;
            _attn = new MultiscaleDeformableAttention(cfg.DModel, cfg.EncoderAttentionHeads, cfg.NumFeatureLevels, cfg.EncoderNPoints);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            _attn.LoadWeights(w, $"{prefix}.self_attn");
            _lnW = w[$"{prefix}.self_attn_layer_norm.weight"]; _lnB = w[$"{prefix}.self_attn_layer_norm.bias"];
            _fc1W = w[$"{prefix}.fc1.weight"]; _fc1B = w[$"{prefix}.fc1.bias"];
            _fc2W = w[$"{prefix}.fc2.weight"]; _fc2B = w[$"{prefix}.fc2.bias"];
            _finalLnW = w[$"{prefix}.final_layer_norm.weight"]; _finalLnB = w[$"{prefix}.final_layer_norm.bias"];
        }

        public Tensor Forward(IBackend backend, Tensor vision, Tensor visionPos, float[] refPoints, int[][] shapes, int[] levelStart)
        {
            int nimg = (int)vision.Shape[1], d = _cfg.DModel;
            Tensor query = new(vision.Shape, DType.F32);
            backend.Add(query, vision, visionPos);
            Tensor attn = _attn.Forward(backend, query, vision, refPoints, 2, shapes, levelStart);
            query.Dispose();
            Tensor res1 = new(attn.Shape, DType.F32);
            backend.Add(res1, attn, vision);
            attn.Dispose();
            Tensor h = new(res1.Shape, DType.F32);
            backend.LayerNorm(h, res1, _lnW!, _lnB!, _cfg.LayerNormEps);
            res1.Dispose();

            Tensor fc1 = new(new TensorShape(1, nimg, _cfg.EncoderFfnDim), DType.F32);
            backend.Linear(fc1, h, _fc1W!, _fc1B);
            backend.Clamp(fc1, fc1, 0f, float.MaxValue);   // ReLU on GPU
            Tensor fc2 = new(new TensorShape(1, nimg, d), DType.F32);
            backend.Linear(fc2, fc1, _fc2W!, _fc2B);
            fc1.Dispose();
            Tensor res2 = new(fc2.Shape, DType.F32);
            backend.Add(res2, fc2, h);
            fc2.Dispose();
            h.Dispose();
            Tensor outT = new(res2.Shape, DType.F32);
            backend.LayerNorm(outT, res2, _finalLnW!, _finalLnB!, _cfg.LayerNormEps);
            res2.Dispose();
            return outT;
        }

        public void Dispose() => _attn.Dispose();
    }

    private static Tensor Copy(Tensor src)
    {
        Tensor dst = new(src.Shape, DType.F32);
        long bytes = src.ElementCount * 4;
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)dst.DataPointer, bytes, bytes);
        return dst;
    }
}

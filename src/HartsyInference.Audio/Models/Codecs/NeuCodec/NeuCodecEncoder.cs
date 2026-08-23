using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.NeuCodec;

/// <summary>NeuCodec / X-Codec2 <b>encode</b> path (16 kHz reference wav → FSQ code indices), the branch NeuTTS-air needs for voice cloning; faithful to the on-disk <c>neuphonic/neucodec</c> checkpoint (HF <c>NeuCodecModel</c>, transformers PR #47143 <c>modeling_neucodec.py</c>).</summary>
/// <remarks>
/// Two parallel branches, both 16 kHz → 50 Hz:
///
/// <list type="number">
///   <item><b>Acoustic</b> (<c>acoustic_encoder.*</c>): Conv1d(1→48,k7) stem → 5 downsample blocks
///     (each: 3 <see cref="ResidualUnit"/>-style SnakeBeta residual units with dilations 1/3/9, then a
///     SnakeBeta + strided Conv1d that doubles channels) → SnakeBeta → Conv1d(1536→1024,k3). Every SnakeBeta is
///     wrapped in BigVGAN anti-aliasing (Kaiser-sinc upsample 2× → snake → downsample 2×). Output <c>[1,1024,Ta]</c>.</item>
///   <item><b>Semantic</b> (<c>semantic_encoder.*</c>): Wav2Vec2-BERT conformer over 80-bin Kaldi log-fbank
///     (stride-2 stacked → 160-dim, per-bin z-normed). feature_projection(160→1024) → 16 conformer layers
///     (FFN·½ ▸ relative-key self-attn ▸ conv-module ▸ FFN·½, all pre-LN) → <c>[Ts,1024]</c>. Then
///     <c>semantic_adapter</c> (4 Conv1d k3 with a ReLU residual) → <c>[1,1024,Ts]</c>.</item>
/// </list>
///
/// <para>Both branches are truncated to <c>min(Ta,Ts)</c>, concatenated <c>[semantic, acoustic]</c> → 2048-dim,
/// mixed by <c>fc_encoder</c> (Linear 2048→2048), projected by <c>quantizer.project_in</c> (2048→8) and FSQ-quantized
/// with the double-<c>bound</c>+round convention the checkpoint stores. Indices pack base-<c>L</c> (dim-0 LSB),
/// mirroring <see cref="NeuCodecDecoder"/>'s dequant so encode∘decode round-trips.</para>
///
/// <para><b>Reuse:</b> Kaldi fbank via <see cref="KaldiFbankExtractor"/> (the <c>×2^15</c> Kaldi scaling cancels
/// under per-bin normalization); conv / layernorm / snake / attention via <see cref="IBackend"/>. Note the engine
/// <see cref="IBackend.Snake"/> does not exponentiate α/β, so SnakeBeta params are pre-<c>exp</c>'d on load.</para></remarks>
public sealed unsafe class NeuCodecEncoder : IDisposable
{
    private readonly NeuCodecEncoderConfig _cfg;
    private readonly KaldiFbankExtractor _fbank;
    private readonly float[] _aaFilter;                       // Kaiser-sinc anti-alias kernel (shared by up/down)
    private readonly Dictionary<int, Tensor> _aaWeights = []; // per-channel-count [C,1,K] filter tensor cache

    // Acoustic branch.
    private Tensor? _acStemW, _acStemB;                       // conv1 : 1 → 48, k7
    private AcBlock[] _acBlocks = [];
    private AaSnake _acFinalSnake;
    private Tensor? _acFinalConvW, _acFinalConvB;             // conv2 : 1536 → 1024, k3

    // Semantic branch.
    private Tensor? _semFpNormW, _semFpNormB, _semFpProjW, _semFpProjB;
    private ConformerLayer[] _semLayers = [];

    // Semantic adapter.
    private Tensor? _saConv1W, _saConv2W, _saConv2B, _saConv3W, _saConv3B, _saConv4W;

    // Fusion + quantizer.
    private Tensor? _fcW, _fcB;                               // fc_encoder : 2048 → 2048
    private Tensor? _projInW, _projInB;                       // quantizer.project_in : 2048 → 8

    private int _disposed;

    public NeuCodecEncoderConfig Config => _cfg;

    public NeuCodecEncoder(NeuCodecEncoderConfig? cfg = null)
    {
        _cfg = cfg ?? NeuCodecEncoderConfig.Default;
        _fbank = new KaldiFbankExtractor(_cfg.InputSampleRate, _cfg.NumMelBins);
        _aaFilter = KaiserSincFilter(0.5 / _cfg.AntiAliasRatio, 0.6 / _cfg.AntiAliasRatio, _cfg.AntiAliasKernel);
    }

    // ── Weight loading (raw checkpoint keys, no prefix rewriting) ─────────────────────────────────
    // `prefix` is accepted for ABI compatibility with consumers compiled against the older 2-arg signature
    // (the checkpoint keys are absolute, so it is ignored).
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string? prefix = null)
    {
        // Acoustic encoder.
        _acStemW = F32(w["acoustic_encoder.conv1.weight"]);
        _acStemB = Try(w, "acoustic_encoder.conv1.bias");
        int nStages = _cfg.DownsamplingRatios.Count;
        _acBlocks = new AcBlock[nStages];
        for (int i = 0; i < nStages; i++)
        {
            int dim = _cfg.AcousticBaseChannels * (1 << (i + 1));   // 96,192,384,768,1536
            int half = dim / 2;                                     // 48,96,192,384,768
            string bp = $"acoustic_encoder.block.{i}";
            ResUnit[] units = new ResUnit[3];
            for (int u = 0; u < 3; u++)
            {
                string up = $"{bp}.res_unit{u + 1}";
                units[u] = new ResUnit
                {
                    Dim = half,
                    Dilation = _cfg.ResidualDilations[u],
                    S1 = LoadSnake(w, $"{up}.snake1.act"),
                    C1W = F32(w[$"{up}.conv1.weight"]),
                    C1B = Try(w, $"{up}.conv1.bias"),
                    S2 = LoadSnake(w, $"{up}.snake2.act"),
                    C2W = F32(w[$"{up}.conv2.weight"]),
                    C2B = Try(w, $"{up}.conv2.bias"),
                };
            }
            _acBlocks[i] = new AcBlock
            {
                Units = units,
                InHalf = half,
                OutDim = dim,
                Stride = _cfg.DownsamplingRatios[i],
                S1 = LoadSnake(w, $"{bp}.snake1.act"),
                DownW = F32(w[$"{bp}.conv1.weight"]),
                DownB = Try(w, $"{bp}.conv1.bias"),
            };
        }
        _acFinalSnake = LoadSnake(w, "acoustic_encoder.snake1.act");
        _acFinalConvW = F32(w["acoustic_encoder.conv2.weight"]);
        _acFinalConvB = Try(w, "acoustic_encoder.conv2.bias");

        // Semantic feature projection.
        _semFpNormW = F32(w["semantic_encoder.feature_projection.layer_norm.weight"]);
        _semFpNormB = F32(w["semantic_encoder.feature_projection.layer_norm.bias"]);
        _semFpProjW = F32(w["semantic_encoder.feature_projection.projection.weight"]);
        _semFpProjB = Try(w, "semantic_encoder.feature_projection.projection.bias");

        // Semantic conformer layers.
        _semLayers = new ConformerLayer[_cfg.SemanticLayers];
        for (int i = 0; i < _cfg.SemanticLayers; i++)
        {
            string p = $"semantic_encoder.encoder.layers.{i}";
            _semLayers[i] = new ConformerLayer
            {
                Ffn1NormW = F32(w[$"{p}.ffn1_layer_norm.weight"]), Ffn1NormB = F32(w[$"{p}.ffn1_layer_norm.bias"]),
                Ffn1InW = F32(w[$"{p}.ffn1.intermediate_dense.weight"]), Ffn1InB = Try(w, $"{p}.ffn1.intermediate_dense.bias"),
                Ffn1OutW = F32(w[$"{p}.ffn1.output_dense.weight"]), Ffn1OutB = Try(w, $"{p}.ffn1.output_dense.bias"),
                AttnNormW = F32(w[$"{p}.self_attn_layer_norm.weight"]), AttnNormB = F32(w[$"{p}.self_attn_layer_norm.bias"]),
                QW = F32(w[$"{p}.self_attn.linear_q.weight"]), QB = Try(w, $"{p}.self_attn.linear_q.bias"),
                KW = F32(w[$"{p}.self_attn.linear_k.weight"]), KB = Try(w, $"{p}.self_attn.linear_k.bias"),
                VW = F32(w[$"{p}.self_attn.linear_v.weight"]), VB = Try(w, $"{p}.self_attn.linear_v.bias"),
                OW = F32(w[$"{p}.self_attn.linear_out.weight"]), OB = Try(w, $"{p}.self_attn.linear_out.bias"),
                DistEmb = F32(w[$"{p}.self_attn.distance_embedding.weight"]),
                ConvNormW = F32(w[$"{p}.conv_module.layer_norm.weight"]), ConvNormB = F32(w[$"{p}.conv_module.layer_norm.bias"]),
                ConvPw1W = F32(w[$"{p}.conv_module.pointwise_conv1.weight"]),
                ConvDwW = F32(w[$"{p}.conv_module.depthwise_conv.weight"]),
                ConvDwNormW = F32(w[$"{p}.conv_module.depthwise_layer_norm.weight"]), ConvDwNormB = F32(w[$"{p}.conv_module.depthwise_layer_norm.bias"]),
                ConvPw2W = F32(w[$"{p}.conv_module.pointwise_conv2.weight"]),
                Ffn2NormW = F32(w[$"{p}.ffn2_layer_norm.weight"]), Ffn2NormB = F32(w[$"{p}.ffn2_layer_norm.bias"]),
                Ffn2InW = F32(w[$"{p}.ffn2.intermediate_dense.weight"]), Ffn2InB = Try(w, $"{p}.ffn2.intermediate_dense.bias"),
                Ffn2OutW = F32(w[$"{p}.ffn2.output_dense.weight"]), Ffn2OutB = Try(w, $"{p}.ffn2.output_dense.bias"),
                FinalNormW = F32(w[$"{p}.final_layer_norm.weight"]), FinalNormB = F32(w[$"{p}.final_layer_norm.bias"]),
            };
        }

        // Semantic adapter.
        _saConv1W = F32(w["semantic_adapter.conv1.weight"]);
        _saConv2W = F32(w["semantic_adapter.conv2.weight"]); _saConv2B = Try(w, "semantic_adapter.conv2.bias");
        _saConv3W = F32(w["semantic_adapter.conv3.weight"]); _saConv3B = Try(w, "semantic_adapter.conv3.bias");
        _saConv4W = F32(w["semantic_adapter.conv4.weight"]);

        // Fusion + quantizer input projection.
        _fcW = F32(w["fc_encoder.weight"]); _fcB = Try(w, "fc_encoder.bias");
        _projInW = F32(w["quantizer.project_in.weight"]); _projInB = Try(w, "quantizer.project_in.bias");
    }

    /// <summary>Encodes a mono 16 kHz waveform into NeuCodec FSQ code indices (one per 320-sample / 20 ms frame).</summary>
    public int[] Encode(IBackend backend, float[] pcm16k)
    {
        if (_acStemW is null) throw new InvalidOperationException("NeuCodecEncoder weights not loaded.");

        // Reference pads the waveform (one zero, then up to a multiple of hop_length 320) before BOTH branches
        // (transformers PR #47143 NeuCodecFeatureExtractor), so acoustic (320× stride) and semantic frame counts
        // stay aligned. Feed the same padded buffer to both.
        float[] pcm = PadForHop(pcm16k);

        Tensor acoustic = AcousticEncode(backend, pcm);             // [1,1024,Ta]
        Tensor semantic = SemanticEncode(backend, pcm);             // [1,1024,Ts]
        int ta = (int)acoustic.Shape[2], ts = (int)semantic.Shape[2];
        int t = Math.Min(ta, ts);
        if (t <= 0) { acoustic.Dispose(); semantic.Dispose(); return []; }
        int dim = _cfg.HiddenSize, fus = _cfg.FusionDim;

        // Concat [semantic, acoustic] along channels → channels-last [t, 2048], truncated to the shorter branch.
        Tensor cat = new(new TensorShape(t, fus), DType.F32);
        float* cp = (float*)cat.DataPointer, ap = (float*)acoustic.DataPointer, sp = (float*)semantic.DataPointer;
        for (int i = 0; i < t; i++)
        {
            for (int c = 0; c < dim; c++) cp[(long)i * fus + c] = sp[(long)c * ts + i];
            for (int c = 0; c < dim; c++) cp[(long)i * fus + dim + c] = ap[(long)c * ta + i];
        }
        acoustic.Dispose(); semantic.Dispose();

        Tensor fused = Linear2D(backend, cat, _fcW!, _fcB, t, fus, fus); cat.Dispose();
        Tensor z = Linear2D(backend, fused, _projInW!, _projInB, t, fus, _cfg.FsqDim); fused.Dispose();

        int[] codes = QuantizeFsq(z, t); z.Dispose();
        return codes;
    }

    /// <summary>Zero-pads the waveform by one sample then up to a multiple of the hop length (16 kHz / 50 Hz = 320), matching the reference feature extractor so the 320×-strided acoustic branch yields a clean frame count.</summary>
    private float[] PadForHop(float[] pcm)
    {
        int hop = _cfg.InputSampleRate / _cfg.FrameRate;           // 320
        if (hop <= 0) return pcm;
        int target = ((pcm.Length + 1 + hop - 1) / hop) * hop;
        if (target <= pcm.Length) return pcm;
        float[] padded = new float[target];
        Array.Copy(pcm, padded, pcm.Length);
        return padded;
    }

    // ── Acoustic branch ──────────────────────────────────────────────────────────────────────────
    private Tensor AcousticEncode(IBackend backend, float[] pcm)
    {
        int n = pcm.Length;
        Tensor wav = new(new TensorShape(1, 1, n), DType.F32);
        float* wp = (float*)wav.DataPointer;
        for (int i = 0; i < n; i++) wp[i] = pcm[i];

        // Stem Conv1d(1 → 48, k7, pad3).
        int baseC = _cfg.AcousticBaseChannels;
        Tensor x = Conv1d(backend, wav, _acStemW!, _acStemB, 1, baseC, n, 7, 1, 3, 3, 1, 1); wav.Dispose();
        int t = n;

        foreach (AcBlock blk in _acBlocks)
        {
            foreach (ResUnit u in blk.Units) x = ResidualUnit(backend, x, u, t);
            x = AntiAliasedSnake(backend, x, blk.S1, blk.InHalf, t);
            int s = blk.Stride, k = 2 * s, pad = (s + 1) / 2;      // math.ceil(s/2)
            int tOut = (t + 2 * pad - k) / s + 1;
            Tensor down = Conv1d(backend, x, blk.DownW!, blk.DownB, blk.InHalf, blk.OutDim, tOut, k, s, pad, pad, 1, 1);
            x.Dispose(); x = down; t = tOut;
        }

        int finalC = baseC * (1 << _acBlocks.Length);   // 48·2^5 = 1536
        x = AntiAliasedSnake(backend, x, _acFinalSnake, finalC, t);
        Tensor feat = Conv1d(backend, x, _acFinalConvW!, _acFinalConvB, finalC, _cfg.HiddenSize, t, 3, 1, 1, 1, 1, 1);
        x.Dispose();
        return feat;   // [1,1024,Ta]
    }

    /// <summary>SnakeBeta residual unit: snake→Conv1d(k7,dil)→snake→Conv1d(k1)→+residual (SAME padding, length-preserving).</summary>
    private Tensor ResidualUnit(IBackend backend, Tensor x, ResUnit u, int t)
    {
        int d = u.Dim, pad = (7 - 1) * u.Dilation / 2;
        Tensor a1 = AntiAliasedSnake(backend, x, u.S1, d, t);
        Tensor h1 = Conv1d(backend, a1, u.C1W!, u.C1B, d, d, t, 7, 1, pad, pad, u.Dilation, 1); a1.Dispose();
        Tensor a2 = AntiAliasedSnake(backend, h1, u.S2, d, t); h1.Dispose();
        Tensor h2 = Conv1d(backend, a2, u.C2W!, u.C2B, d, d, t, 1, 1, 0, 0, 1, 1); a2.Dispose();
        Tensor sum = new(new TensorShape(1, d, t), DType.F32);
        backend.Add(sum, x, h2); x.Dispose(); h2.Dispose();
        return sum;
    }

    /// <summary>BigVGAN Activation1d: Kaiser-sinc upsample 2× → SnakeBeta → downsample 2× (length-preserving).</summary>
    private Tensor AntiAliasedSnake(IBackend backend, Tensor x, AaSnake s, int c, int t)
    {
        Tensor up = Upsample2(backend, x, c, t);                    // [1,c,2t]
        int tu = (int)up.Shape[2];
        Tensor act = new(up.Shape, DType.F32);
        backend.Snake(act, up, s.ExpAlpha!, s.ExpBeta); up.Dispose();
        Tensor down = Downsample2(backend, act, c, tu); act.Dispose();
        return down;                                                // [1,c,t]
    }

    private Tensor Upsample2(IBackend backend, Tensor x, int c, int t)
    {
        int pad = _cfg.AntiAliasKernel / _cfg.AntiAliasRatio - 1;   // 5
        Tensor xp = ReplicatePad(x, c, t, pad, pad);
        int lin = t + 2 * pad;
        int trim = pad * _cfg.AntiAliasRatio + (_cfg.AntiAliasKernel - _cfg.AntiAliasRatio) / 2;  // 15
        int outLen = (lin - 1) * _cfg.AntiAliasRatio + (_cfg.AntiAliasKernel - 1) + 1 - 2 * trim; // = 2t
        Tensor filt = AaFilter(c);
        Tensor o = new(new TensorShape(1, c, outLen), DType.F32);
        backend.ConvTranspose1d(o, xp, filt, null, _cfg.AntiAliasRatio, trim, trim, 1, c); xp.Dispose();
        float* op = (float*)o.DataPointer;
        long nEl = (long)c * outLen;
        for (long i = 0; i < nEl; i++) op[i] *= _cfg.AntiAliasRatio;  // ratio scaling
        return o;
    }

    private Tensor Downsample2(IBackend backend, Tensor x, int c, int len)
    {
        int padL = _cfg.AntiAliasKernel / 2 - 1, padR = _cfg.AntiAliasKernel / 2;  // 5,6
        Tensor xp = ReplicatePad(x, c, len, padL, padR);
        int outLen = (len + padL + padR - _cfg.AntiAliasKernel) / _cfg.AntiAliasRatio + 1;
        Tensor filt = AaFilter(c);
        Tensor o = new(new TensorShape(1, c, outLen), DType.F32);
        backend.Conv1d(o, xp, filt, null, _cfg.AntiAliasRatio, 0, 0, 1, c); xp.Dispose();
        return o;
    }

    /// <summary>Depthwise [C,1,K] filter tensor (same Kaiser-sinc kernel per channel), cached per channel count.</summary>
    private Tensor AaFilter(int c)
    {
        if (_aaWeights.TryGetValue(c, out Tensor? cached)) return cached;
        int k = _cfg.AntiAliasKernel;
        Tensor w = new(new TensorShape(c, 1, k), DType.F32);
        float* wp = (float*)w.DataPointer;
        for (int ci = 0; ci < c; ci++)
            for (int j = 0; j < k; j++) wp[(long)ci * k + j] = _aaFilter[j];
        _aaWeights[c] = w;
        return w;
    }

    // ── Semantic branch ──────────────────────────────────────────────────────────────────────────
    private Tensor SemanticEncode(IBackend backend, float[] pcm)
    {
        Tensor feats = ExtractInputFeatures(pcm);                   // [Ts,160]
        int ts = (int)feats.Shape[0];
        int dim = _cfg.HiddenSize;

        // feature_projection: LayerNorm(160) → Linear(160→1024).
        Tensor normed = new(feats.Shape, DType.F32);
        backend.LayerNorm(normed, feats, _semFpNormW!, _semFpNormB!, _cfg.LayerNormEps); feats.Dispose();
        Tensor h = Linear2D(backend, normed, _semFpProjW!, _semFpProjB, ts, _cfg.FeatureProjInputDim, dim); normed.Dispose();

        foreach (ConformerLayer l in _semLayers) h = ConformerBlock(backend, h, l, ts);

        // semantic_adapter operates channels-first [1,1024,Ts].
        Tensor hcf = new(new TensorShape(1, dim, ts), DType.F32);
        backend.Transpose2D(hcf, h, ts, dim); h.Dispose();
        Tensor adapted = SemanticAdapter(backend, hcf, ts); hcf.Dispose();
        return adapted;                                            // [1,1024,Ts]
    }

    /// <summary>Conformer block (pre-LN): x += ½·ffn1(LN); x += attn(LN); x += conv(x); x += ½·ffn2(LN); x = LN(x).</summary>
    private Tensor ConformerBlock(IBackend backend, Tensor h, ConformerLayer l, int t)
    {
        int dim = _cfg.HiddenSize, inter = _cfg.SemanticIntermediate;

        // 1. FFN1 (half-step residual).
        Tensor n1 = LN(backend, h, l.Ffn1NormW!, l.Ffn1NormB!, t, dim);
        Tensor f1 = FeedForward(backend, n1, l.Ffn1InW!, l.Ffn1InB, l.Ffn1OutW!, l.Ffn1OutB, t, dim, inter); n1.Dispose();
        AddScaled(h, f1, 0.5f, t, dim); f1.Dispose();

        // 2. Self-attention (relative-key).
        Tensor n2 = LN(backend, h, l.AttnNormW!, l.AttnNormB!, t, dim);
        Tensor attn = RelKeyAttention(backend, n2, l, t); n2.Dispose();
        AddScaled(h, attn, 1f, t, dim); attn.Dispose();

        // 3. Convolution module.
        Tensor conv = ConvModule(backend, h, l, t);
        AddScaled(h, conv, 1f, t, dim); conv.Dispose();

        // 4. FFN2 (half-step residual) + final LayerNorm.
        Tensor n4 = LN(backend, h, l.Ffn2NormW!, l.Ffn2NormB!, t, dim);
        Tensor f2 = FeedForward(backend, n4, l.Ffn2InW!, l.Ffn2InB, l.Ffn2OutW!, l.Ffn2OutB, t, dim, inter); n4.Dispose();
        AddScaled(h, f2, 0.5f, t, dim); f2.Dispose();
        Tensor outT = LN(backend, h, l.FinalNormW!, l.FinalNormB!, t, dim); h.Dispose();
        return outT;
    }

    private Tensor FeedForward(IBackend backend, Tensor x, Tensor inW, Tensor? inB, Tensor outW, Tensor? outB, int t, int dim, int inter)
    {
        Tensor hidden = Linear2D(backend, x, inW, inB, t, dim, inter);
        Tensor act = new(hidden.Shape, DType.F32);
        backend.Silu(act, hidden); hidden.Dispose();               // hidden_act = "swish" = SiLU
        Tensor o = Linear2D(backend, act, outW, outB, t, inter, dim); act.Dispose();
        return o;
    }

    /// <summary>Wav2Vec2-BERT relative-key (Shaw) bidirectional self-attention. B=1, no padding mask.</summary>
    private Tensor RelKeyAttention(IBackend backend, Tensor x, ConformerLayer l, int t)
    {
        int dim = _cfg.HiddenSize, heads = _cfg.SemanticHeads, hd = _cfg.SemanticHeadDim;
        int left = _cfg.LeftMaxPosition, right = _cfg.RightMaxPosition;
        float scale = 1f / MathF.Sqrt(hd);

        Tensor q = Linear2D(backend, x, l.QW!, l.QB, t, dim, dim);
        Tensor k = Linear2D(backend, x, l.KW!, l.KB, t, dim, dim);
        Tensor v = Linear2D(backend, x, l.VW!, l.VB, t, dim, dim);
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        float* dp = (float*)l.DistEmb!.DataPointer;                 // [left+right+1, hd]
        int distRows = (int)l.DistEmb!.Shape[0];                    // clamp target for the position index

        Tensor ctx = new(new TensorShape(t, dim), DType.F32);       // attention output, then linear_out
        float* xp = (float*)ctx.DataPointer;
        float[] scores = new float[t];
        for (int h = 0; h < heads; h++)
        {
            int hoff = h * hd;
            for (int i = 0; i < t; i++)
            {
                float* qi = qp + (long)i * dim + hoff;
                float max = float.NegativeInfinity;
                for (int j = 0; j < t; j++)
                {
                    float* kj = kp + (long)j * dim + hoff;
                    int dist = j - i;
                    if (dist < -left) dist = -left; else if (dist > right) dist = right;
                    int prow = dist + left;                          // in [0, left+right] == [0, distRows-1]
                    if (prow < 0) prow = 0; else if (prow >= distRows) prow = distRows - 1;
                    float* pe = dp + (long)prow * hd;
                    float dotK = 0f, dotP = 0f;
                    for (int c = 0; c < hd; c++) { float qc = qi[c]; dotK += qc * kj[c]; dotP += qc * pe[c]; }
                    float sc = (dotK + dotP) * scale;
                    scores[j] = sc;
                    if (sc > max) max = sc;
                }
                float sum = 0f;
                for (int j = 0; j < t; j++) { float e = MathF.Exp(scores[j] - max); scores[j] = e; sum += e; }
                float inv = 1f / sum;
                float* outi = xp + (long)i * dim + hoff;
                for (int c = 0; c < hd; c++) outi[c] = 0f;
                for (int j = 0; j < t; j++)
                {
                    float pw = scores[j] * inv;
                    float* vj = vp + (long)j * dim + hoff;
                    for (int c = 0; c < hd; c++) outi[c] += pw * vj[c];
                }
            }
        }
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor o = Linear2D(backend, ctx, l.OW!, l.OB, t, dim, dim); ctx.Dispose();
        return o;
    }

    /// <summary>Conformer convolution module: LN → pw-conv1(→2·d) → GLU → causal depthwise(k31) → LN → SiLU → pw-conv2.</summary>
    private Tensor ConvModule(IBackend backend, Tensor h, ConformerLayer l, int t)
    {
        int dim = _cfg.HiddenSize, k = _cfg.ConvDepthwiseKernel;
        Tensor n = LN(backend, h, l.ConvNormW!, l.ConvNormB!, t, dim);
        Tensor cf = new(new TensorShape(1, dim, t), DType.F32);
        backend.Transpose2D(cf, n, t, dim); n.Dispose();           // [1,dim,t]

        // pointwise_conv1: dim → 2·dim (k1, no bias), then GLU over channels.
        Tensor pw1 = Conv1d(backend, cf, l.ConvPw1W!, null, dim, 2 * dim, t, 1, 1, 0, 0, 1, 1); cf.Dispose();
        Tensor glu = new(new TensorShape(1, dim, t), DType.F32);
        float* gp = (float*)glu.DataPointer, pp = (float*)pw1.DataPointer;
        for (int c = 0; c < dim; c++)
            for (int ti = 0; ti < t; ti++)
            {
                float a = pp[(long)c * t + ti];
                float b = pp[(long)(dim + c) * t + ti];
                gp[(long)c * t + ti] = a * (1f / (1f + MathF.Exp(-b)));
            }
        pw1.Dispose();

        // Causal depthwise conv: left-pad (k-1) zeros, k31, groups=dim.
        Tensor dw = Conv1d(backend, glu, l.ConvDwW!, null, dim, dim, t, k, 1, k - 1, 0, 1, dim); glu.Dispose();

        // depthwise_layer_norm over channels (channels-last), SiLU, then pointwise_conv2.
        Tensor dwCl = new(new TensorShape(t, dim), DType.F32);
        backend.Transpose2D(dwCl, dw, dim, t); dw.Dispose();
        Tensor dwLn = new(dwCl.Shape, DType.F32);
        backend.LayerNorm(dwLn, dwCl, l.ConvDwNormW!, l.ConvDwNormB!, _cfg.LayerNormEps); dwCl.Dispose();
        Tensor dwAct = new(dwLn.Shape, DType.F32);
        backend.Silu(dwAct, dwLn); dwLn.Dispose();
        Tensor dwCf = new(new TensorShape(1, dim, t), DType.F32);
        backend.Transpose2D(dwCf, dwAct, t, dim); dwAct.Dispose();

        Tensor pw2 = Conv1d(backend, dwCf, l.ConvPw2W!, null, dim, dim, t, 1, 1, 0, 0, 1, 1); dwCf.Dispose();
        Tensor outCl = new(new TensorShape(t, dim), DType.F32);
        backend.Transpose2D(outCl, pw2, dim, t); pw2.Dispose();
        return outCl;                                              // [t,dim]
    }

    /// <summary>SemanticEncoder (semantic_adapter): x=conv1(in); x = residual_blocks(x)+x; conv4(x), where residual_blocks = ReLU→conv2→ReLU→conv3; the skip adds back the conv1 output <b>pre-ReLU</b> (the reference does <c>self.residual_blocks(x) + x</c>, and residual_blocks applies its own leading ReLU).</summary>
    private Tensor SemanticAdapter(IBackend backend, Tensor xCf, int t)
    {
        int d = _cfg.HiddenSize;
        Tensor a = Conv1d(backend, xCf, _saConv1W!, null, d, d, t, 3, 1, 1, 1, 1, 1);   // conv1 output (residual)
        Tensor reluA = new(a.Shape, DType.F32);                                          // ReLU(conv1) → conv2 input
        { float* sp = (float*)a.DataPointer, dp = (float*)reluA.DataPointer; long n = a.ElementCount;
          for (long i = 0; i < n; i++) { float v = sp[i]; dp[i] = v < 0f ? 0f : v; } }
        Tensor c2 = Conv1d(backend, reluA, _saConv2W!, _saConv2B, d, d, t, 3, 1, 1, 1, 1, 1); reluA.Dispose();
        Relu(c2);
        Tensor c3 = Conv1d(backend, c2, _saConv3W!, _saConv3B, d, d, t, 3, 1, 1, 1, 1, 1); c2.Dispose();
        Tensor sum = new(new TensorShape(1, d, t), DType.F32);
        backend.Add(sum, c3, a); c3.Dispose(); a.Dispose();                              // + pre-ReLU conv1 output
        Tensor c4 = Conv1d(backend, sum, _saConv4W!, null, d, d, t, 3, 1, 1, 1, 1, 1); sum.Dispose();
        return c4;
    }

    // ── Feature extraction: 80-bin Kaldi log-fbank → per-bin z-norm → stride-2 stacking → [Ts,160] ──
    private Tensor ExtractInputFeatures(float[] pcm)
    {
        // SeamlessM4TFeatureExtractor scales the waveform by 2^15 (Kaldi 16-bit compliance) BEFORE the
        // fbank. The constant log-offset cancels under per-bin normalization, but the mel_floor is applied
        // in the SCALED domain — so the scaling must be applied here for the floor to bite identically.
        float[] scaled = new float[pcm.Length];
        for (int i = 0; i < pcm.Length; i++) scaled[i] = pcm[i] * 32768f;
        float[,] fb = _fbank.Compute(scaled);                      // [frames,80]
        int frames = fb.GetLength(0), bins = _cfg.NumMelBins;
        if (frames < 2) throw new InvalidOperationException("NeuCodec reference audio too short for semantic features.");

        // Per-mel-bin zero-mean unit-variance normalization (unbiased var, ddof=1) across time.
        for (int m = 0; m < bins; m++)
        {
            double mean = 0; for (int f = 0; f < frames; f++) mean += fb[f, m]; mean /= frames;
            double var = 0; for (int f = 0; f < frames; f++) { double d = fb[f, m] - mean; var += d * d; }
            var /= Math.Max(1, frames - 1);
            float inv = 1f / MathF.Sqrt((float)var + 1e-7f);
            for (int f = 0; f < frames; f++) fb[f, m] = (float)((fb[f, m] - mean) * inv);
        }

        // SeamlessM4TFeatureExtractor pads mel frames to a multiple of the stride with padding_value 0.0
        // (its default), AFTER per-bin normalization, then reshapes [F,80] → [ceil(F/2),160]. Keep the
        // trailing odd frame (pad its partner with 0.0 = the per-bin mean), never reading past the last frame.
        int stride = _cfg.AntiAliasRatio;                          // 2
        int ts = (frames + stride - 1) / stride;                   // ceil
        int feat = bins * stride;
        Tensor o = new(new TensorShape(ts, feat), DType.F32);
        float* op = (float*)o.DataPointer;
        for (int g = 0; g < ts; g++)
            for (int s = 0; s < stride; s++)
            {
                int fr = g * stride + s;
                long baseIdx = (long)g * feat + (long)s * bins;
                if (fr < frames) for (int m = 0; m < bins; m++) op[baseIdx + m] = fb[fr, m];
                else for (int m = 0; m < bins; m++) op[baseIdx + m] = 0f;   // padding_value = 0.0
            }
        return o;
    }

    // ── FSQ quantize (double-bound + round + base-L pack, matching the checkpoint's stored convention) ──
    private int[] QuantizeFsq(Tensor z, int t)
    {
        int d = _cfg.FsqDim;
        Span<float> halfRange = stackalloc float[d];
        Span<float> offset = stackalloc float[d];
        Span<float> shift = stackalloc float[d];
        Span<int> half = stackalloc int[d];
        Span<int> basis = stackalloc int[d];
        const float eps = 1e-3f;
        basis[0] = 1;
        for (int i = 0; i < d; i++)
        {
            int lv = _cfg.FsqLevels[i];
            halfRange[i] = (lv - 1) * (1f + eps) / 2f;
            offset[i] = lv % 2 == 0 ? 0.5f : 0f;
            shift[i] = Atanh(offset[i] / halfRange[i]);
            half[i] = lv / 2;
            if (i > 0) basis[i] = basis[i - 1] * _cfg.FsqLevels[i - 1];
        }

        float* zp = (float*)z.DataPointer;
        int[] codes = new int[t];
        for (int ti = 0; ti < t; ti++)
        {
            int code = 0;
            long row = (long)ti * d;
            for (int i = 0; i < d; i++)
            {
                float x = zp[row + i];
                // The reference (neucodec 0.0.6 → vector_quantize_pytorch ResidualFSQ) bounds TWICE:
                // ResidualFSQ.forward does `residual = layer.bound(project_in(x))`, then FSQ.forward's
                // quantize() bounds again `round(bound(residual))`. Both bounds must be applied before
                // rounding, else every code is shifted → garbled/wrong-pitch clone.
                x = MathF.Tanh(x + shift[i]) * halfRange[i] - offset[i];   // ResidualFSQ.bound
                x = MathF.Tanh(x + shift[i]) * halfRange[i] - offset[i];   // FSQ.quantize → bound
                int digit = (int)MathF.Round(x, MidpointRounding.ToEven) + half[i];
                if (digit < 0) digit = 0; else if (digit >= _cfg.FsqLevels[i]) digit = _cfg.FsqLevels[i] - 1;
                code += digit * basis[i];
            }
            codes[ti] = code;
        }
        return codes;
    }

    // ── Small ops / helpers ──────────────────────────────────────────────────────────────────────
    private static Tensor Linear2D(IBackend backend, Tensor x, Tensor w, Tensor? b, int t, int inDim, int outDim)
    {
        Tensor o = new(new TensorShape(t, outDim), DType.F32);
        backend.Linear(o, x, w, b);
        return o;
    }

    private Tensor LN(IBackend backend, Tensor x, Tensor w, Tensor b, int t, int dim)
    {
        Tensor o = new(new TensorShape(t, dim), DType.F32);
        backend.LayerNorm(o, x, w, b, _cfg.LayerNormEps);
        return o;
    }

    private static void AddScaled(Tensor acc, Tensor add, float scale, int t, int dim)
    {
        float* accp = (float*)acc.DataPointer, addp = (float*)add.DataPointer;
        long n = (long)t * dim;
        for (long i = 0; i < n; i++) accp[i] += scale * addp[i];
    }

    private static void Relu(Tensor x)
    {
        float* p = (float*)x.DataPointer;
        long n = x.ElementCount;
        for (long i = 0; i < n; i++) if (p[i] < 0f) p[i] = 0f;
    }

    private static Tensor ReplicatePad(Tensor x, int c, int t, int padL, int padR)
    {
        int outLen = t + padL + padR;
        Tensor o = new(new TensorShape(1, c, outLen), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
        {
            float* src = xp + (long)ci * t;
            float* dst = op + (long)ci * outLen;
            for (int i = 0; i < padL; i++) dst[i] = src[0];
            for (int i = 0; i < t; i++) dst[padL + i] = src[i];
            for (int i = 0; i < padR; i++) dst[padL + t + i] = src[t - 1];
        }
        return o;
    }

    private static Tensor Conv1d(IBackend backend, Tensor x, Tensor w, Tensor? b,
        int inC, int outC, int tOut, int k, int stride, int padL, int padR, int dilation, int groups)
    {
        Tensor o = new(new TensorShape(1, outC, tOut), DType.F32);
        backend.Conv1d(o, x, w, b, stride, padL, padR, dilation, groups);
        return o;
    }

    private AaSnake LoadSnake(IReadOnlyDictionary<string, Tensor> w, string prefix)
        => new() { ExpAlpha = ExpTensor(w[$"{prefix}.alpha"]), ExpBeta = ExpTensor(w[$"{prefix}.beta"]) };

    /// <summary>Returns <c>exp(x)</c> — the engine <see cref="IBackend.Snake"/> uses α/β directly, but NeuCodec's SnakeBeta exponentiates them (<c>x + sin(exp(α)·x)²/exp(β)</c>).</summary>
    private static Tensor ExpTensor(Tensor raw)
    {
        Tensor src = F32(raw);
        Tensor o = new(src.Shape, DType.F32);
        float* sp = (float*)src.DataPointer, op = (float*)o.DataPointer;
        long n = src.ElementCount;
        for (long i = 0; i < n; i++) op[i] = MathF.Exp(sp[i]);
        return o;
    }

    private static float Atanh(float x) => 0.5f * MathF.Log((1f + x) / (1f - x));

    private static Tensor F32(Tensor t) => WhisperOps.EnsureF32(t);
    private static Tensor? Try(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    // ── Kaiser-windowed sinc filter (matches modeling_neucodec.kaiser_sinc_filter1d) ──
    private static float[] KaiserSincFilter(double cutoff, double halfWidth, int kernelSize)
    {
        bool even = kernelSize % 2 == 0;
        int halfSize = kernelSize / 2;
        double deltaF = 4 * halfWidth;
        double atten = 2.285 * (halfSize - 1) * Math.PI * deltaF + 7.95;
        double beta = atten > 50.0 ? 0.1102 * (atten - 8.7)
            : atten >= 21.0 ? 0.5842 * Math.Pow(atten - 21.0, 0.4) + 0.07886 * (atten - 21.0)
            : 0.0;
        double[] win = KaiserWindow(kernelSize, beta);
        double[] f = new double[kernelSize];
        double sum = 0;
        for (int i = 0; i < kernelSize; i++)
        {
            double ti = even ? -halfSize + i + 0.5 : i - halfSize;
            double xx = 2 * cutoff * ti;
            double sinc = xx == 0 ? 1.0 : Math.Sin(Math.PI * xx) / (Math.PI * xx);
            f[i] = 2 * cutoff * win[i] * sinc;
            sum += f[i];
        }
        float[] outF = new float[kernelSize];
        for (int i = 0; i < kernelSize; i++) outF[i] = (float)(f[i] / sum);
        return outF;
    }

    private static double[] KaiserWindow(int n, double beta)
    {
        double[] w = new double[n];
        double alpha = (n - 1) / 2.0;
        double denom = BesselI0(beta);
        for (int i = 0; i < n; i++)
        {
            double r = (i - alpha) / alpha;
            double arg = beta * Math.Sqrt(Math.Max(0.0, 1.0 - r * r));
            w[i] = BesselI0(arg) / denom;
        }
        return w;
    }

    private static double BesselI0(double x)
    {
        double sum = 1.0, term = 1.0, half = x / 2.0;
        for (int k = 1; k < 50; k++)
        {
            term *= half * half / (k * k);
            sum += term;
            if (term < 1e-12 * sum) break;
        }
        return sum;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top =
        [
            _acStemW, _acStemB, _acFinalSnake.ExpAlpha, _acFinalSnake.ExpBeta, _acFinalConvW, _acFinalConvB,
            _semFpNormW, _semFpNormB, _semFpProjW, _semFpProjB,
            _saConv1W, _saConv2W, _saConv2B, _saConv3W, _saConv3B, _saConv4W,
            _fcW, _fcB, _projInW, _projInB,
        ];
        foreach (Tensor? x in top) if (x is not null) yield return x;
        foreach (AcBlock b in _acBlocks) foreach (Tensor x in b.All()) yield return x;
        foreach (ConformerLayer l in _semLayers) foreach (Tensor x in l.All()) yield return x;
        foreach (Tensor x in _aaWeights.Values) yield return x;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    // ── Weight containers ──
    private struct AaSnake { public Tensor? ExpAlpha, ExpBeta; }

    private sealed class ResUnit
    {
        public int Dim, Dilation;
        public AaSnake S1, S2;
        public Tensor? C1W, C1B, C2W, C2B;
        public IEnumerable<Tensor> All()
        {
            Tensor?[] a = [S1.ExpAlpha, S1.ExpBeta, C1W, C1B, S2.ExpAlpha, S2.ExpBeta, C2W, C2B];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    private sealed class AcBlock
    {
        public required ResUnit[] Units;
        public int InHalf, OutDim, Stride;
        public AaSnake S1;
        public Tensor? DownW, DownB;
        public IEnumerable<Tensor> All()
        {
            foreach (ResUnit u in Units) foreach (Tensor t in u.All()) yield return t;
            Tensor?[] a = [S1.ExpAlpha, S1.ExpBeta, DownW, DownB];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    private sealed class ConformerLayer
    {
        public Tensor? Ffn1NormW, Ffn1NormB, Ffn1InW, Ffn1InB, Ffn1OutW, Ffn1OutB;
        public Tensor? AttnNormW, AttnNormB, QW, QB, KW, KB, VW, VB, OW, OB, DistEmb;
        public Tensor? ConvNormW, ConvNormB, ConvPw1W, ConvDwW, ConvDwNormW, ConvDwNormB, ConvPw2W;
        public Tensor? Ffn2NormW, Ffn2NormB, Ffn2InW, Ffn2InB, Ffn2OutW, Ffn2OutB, FinalNormW, FinalNormB;
        public IEnumerable<Tensor> All()
        {
            Tensor?[] a =
            [
                Ffn1NormW, Ffn1NormB, Ffn1InW, Ffn1InB, Ffn1OutW, Ffn1OutB,
                AttnNormW, AttnNormB, QW, QB, KW, KB, VW, VB, OW, OB, DistEmb,
                ConvNormW, ConvNormB, ConvPw1W, ConvDwW, ConvDwNormW, ConvDwNormB, ConvPw2W,
                Ffn2NormW, Ffn2NormB, Ffn2InW, Ffn2InB, Ffn2OutW, Ffn2OutB, FinalNormW, FinalNormB,
            ];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }
}

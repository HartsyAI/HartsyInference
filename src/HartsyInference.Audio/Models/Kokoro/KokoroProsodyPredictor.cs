using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Kokoro;

/// <summary>Kokoro's prosody predictor. Operates on the PLBERT-derived
/// <c>d_bert [B, T, 512]</c> + the 128-dim predictor-half style vector. Predicts:
/// <list type="number">
///   <item><b>durations</b>: <c>[B, T]</c> integer frames per phoneme</item>
///   <item><b>F0 curve</b>: <c>[B, 1, 2*T_total]</c> at twice the mel rate</item>
///   <item><b>Energy curve</b> (N): <c>[B, 1, 2*T_total]</c></item>
/// </list>
///
/// <para>Two parallel paths share the DurationEncoder front-end:</para>
/// <code>
///   d_bert → DurationEncoder(s_pred) → [B, T, 640]
///                                       │
///                          ┌────────────┴────────────┐
///                          ▼                          ▼
///                   predictor.lstm              (later, per-frame)
///                   BiLSTM(640→512)             d_bert_expanded ←─ alignment
///                   duration_proj(50)+sig+sum   │
///                          │                    ▼ concat(style)
///                          ▼                   predictor.shared (BiLSTM 640→512)
///                      durations               │
///                                              ▼
///                                    F0 chain (3× AdainResBlk1d, 1 upsample)
///                                    F0_proj 256→1
///                                    Energy chain (same)
/// </code>
///
/// <para>The pipeline that owns this class is responsible for converting durations into
/// an alignment matrix and producing <c>d_bert_expanded</c> — that's a simple repeat
/// based on integer durations and doesn't need to live here.</para></summary>
public sealed unsafe class KokoroProsodyPredictor
{
    private readonly KokoroConfig _cfg;

    // DurationEncoder: alternating BiLSTM + AdaLayerNorm × 3.
    // Each BiLSTM takes input=d_hid + style_dim = 640, output=d_hid = 512.
    private readonly BiLstm[] _durEncLstms;
    private readonly Tensor?[] _adaLnFcW;     // [1024, 128]
    private readonly Tensor?[] _adaLnFcB;

    // Duration head.
    private readonly BiLstm _durationLstm;
    private Tensor? _durProjW;     // [50, 512]
    private Tensor? _durProjB;     // [50]

    // F0 / N pathway.
    private readonly BiLstm _sharedLstm;
    private readonly KokoroAdainResBlk1d[] _f0Blocks;
    private readonly KokoroAdainResBlk1d[] _nBlocks;
    private Tensor? _f0ProjW, _f0ProjB;     // [1, 256, 1] / [1]
    private Tensor? _nProjW, _nProjB;

    public int MaxDuration => _cfg.MaxDuration;

    public KokoroProsodyPredictor(KokoroConfig cfg)
    {
        _cfg = cfg;
        int dHid = cfg.HiddenDim;     // 512
        int sty = cfg.StyleDim;       // 128

        _durEncLstms = new BiLstm[3];
        _adaLnFcW = new Tensor[3];
        _adaLnFcB = new Tensor[3];
        for (int i = 0; i < 3; i++) _durEncLstms[i] = new BiLstm(inputDim: dHid + sty, hiddenDim: dHid / 2);

        _durationLstm = new BiLstm(inputDim: dHid + sty, hiddenDim: dHid / 2);
        _sharedLstm = new BiLstm(inputDim: dHid + sty, hiddenDim: dHid / 2);

        _f0Blocks =
        [
            new KokoroAdainResBlk1d(dHid, dHid, upsample: false),
            new KokoroAdainResBlk1d(dHid, dHid / 2, upsample: true),
            new KokoroAdainResBlk1d(dHid / 2, dHid / 2, upsample: false),
        ];
        _nBlocks =
        [
            new KokoroAdainResBlk1d(dHid, dHid, upsample: false),
            new KokoroAdainResBlk1d(dHid, dHid / 2, upsample: true),
            new KokoroAdainResBlk1d(dHid / 2, dHid / 2, upsample: false),
        ];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // DurationEncoder — lstms.{0,2,4} are BiLSTMs, lstms.{1,3,5} are AdaLayerNorm fc.
        for (int i = 0; i < 3; i++)
        {
            int lstmIdx = i * 2;
            _durEncLstms[i].LoadWeights(w, $"predictor.text_encoder.lstms.{lstmIdx}");
            int adaIdx = i * 2 + 1;
            _adaLnFcW[i] = WhisperOps.EnsureF32(w[$"predictor.text_encoder.lstms.{adaIdx}.fc.weight"]);
            _adaLnFcB[i] = WhisperOps.EnsureF32(w[$"predictor.text_encoder.lstms.{adaIdx}.fc.bias"]);
        }

        _durationLstm.LoadWeights(w, "predictor.lstm");
        _durProjW = WhisperOps.EnsureF32(w["predictor.duration_proj.linear_layer.weight"]);
        _durProjB = WhisperOps.EnsureF32(w["predictor.duration_proj.linear_layer.bias"]);

        _sharedLstm.LoadWeights(w, "predictor.shared");

        for (int i = 0; i < 3; i++)
        {
            _f0Blocks[i].LoadWeights(w, $"predictor.F0.{i}");
            _nBlocks[i].LoadWeights(w, $"predictor.N.{i}");
        }

        // F0_proj / N_proj — 1×1 Conv1D 256→1. Stored as raw Conv weight, not weight-
        // normed; saved shape is [1, 256, 1] (out_channels=1, in_channels=256, k=1).
        _f0ProjW = WhisperOps.EnsureF32(w["predictor.F0_proj.weight"]);
        _f0ProjB = WhisperOps.EnsureF32(w["predictor.F0_proj.bias"]);
        _nProjW = WhisperOps.EnsureF32(w["predictor.N_proj.weight"]);
        _nProjB = WhisperOps.EnsureF32(w["predictor.N_proj.bias"]);
    }

    /// <summary>Runs the DurationEncoder + duration head. <paramref name="dBert"/> is
    /// <c>[1, T, 512]</c> (PLBERT output, already projected by bert_encoder).
    /// <paramref name="stylePred"/> is <c>[1, 128]</c>.
    ///
    /// <para>Returns <c>(durFeatures, durations)</c> where <c>durFeatures</c> is the
    /// DurationEncoder's <c>[1, T, 640]</c> output (style-concatenated — what
    /// <c>predictor.lstm</c> consumed) and <c>durations</c> is an <c>int[T]</c> array of
    /// non-negative frame counts. The caller passes <c>durFeatures</c> on to other
    /// downstream consumers when needed and uses <c>durations</c> to build the alignment
    /// matrix.</para></summary>
    public (Tensor DurFeatures, int[] Durations) PredictDurations(IBackend backend, Tensor dBert, Tensor stylePred, float speed = 1f)
    {
        int batch = (int)dBert.Shape[0];
        int t = (int)dBert.Shape[1];
        int dHid = _cfg.HiddenDim;
        int sty = _cfg.StyleDim;

        // 1. DurationEncoder: 3× (concat-with-style → BiLSTM → AdaLayerNorm).
        // Working buffer x is always [B, T, dHid] after each AdaLN; we re-concat style
        // before each BiLSTM to make the 640-dim input. The style row is broadcast over T once.
        Tensor styleRep = KokoroOps.RepeatStyleAcrossTime(backend, stylePred, batch, t, sty);
        Tensor x = AppendStyleAcrossTime(backend, dBert, styleRep);
        for (int i = 0; i < 3; i++)
        {
            // BiLSTM: [B, T, 640] → [B, T, 512].
            Tensor lstmOut = _durEncLstms[i].Forward(backend, x, batch, t);
            x.Dispose();

            // AdaLayerNorm: split fc(style) into gamma + beta, then (1+γ)*LN(x) + β.
            (Tensor gamma, Tensor beta) = KokoroOps.StyleToGammaBeta(backend, _adaLnFcW[i]!, _adaLnFcB[i]!, stylePred, dHid);
            Tensor adaLnOut = KokoroOps.ApplyAdaLayerNorm(backend, lstmOut, gamma, beta, t, dHid);
            lstmOut.Dispose();
            gamma.Dispose(); beta.Dispose();

            // Re-concat style for the next BiLSTM (or to be the final DurationEncoder
            // output — predictor.lstm expects [B, T, 640] = dHid + style).
            x = AppendStyleAcrossTime(backend, adaLnOut, styleRep);
            adaLnOut.Dispose();
        }
        styleRep.Dispose();

        // 2. predictor.lstm: BiLSTM(640 → 512).
        Tensor durLstmOut = _durationLstm.Forward(backend, x, batch, t);

        // 3. duration_proj: Linear(512 → 50) → sigmoid → sum over last dim → durations.
        int maxDur = _cfg.MaxDuration;
        Tensor durLogits = WhisperOps.ProjectLinear(backend, durLstmOut, _durProjW!, _durProjB, batch, t, dHid, maxDur);
        durLstmOut.Dispose();

        backend.Sigmoid(durLogits, durLogits);

        int[] durations = new int[t];
        float* dp = (float*)durLogits.DataPointer;
        for (int i = 0; i < t; i++)
        {
            double s = 0d;
            int row = i * maxDur;
            for (int j = 0; j < maxDur; j++) s += dp[row + j];
            // Speed knob: speed > 1 → faster (smaller durations). Banker's rounding
            // mirrors PyTorch's torch.round.
            int dur = (int)Math.Round(s / speed, MidpointRounding.ToEven);
            durations[i] = Math.Max(1, dur);
        }
        durLogits.Dispose();

        return (x, durations);
    }

    /// <summary>F0Ntrain — runs predictor.shared + F0 + N chains over the expanded
    /// <c>d_bert</c>. <paramref name="dBertExpanded"/> is <c>[1, 512, T_total]</c>
    /// channels-first (already passed through the alignment-based length regulator).
    /// <paramref name="stylePred"/> is <c>[1, 128]</c>.
    ///
    /// <para>Returns <c>(F0, N)</c> both <c>[1, 1, 2*T_total]</c> channels-first. Caller
    /// owns disposal of both tensors.</para></summary>
    public (Tensor F0, Tensor N) F0Ntrain(IBackend backend, Tensor dBertExpanded, Tensor stylePred)
    {
        int batch = (int)dBertExpanded.Shape[0];
        int dHid = _cfg.HiddenDim;
        int sty = _cfg.StyleDim;
        if ((int)dBertExpanded.Shape[1] != dHid)
            throw new ArgumentException($"F0Ntrain expects [B, {dHid}, T_total], got {dBertExpanded.Shape}.");
        int tTotal = (int)dBertExpanded.Shape[2];

        // 1. Transpose to channels-last for the BiLSTM, concat style.
        Tensor xCL = new(new TensorShape(batch, tTotal, dHid), DType.F32);
        backend.Transpose2D(xCL, dBertExpanded, dHid, tTotal);

        Tensor styleRep = KokoroOps.RepeatStyleAcrossTime(backend, stylePred, batch, tTotal, sty);
        Tensor lstmIn = AppendStyleAcrossTime(backend, xCL, styleRep);
        styleRep.Dispose();
        xCL.Dispose();

        // 2. predictor.shared BiLSTM(640 → 512).
        Tensor shared = _sharedLstm.Forward(backend, lstmIn, batch, tTotal);
        lstmIn.Dispose();

        // 3. Transpose to channels-first for the F0/N convolution chains.
        Tensor sharedCF = new(new TensorShape(batch, dHid, tTotal), DType.F32);
        backend.Transpose2D(sharedCF, shared, tTotal, dHid);
        shared.Dispose();

        // 4. F0 chain and 5. N chain, both reading sharedCF (neither block mutates its input).
        Tensor f0Out = RunProsodyChain(backend, _f0Blocks, _f0ProjW!, _f0ProjB, sharedCF, stylePred, batch);
        Tensor nOut = RunProsodyChain(backend, _nBlocks, _nProjW!, _nProjB, sharedCF, stylePred, batch);
        sharedCF.Dispose();
        return (f0Out, nOut);
    }

    /// <summary>Three AdainResBlk1d blocks then the 1×1 projection to one curve: <c>[B, 512, T] → [B, 1, 2T]</c>.</summary>
    private static Tensor RunProsodyChain(IBackend backend, KokoroAdainResBlk1d[] blocks, Tensor projW, Tensor? projB,
        Tensor input, Tensor stylePred, int batch)
    {
        Tensor x = input;
        for (int i = 0; i < blocks.Length; i++)
        {
            Tensor next = blocks[i].Forward(backend, x, stylePred);
            if (i > 0) x.Dispose();
            x = next;
        }
        Tensor output = new(new TensorShape(batch, 1, (int)x.Shape[2]), DType.F32);
        backend.Conv1d(output, x, projW, projB, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        if (blocks.Length > 0) x.Dispose();
        return output;
    }

    /// <summary><c>[B, T, dHid]</c> ++ <c>[B, T, styleDim]</c> along the last dim → <c>[B, T, dHid + styleDim]</c>, on device.</summary>
    private static Tensor AppendStyleAcrossTime(IBackend backend, Tensor x, Tensor styleRep)
    {
        Tensor output = new(new TensorShape(x.Shape[0], x.Shape[1], x.Shape[2] + styleRep.Shape[2]), DType.F32);
        backend.Concat(output, [x, styleRep], dim: 2);
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < 3; i++)
        {
            foreach (Tensor x in _durEncLstms[i].EnumerateWeights()) yield return x;
            if (_adaLnFcW[i] is not null) yield return _adaLnFcW[i]!;
            if (_adaLnFcB[i] is not null) yield return _adaLnFcB[i]!;
        }
        foreach (Tensor x in _durationLstm.EnumerateWeights()) yield return x;
        if (_durProjW is not null) yield return _durProjW;
        if (_durProjB is not null) yield return _durProjB;
        foreach (Tensor x in _sharedLstm.EnumerateWeights()) yield return x;
        for (int i = 0; i < 3; i++)
        {
            foreach (Tensor x in _f0Blocks[i].EnumerateWeights()) yield return x;
            foreach (Tensor x in _nBlocks[i].EnumerateWeights()) yield return x;
        }
        Tensor?[] tail = [_f0ProjW, _f0ProjB, _nProjW, _nProjB];
        foreach (Tensor? x in tail) if (x is not null) yield return x;
    }
}

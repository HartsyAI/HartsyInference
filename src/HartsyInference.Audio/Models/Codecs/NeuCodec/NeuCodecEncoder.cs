using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.NeuCodec;

/// <summary>NeuCodec encoder (audio → FSQ tokens) — the reference-cloning path NeuTTS needs. This is the
/// BigCodec acoustic encoder: a SnakeBeta-residual Conv1d stack (ngf 48, downsample ratios <c>[2,2,4,4,5]</c>
/// → 320× = 50 Hz at 16 kHz) producing acoustic features, followed by <c>fc_prior</c> (project to the FSQ code
/// dimension) and FSQ quantization (<c>levels = [4]^8</c> = 65536 codes, via <see cref="Fsq.Quantize"/>).
///
/// <para>The w2v-bert semantic branch is an intentional documented no-op here: NeuCodec sums an acoustic and a
/// semantic embedding before <c>fc_prior</c>, but the acoustic branch alone reconstructs the reference well
/// enough for token-based cloning, and the semantic path requires a separate 600M w2v-bert front-end that is
/// out of scope for this pass. When that front-end lands, sum its projected features into <c>x</c> before
/// <c>fc_prior</c>.</para>
///
/// <para>Reuses the shared <see cref="IBackend"/> Conv1d / Snake ops and <see cref="Fsq"/>. Each downsample
/// stage is: <c>N × ResidualUnit(SnakeBeta → Conv1d(dilated) → SnakeBeta → Conv1d(1))</c> then
/// <c>SnakeBeta → strided Conv1d(2×ratio kernel)</c>. Structural scaffold (validation-pending): the exact
/// per-stage channel doubling and SnakeBeta alpha/beta init are checkpoint-reconcile items.</para></summary>
public sealed unsafe class NeuCodecEncoder : IDisposable
{
    private readonly NeuCodecEncoderConfig _cfg;
    private readonly int[] _downRatios;
    private readonly Stage[] _stages;

    private Tensor? _stemW, _stemB;                 // Conv1d(1 → ngf, k7, p3)
    private Tensor? _finalSnakeAlpha, _finalSnakeBeta;
    private Tensor? _finalConvW, _finalConvB;       // Conv1d(lastC → fcIn, k3, p1)
    private Tensor? _fcPriorW, _fcPriorB;           // Linear(fcIn → FsqDim)
    private int _disposed;

    public NeuCodecEncoderConfig Config => _cfg;

    public NeuCodecEncoder(NeuCodecEncoderConfig cfg)
    {
        _cfg = cfg;
        _downRatios = [.. cfg.DownRatios];
        _stages = new Stage[_downRatios.Length];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "encoder")
    {
        _stemW = WhisperOps.EnsureF32(w[$"{prefix}.stem.weight"]);
        _stemB = TryGet(w, $"{prefix}.stem.bias");

        int dim = _cfg.Ngf;
        for (int i = 0; i < _stages.Length; i++)
        {
            int outDim = dim * 2;
            string sp = $"{prefix}.stages.{i}";
            ResUnit[] units = new ResUnit[_cfg.ResidualUnitsPerStage];
            for (int j = 0; j < units.Length; j++)
            {
                string up = $"{sp}.res.{j}";
                units[j] = new ResUnit
                {
                    Snake1Alpha = WhisperOps.EnsureF32(w[$"{up}.snake1.alpha"]).Reshape(new TensorShape(1, dim, 1)),
                    Snake1Beta = SnakeBeta(w, $"{up}.snake1.beta", dim),
                    Conv1W = WhisperOps.EnsureF32(w[$"{up}.conv1.weight"]),
                    Conv1B = TryGet(w, $"{up}.conv1.bias"),
                    Snake2Alpha = WhisperOps.EnsureF32(w[$"{up}.snake2.alpha"]).Reshape(new TensorShape(1, dim, 1)),
                    Snake2Beta = SnakeBeta(w, $"{up}.snake2.beta", dim),
                    Conv2W = WhisperOps.EnsureF32(w[$"{up}.conv2.weight"]),
                    Conv2B = TryGet(w, $"{up}.conv2.bias"),
                    Dilation = _cfg.ResidualDilations[j % _cfg.ResidualDilations.Count],
                    C = dim,
                };
            }
            _stages[i] = new Stage
            {
                Units = units,
                DownSnakeAlpha = WhisperOps.EnsureF32(w[$"{sp}.down_snake.alpha"]).Reshape(new TensorShape(1, dim, 1)),
                DownSnakeBeta = SnakeBeta(w, $"{sp}.down_snake.beta", dim),
                DownConvW = WhisperOps.EnsureF32(w[$"{sp}.down.weight"]),
                DownConvB = TryGet(w, $"{sp}.down.bias"),
                OutC = outDim,
                Ratio = _downRatios[i],
            };
            dim = outDim;
        }

        _finalSnakeAlpha = WhisperOps.EnsureF32(w[$"{prefix}.final_snake.alpha"]).Reshape(new TensorShape(1, dim, 1));
        _finalSnakeBeta = SnakeBeta(w, $"{prefix}.final_snake.beta", dim);
        _finalConvW = WhisperOps.EnsureF32(w[$"{prefix}.final_conv.weight"]);
        _finalConvB = TryGet(w, $"{prefix}.final_conv.bias");
        _fcPriorW = WhisperOps.EnsureF32(w["fc_prior.weight"]);
        _fcPriorB = TryGet(w, "fc_prior.bias");
    }

    /// <summary>Encodes 16 kHz mono PCM into FSQ indices (one per 320-sample frame).</summary>
    public int[] Encode(IBackend backend, float[] pcm16k)
    {
        if (_stemW is null) throw new InvalidOperationException("NeuCodecEncoder weights not loaded.");

        int tPcm = pcm16k.Length;
        Tensor pcm = new(new TensorShape(1, 1, tPcm), DType.F32);
        float* pp = (float*)pcm.DataPointer;
        for (int i = 0; i < tPcm; i++) pp[i] = pcm16k[i];

        // Stem Conv1d(1 → ngf, k7, p3).
        int stemPad = _cfg.StemKernel / 2;
        int tStem = tPcm + 2 * stemPad - (_cfg.StemKernel - 1);
        Tensor x = new(new TensorShape(1, _cfg.Ngf, tStem), DType.F32);
        backend.Conv1d(x, pcm, _stemW!, _stemB, 1, stemPad, stemPad, 1, 1);
        pcm.Dispose();

        int t = tStem;
        foreach (Stage stage in _stages)
        {
            foreach (ResUnit unit in stage.Units)
            {
                Tensor next = ForwardResUnit(backend, x, unit, t);
                x.Dispose();
                x = next;
            }
            // SnakeBeta then strided downsample conv (k = 2*ratio, stride = ratio, pad = ceil(ratio/2)).
            Tensor act = new(x.Shape, DType.F32);
            backend.Snake(act, x, stage.DownSnakeAlpha!, stage.DownSnakeBeta);
            x.Dispose();
            int ratio = stage.Ratio;
            int kDown = 2 * ratio;
            int padDown = (ratio + 1) / 2;
            int tDown = (t + 2 * padDown - (kDown - 1) - 1) / ratio + 1;
            Tensor down = new(new TensorShape(1, stage.OutC, tDown), DType.F32);
            backend.Conv1d(down, act, stage.DownConvW!, stage.DownConvB, ratio, padDown, padDown, 1, 1);
            act.Dispose();
            x = down;
            t = tDown;
        }

        // Final SnakeBeta + Conv1d → fcIn channels.
        Tensor fAct = new(x.Shape, DType.F32);
        backend.Snake(fAct, x, _finalSnakeAlpha!, _finalSnakeBeta);
        x.Dispose();
        int fcPad = _cfg.FinalKernel / 2;
        int tFinal = t + 2 * fcPad - (_cfg.FinalKernel - 1);
        Tensor feat = new(new TensorShape(1, _cfg.FcInDim, tFinal), DType.F32);
        backend.Conv1d(feat, fAct, _finalConvW!, _finalConvB, 1, fcPad, fcPad, 1, 1);
        fAct.Dispose();

        // fc_prior: Linear(fcIn → FsqDim). Operate channels-last [1, T, fcIn].
        Tensor featCl = new(new TensorShape(1, tFinal, _cfg.FcInDim), DType.F32);
        backend.Transpose2D(featCl, feat, _cfg.FcInDim, tFinal);
        feat.Dispose();
        Tensor z = WhisperOps.ProjectLinear(backend, featCl, _fcPriorW!, _fcPriorB, 1, tFinal, _cfg.FcInDim, _cfg.FsqDim);
        featCl.Dispose();

        // FSQ quantize → [1, T] indices.
        Tensor codes = new(new TensorShape(1, tFinal), DType.I32);
        Fsq.Quantize(codes, z, [.. _cfg.FsqLevels]);
        z.Dispose();
        int* cp = (int*)codes.DataPointer;
        int[] result = new int[tFinal];
        for (int i = 0; i < tFinal; i++) result[i] = cp[i];
        codes.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_stemW, _stemB, _finalSnakeAlpha, _finalSnakeBeta, _finalConvW, _finalConvB, _fcPriorW, _fcPriorB];
        foreach (Tensor? x in top) if (x is not null) yield return x;
        foreach (Stage s in _stages) foreach (Tensor t in s.All()) yield return t;
    }

    /// <summary>One BigCodec residual unit: SnakeBeta → dilated Conv1d → SnakeBeta → Conv1d(k1) → +residual.</summary>
    private Tensor ForwardResUnit(IBackend backend, Tensor x, ResUnit u, int t)
    {
        Tensor a1 = new(x.Shape, DType.F32);
        backend.Snake(a1, x, u.Snake1Alpha!, u.Snake1Beta);
        int k = _cfg.ResidualKernel;
        int pad = (k - 1) / 2 * u.Dilation;
        Tensor h1 = new(new TensorShape(1, u.C, t), DType.F32);
        backend.Conv1d(h1, a1, u.Conv1W!, u.Conv1B, 1, pad, pad, u.Dilation, 1);
        a1.Dispose();
        Tensor a2 = new(h1.Shape, DType.F32);
        backend.Snake(a2, h1, u.Snake2Alpha!, u.Snake2Beta);
        h1.Dispose();
        Tensor h2 = new(new TensorShape(1, u.C, t), DType.F32);
        backend.Conv1d(h2, a2, u.Conv2W!, u.Conv2B, 1, 0, 0, 1, 1);
        a2.Dispose();
        Tensor sum = new(x.Shape, DType.F32);
        backend.Add(sum, x, h2);
        h2.Dispose();
        return sum;
    }

    private static Tensor? SnakeBeta(IReadOnlyDictionary<string, Tensor> w, string key, int dim)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t).Reshape(new TensorShape(1, dim, 1)) : null;

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private sealed class Stage
    {
        public required ResUnit[] Units { get; init; }
        public Tensor? DownSnakeAlpha, DownSnakeBeta, DownConvW, DownConvB;
        public int OutC, Ratio;
        public IEnumerable<Tensor> All()
        {
            foreach (ResUnit u in Units) foreach (Tensor t in u.All()) yield return t;
            Tensor?[] a = [DownSnakeAlpha, DownSnakeBeta, DownConvW, DownConvB];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    private sealed class ResUnit
    {
        public Tensor? Snake1Alpha, Snake1Beta, Conv1W, Conv1B, Snake2Alpha, Snake2Beta, Conv2W, Conv2B;
        public int Dilation, C;
        public IEnumerable<Tensor> All()
        {
            Tensor?[] a = [Snake1Alpha, Snake1Beta, Conv1W, Conv1B, Snake2Alpha, Snake2Beta, Conv2W, Conv2B];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }
}

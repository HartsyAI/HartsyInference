using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Snac;

/// <summary>SNAC encoder. Structure parallels <see cref="Dac.DacEncoder"/>:
/// <code>
///   stem: WNConv1d(1 → encoder_dim, k=7, padding=3)
///   for stride in encoder_rates:
///     d *= 2
///     EncoderBlock(d, stride) = 3 × SnacResidualUnit + Snake + WNConv1d down
///   Snake + WNConv1d(d_max → latent_dim, k=7, padding=3)
/// </code>
/// SNAC's final projection uses a kernel of 7 (vs DAC's 3); padding adjusts accordingly.</summary>
internal sealed unsafe class SnacEncoder
{
    private readonly SnacConfig _cfg;
    private readonly string _prefix;
    private readonly int _nStages;
    private readonly int[] _strides;

    private Tensor? _stemW;
    private Tensor? _stemB;
    private readonly SnacResidualUnit[][] _stageUnits;
    private readonly Tensor?[] _downsampleSnakeAlpha;
    private readonly Tensor?[] _downsampleW;
    private readonly Tensor?[] _downsampleB;
    private Tensor? _finalSnakeAlpha;
    private Tensor? _finalProjW;
    private Tensor? _finalProjB;

    public SnacEncoder(SnacConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;
        _nStages = cfg.EncoderRates.Count;
        _strides = [.. cfg.EncoderRates];

        _stageUnits = new SnacResidualUnit[_nStages][];
        _downsampleSnakeAlpha = new Tensor?[_nStages];
        _downsampleW = new Tensor?[_nStages];
        _downsampleB = new Tensor?[_nStages];

        int dim = cfg.EncoderDim;
        for (int i = 0; i < _nStages; i++)
        {
            dim *= 2;
            int innerDim = dim / 2;
            _stageUnits[i] = new SnacResidualUnit[cfg.ResidualDilations.Count];
            for (int j = 0; j < cfg.ResidualDilations.Count; j++)
            {
                _stageUnits[i][j] = new SnacResidualUnit(
                    prefix: $"{prefix}.block.{i + 1}.block.{j}",
                    dim: innerDim,
                    kernel: cfg.ResidualKernelSize,
                    dilation: cfg.ResidualDilations[j]);
            }
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _stemW = LoadFusedWeight(w, $"{_prefix}.block.0");
        _stemB = WhisperOps.EnsureF32(w[$"{_prefix}.block.0.bias"]);

        int dim = _cfg.EncoderDim;
        for (int i = 0; i < _nStages; i++)
        {
            dim *= 2;
            int innerDim = dim / 2;
            for (int j = 0; j < _cfg.ResidualDilations.Count; j++)
                _stageUnits[i][j].LoadWeights(w);

            int snakeIdx = _cfg.ResidualDilations.Count;
            int convIdx = snakeIdx + 1;
            _downsampleSnakeAlpha[i] = WhisperOps.EnsureF32(w[$"{_prefix}.block.{i + 1}.block.{snakeIdx}.alpha"]).Reshape(new TensorShape(innerDim));
            _downsampleW[i] = LoadFusedWeight(w, $"{_prefix}.block.{i + 1}.block.{convIdx}");
            _downsampleB[i] = WhisperOps.EnsureF32(w[$"{_prefix}.block.{i + 1}.block.{convIdx}.bias"]);
        }

        int finalDim = _cfg.EncoderDim * (1 << _nStages);
        _finalSnakeAlpha = WhisperOps.EnsureF32(w[$"{_prefix}.block.{_nStages + 1}.alpha"]).Reshape(new TensorShape(finalDim));
        _finalProjW = LoadFusedWeight(w, $"{_prefix}.block.{_nStages + 2}");
        _finalProjB = WhisperOps.EnsureF32(w[$"{_prefix}.block.{_nStages + 2}.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor pcm, int batch, int tPcm)
    {
        if (_stemW is null) throw new InvalidOperationException("SnacEncoder weights not loaded.");

        int stemPad = _cfg.StemKernelSize / 2;
        int tStem = tPcm + 2 * stemPad - (_cfg.StemKernelSize - 1);
        Tensor x = new(new TensorShape(batch, _cfg.EncoderDim, tStem), DType.F32);
        backend.Conv1d(x, pcm, _stemW!, _stemB,
            stride: 1, padLeft: stemPad, padRight: stemPad, dilation: 1, groups: 1);

        int t = tStem;
        int dim = _cfg.EncoderDim;

        for (int i = 0; i < _nStages; i++)
        {
            int innerDim = dim;
            foreach (SnacResidualUnit unit in _stageUnits[i])
            {
                Tensor next = unit.Forward(backend, x, batch, t);
                x.Dispose();
                x = next;
                t = (int)x.Shape[2];
            }

            Tensor snk = new(x.Shape, DType.F32);
            backend.Snake(snk, x, _downsampleSnakeAlpha[i]!, null);
            x.Dispose();
            x = snk;

            int stride = _strides[i];
            int kDown = 2 * stride;
            int padDown = (stride + 1) / 2;
            int dimOut = innerDim * 2;
            int tDown = (t + 2 * padDown - (kDown - 1) - 1) / stride + 1;
            Tensor down = new(new TensorShape(batch, dimOut, tDown), DType.F32);
            backend.Conv1d(down, x, _downsampleW[i]!, _downsampleB[i],
                stride: stride, padLeft: padDown, padRight: padDown, dilation: 1, groups: 1);
            x.Dispose();
            x = down;
            t = tDown;
            dim = dimOut;
        }

        Tensor preProj = new(x.Shape, DType.F32);
        backend.Snake(preProj, x, _finalSnakeAlpha!, null);
        x.Dispose();

        int finalPad = _cfg.StemKernelSize / 2;
        int tFinal = t + 2 * finalPad - (_cfg.StemKernelSize - 1);
        Tensor latent = new(new TensorShape(batch, _cfg.LatentDim, tFinal), DType.F32);
        backend.Conv1d(latent, preProj, _finalProjW!, _finalProjB,
            stride: 1, padLeft: finalPad, padRight: finalPad, dilation: 1, groups: 1);
        preProj.Dispose();
        return latent;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_stemW is not null) yield return _stemW;
        if (_stemB is not null) yield return _stemB;
        for (int i = 0; i < _nStages; i++)
        {
            foreach (SnacResidualUnit unit in _stageUnits[i])
                foreach (Tensor t in unit.EnumerateWeights()) yield return t;
            if (_downsampleSnakeAlpha[i] is not null) yield return _downsampleSnakeAlpha[i]!;
            if (_downsampleW[i] is not null) yield return _downsampleW[i]!;
            if (_downsampleB[i] is not null) yield return _downsampleB[i]!;
        }
        if (_finalSnakeAlpha is not null) yield return _finalSnakeAlpha;
        if (_finalProjW is not null) yield return _finalProjW;
        if (_finalProjB is not null) yield return _finalProjB;
    }

    private static Tensor LoadFusedWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor g = WhisperOps.EnsureF32(w[$"{prefix}.weight_g"]);
        Tensor v = WhisperOps.EnsureF32(w[$"{prefix}.weight_v"]);
        return WeightNormFusion.Fuse(g, v);
    }
}

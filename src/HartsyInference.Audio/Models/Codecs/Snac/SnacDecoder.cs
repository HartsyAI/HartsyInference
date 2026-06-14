using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Snac;

/// <summary>SNAC decoder. Mirrors <see cref="Dac.DacDecoder"/> structurally but skips
/// the final <c>tanh</c> — SNAC reconstructions are emitted in raw float, the caller
/// applies any soft-clipping if needed.</summary>
internal sealed class SnacDecoder
{
    private readonly SnacConfig _cfg;
    private readonly string _prefix;
    private readonly int _nStages;
    private readonly int[] _strides;

    private Tensor? _initialW;
    private Tensor? _initialB;
    private readonly Tensor?[] _stageSnakeAlpha;
    private readonly Tensor?[] _stageUpW;
    private readonly Tensor?[] _stageUpB;
    private readonly SnacResidualUnit[][] _stageUnits;
    private Tensor? _finalSnakeAlpha;
    private Tensor? _finalConvW;
    private Tensor? _finalConvB;

    public SnacDecoder(SnacConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;
        _nStages = cfg.DecoderRates.Count;
        _strides = [.. cfg.DecoderRates];

        _stageSnakeAlpha = new Tensor?[_nStages];
        _stageUpW = new Tensor?[_nStages];
        _stageUpB = new Tensor?[_nStages];
        _stageUnits = new SnacResidualUnit[_nStages][];

        int dim = cfg.DecoderDim;
        for (int i = 0; i < _nStages; i++)
        {
            int outDim = dim / 2;
            _stageUnits[i] = new SnacResidualUnit[cfg.ResidualDilations.Count];
            for (int j = 0; j < cfg.ResidualDilations.Count; j++)
            {
                _stageUnits[i][j] = new SnacResidualUnit(
                    prefix: $"{prefix}.model.{i + 1}.block.{j + 2}",
                    dim: outDim,
                    kernel: cfg.ResidualKernelSize,
                    dilation: cfg.ResidualDilations[j]);
            }
            dim = outDim;
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _initialW = LoadFusedWeight(w, $"{_prefix}.model.0");
        _initialB = WhisperOps.EnsureF32(w[$"{_prefix}.model.0.bias"]);

        int dim = _cfg.DecoderDim;
        for (int i = 0; i < _nStages; i++)
        {
            _stageSnakeAlpha[i] = WhisperOps.EnsureF32(w[$"{_prefix}.model.{i + 1}.block.0.alpha"]).Reshape(new TensorShape(dim));
            _stageUpW[i] = LoadFusedTransposeWeight(w, $"{_prefix}.model.{i + 1}.block.1");
            _stageUpB[i] = WhisperOps.EnsureF32(w[$"{_prefix}.model.{i + 1}.block.1.bias"]);

            for (int j = 0; j < _cfg.ResidualDilations.Count; j++)
                _stageUnits[i][j].LoadWeights(w);

            dim /= 2;
        }

        _finalSnakeAlpha = WhisperOps.EnsureF32(w[$"{_prefix}.model.{_nStages + 1}.alpha"]).Reshape(new TensorShape(dim));
        _finalConvW = LoadFusedWeight(w, $"{_prefix}.model.{_nStages + 2}");
        _finalConvB = WhisperOps.EnsureF32(w[$"{_prefix}.model.{_nStages + 2}.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor latent, int batch, int tFrames)
    {
        if (_initialW is null) throw new InvalidOperationException("SnacDecoder weights not loaded.");

        int initPad = _cfg.StemKernelSize / 2;
        int tInit = tFrames + 2 * initPad - (_cfg.StemKernelSize - 1);
        Tensor x = new(new TensorShape(batch, _cfg.DecoderDim, tInit), DType.F32);
        backend.Conv1d(x, latent, _initialW!, _initialB,
            stride: 1, padLeft: initPad, padRight: initPad, dilation: 1, groups: 1);

        int t = tInit;
        int dim = _cfg.DecoderDim;

        for (int i = 0; i < _nStages; i++)
        {
            Tensor snk = new(x.Shape, DType.F32);
            backend.Snake(snk, x, _stageSnakeAlpha[i]!, null);
            x.Dispose();
            x = snk;

            int stride = _strides[i];
            int kUp = 2 * stride;
            int padding = (stride + 1) / 2;
            int outputPadding = stride % 2;
            int padLeft = padding;
            int padRight = padding - outputPadding;
            int dimOut = dim / 2;
            int tUp = (t - 1) * stride + kUp - padLeft - padRight;
            Tensor up = new(new TensorShape(batch, dimOut, tUp), DType.F32);
            backend.ConvTranspose1d(up, x, _stageUpW[i]!, _stageUpB[i],
                stride: stride, padLeft: padLeft, padRight: padRight, dilation: 1);
            x.Dispose();
            x = up;
            t = tUp;
            dim = dimOut;

            foreach (SnacResidualUnit unit in _stageUnits[i])
            {
                Tensor next = unit.Forward(backend, x, batch, t);
                x.Dispose();
                x = next;
                t = (int)x.Shape[2];
            }
        }

        Tensor activated = new(x.Shape, DType.F32);
        backend.Snake(activated, x, _finalSnakeAlpha!, null);
        x.Dispose();

        int finalPad = _cfg.DecoderFinalKernelSize / 2;
        int tOut = t + 2 * finalPad - (_cfg.DecoderFinalKernelSize - 1);
        Tensor pcm = new(new TensorShape(batch, _cfg.Channels, tOut), DType.F32);
        backend.Conv1d(pcm, activated, _finalConvW!, _finalConvB,
            stride: 1, padLeft: finalPad, padRight: finalPad, dilation: 1, groups: 1);
        activated.Dispose();
        return pcm;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_initialW is not null) yield return _initialW;
        if (_initialB is not null) yield return _initialB;
        for (int i = 0; i < _nStages; i++)
        {
            if (_stageSnakeAlpha[i] is not null) yield return _stageSnakeAlpha[i]!;
            if (_stageUpW[i] is not null) yield return _stageUpW[i]!;
            if (_stageUpB[i] is not null) yield return _stageUpB[i]!;
            foreach (SnacResidualUnit unit in _stageUnits[i])
                foreach (Tensor t in unit.EnumerateWeights()) yield return t;
        }
        if (_finalSnakeAlpha is not null) yield return _finalSnakeAlpha;
        if (_finalConvW is not null) yield return _finalConvW;
        if (_finalConvB is not null) yield return _finalConvB;
    }

    private static Tensor LoadFusedWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor g = WhisperOps.EnsureF32(w[$"{prefix}.weight_g"]);
        Tensor v = WhisperOps.EnsureF32(w[$"{prefix}.weight_v"]);
        return WeightNormFusion.Fuse(g, v);
    }

    private static Tensor LoadFusedTransposeWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor g = WhisperOps.EnsureF32(w[$"{prefix}.weight_g"]);
        Tensor v = WhisperOps.EnsureF32(w[$"{prefix}.weight_v"]);
        return WeightNormFusionT.Fuse(g, v);
    }
}

using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Codecs.Dac;

/// <summary>DAC encoder. Mirrors the descript-audio-codec encoder:
/// <code>
///   stem: WNConv1d(1 → encoder_dim, k=7, padding=3)        [seq idx 0]
///   for stride in encoder_rates:
///     d *= 2
///     EncoderBlock(d, stride)                              [seq idx 1..N]
///       = 3 × ResidualUnit(d/2, dilation=[1,3,9])
///         + Snake1d(d/2)
///         + WNConv1d(d/2 → d, k=2*stride, stride=stride, padding=ceil(stride/2))
///   Snake1d(d_max)                                          [seq idx N+1]
///   WNConv1d(d_max → latent_dim, k=3, padding=1)            [seq idx N+2]
/// </code>
///
/// <para>Output is channels-first <c>[B, latent_dim, T_pcm/hop]</c> where
/// <c>hop = product(encoder_rates) = 512</c> for the 44.1 kHz model (frame rate ≈ 86 Hz).
/// All padding is symmetric.</para>
///
/// <para>State-dict prefix is <c>encoder.block.*</c>. The <see cref="DacResidualUnit"/>
/// instances under each EncoderBlock pick up their own nested keys via the per-block
/// <c>.block.</c> Sequential path.</para></summary>
internal sealed unsafe class DacEncoder
{
    private readonly DacConfig _cfg;
    private readonly string _prefix;
    private readonly int _nStages;
    private readonly int[] _strides;

    private Tensor? _stemW;
    private Tensor? _stemB;
    private readonly DacResidualUnit[][] _stageUnits;
    private readonly Tensor?[] _downsampleSnakeAlpha;
    private readonly Tensor?[] _downsampleW;
    private readonly Tensor?[] _downsampleB;
    private Tensor? _finalSnakeAlpha;
    private Tensor? _finalProjW;
    private Tensor? _finalProjB;

    public DacEncoder(DacConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;
        _nStages = cfg.EncoderRates.Count;
        _strides = [.. cfg.EncoderRates];

        _stageUnits = new DacResidualUnit[_nStages][];
        _downsampleSnakeAlpha = new Tensor?[_nStages];
        _downsampleW = new Tensor?[_nStages];
        _downsampleB = new Tensor?[_nStages];

        int dim = cfg.EncoderDim;
        for (int i = 0; i < _nStages; i++)
        {
            dim *= 2;
            int innerDim = dim / 2;     // residual units operate at dim/2 then the conv lifts to dim
            _stageUnits[i] = new DacResidualUnit[cfg.ResidualDilations.Count];
            for (int j = 0; j < cfg.ResidualDilations.Count; j++)
            {
                _stageUnits[i][j] = new DacResidualUnit(
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
            // Residual units at encoder.block.{i+1}.block.{0..2}.*
            for (int j = 0; j < _cfg.ResidualDilations.Count; j++)
                _stageUnits[i][j].LoadWeights(w);

            // Snake + downsample conv at encoder.block.{i+1}.block.{n_res} and .{n_res+1}.
            int snakeIdx = _cfg.ResidualDilations.Count;
            int convIdx = snakeIdx + 1;
            _downsampleSnakeAlpha[i] = WhisperOps.EnsureF32(w[$"{_prefix}.block.{i + 1}.block.{snakeIdx}.alpha"]).Reshape(new TensorShape(innerDim));
            _downsampleW[i] = LoadFusedWeight(w, $"{_prefix}.block.{i + 1}.block.{convIdx}");
            _downsampleB[i] = WhisperOps.EnsureF32(w[$"{_prefix}.block.{i + 1}.block.{convIdx}.bias"]);
        }

        // Final snake + projection — at encoder.block.{n_stages+1} and .{n_stages+2}.
        int finalDim = _cfg.EncoderDim * (1 << _nStages);
        _finalSnakeAlpha = WhisperOps.EnsureF32(w[$"{_prefix}.block.{_nStages + 1}.alpha"]).Reshape(new TensorShape(finalDim));
        _finalProjW = LoadFusedWeight(w, $"{_prefix}.block.{_nStages + 2}");
        _finalProjB = WhisperOps.EnsureF32(w[$"{_prefix}.block.{_nStages + 2}.bias"]);
    }

    /// <summary>Forward — <paramref name="pcm"/> channels-first <c>[B, 1, T_pcm]</c> →
    /// <c>[B, latent_dim, T_pcm/hop]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor pcm, int batch, int tPcm)
    {
        if (_stemW is null) throw new InvalidOperationException("DacEncoder weights not loaded.");

        // Stem: WNConv1d(1 → encoder_dim, k=7, padding=3).
        int stemPad = _cfg.StemKernelSize / 2;
        int tStem = tPcm + 2 * stemPad - (_cfg.StemKernelSize - 1);
        Tensor x = new(new TensorShape(batch, _cfg.EncoderDim, tStem), DType.F32);
        backend.Conv1d(x, pcm, _stemW!, _stemB,
            stride: 1, padLeft: stemPad, padRight: stemPad, dilation: 1, groups: 1);

        int t = tStem;
        int dim = _cfg.EncoderDim;

        // Encoder stages.
        for (int i = 0; i < _nStages; i++)
        {
            int innerDim = dim;     // current channel count before the downsample step
            // Residual units (3 of them, dilations [1, 3, 9]).
            foreach (DacResidualUnit unit in _stageUnits[i])
            {
                Tensor next = unit.Forward(backend, x, batch, t);
                x.Dispose();
                x = next;
                t = (int)x.Shape[2];
            }

            // Snake activation.
            Tensor snk = new(x.Shape, DType.F32);
            backend.Snake(snk, x, _downsampleSnakeAlpha[i]!, null);
            x.Dispose();
            x = snk;

            // Downsample WNConv1d(d → 2d, k=2*stride, stride=stride, padding=ceil(stride/2)).
            int stride = _strides[i];
            int kDown = 2 * stride;
            int padDown = (stride + 1) / 2;     // ceil(stride/2)
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

        // Final snake + projection.
        Tensor preProj = new(x.Shape, DType.F32);
        backend.Snake(preProj, x, _finalSnakeAlpha!, null);
        x.Dispose();

        int finalPad = _cfg.EncoderProjKernelSize / 2;
        int tFinal = t + 2 * finalPad - (_cfg.EncoderProjKernelSize - 1);
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
            foreach (DacResidualUnit unit in _stageUnits[i])
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

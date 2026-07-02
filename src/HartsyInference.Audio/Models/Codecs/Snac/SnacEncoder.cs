using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Snac;

/// <summary>SNAC encoder, matching the official hubertsiuzdak/snac <c>Encoder</c> (snac/layers.py):
/// <code>
///   block.0          : WNConv1d(1 -> encoder_dim, k7, pad3)
///   block.{i+1}      : EncoderBlock(stride) = 3 x ResidualUnit(groups) + Snake + WNConv1d down
///   [LocalMHA]       : only when attn_window_size != null (Phase 4, not wired)
///   block.{N+1}      : WNConv1d(d_model -> d_model, k7, pad3, groups = d_model if depthwise)
/// </code>
/// There is NO Snake before the final conv (unlike DAC). With <c>depthwise=true</c> the residual-unit first
/// conv uses groups = block input dim, and the final conv uses groups = d_model.
///
/// PARITY-TODO: only used for the encode (audio -> codes) path; verify against real snac_24khz weights
/// (Orpheus uses decode only, and <see cref="Snac"/> loads this tolerantly).</summary>
internal sealed unsafe class SnacEncoder
{
    private readonly SnacConfig _cfg;
    private readonly string _prefix;
    private readonly int _nStages;
    private readonly int[] _strides;
    private readonly bool _depthwise;

    private Tensor? _stemW;
    private Tensor? _stemB;
    private readonly SnacResidualUnit[][] _stageUnits;
    private readonly Tensor?[] _downsampleSnakeAlpha;
    private readonly Tensor?[] _downsampleW;
    private readonly Tensor?[] _downsampleB;
    private Tensor? _finalProjW;
    private Tensor? _finalProjB;

    public SnacEncoder(SnacConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;
        _nStages = cfg.EncoderRates.Count;
        _strides = [.. cfg.EncoderRates];
        _depthwise = cfg.Depthwise;

        // LocalMHA (attn_window_size, 32/44 kHz) is not wired yet — deferred to Forward so config/construction
        // still work (the encode path throws only when actually invoked). Phase 4 PARITY-TODO.
        _stageUnits = new SnacResidualUnit[_nStages][];
        _downsampleSnakeAlpha = new Tensor?[_nStages];
        _downsampleW = new Tensor?[_nStages];
        _downsampleB = new Tensor?[_nStages];

        int dim = cfg.EncoderDim;
        for (int i = 0; i < _nStages; i++)
        {
            dim *= 2;
            int innerDim = dim / 2;
            int groups = _depthwise ? innerDim : 1;
            _stageUnits[i] = new SnacResidualUnit[cfg.ResidualDilations.Count];
            for (int j = 0; j < cfg.ResidualDilations.Count; j++)
            {
                _stageUnits[i][j] = new SnacResidualUnit(
                    prefix: $"{prefix}.block.{i + 1}.block.{j}",
                    dim: innerDim,
                    kernel: cfg.ResidualKernelSize,
                    dilation: cfg.ResidualDilations[j],
                    groups: groups);
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

        // Final conv sits directly after the encoder blocks (no Snake). attn would shift this index by 1.
        int finalIdx = _nStages + 1;
        _finalProjW = LoadFusedWeight(w, $"{_prefix}.block.{finalIdx}");
        _finalProjB = WhisperOps.EnsureF32(w[$"{_prefix}.block.{finalIdx}.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor pcm, int batch, int tPcm)
    {
        if (_cfg.AttnWindowSize is not null)
            throw new NotSupportedException("SNAC LocalMHA (attn_window_size) encode is not wired yet (Phase 4 PARITY-TODO).");
        if (_stemW is null) throw new InvalidOperationException("SnacEncoder weights not loaded.");

        int stemPad = _cfg.StemKernelSize / 2;
        int tStem = tPcm;
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

        // Final conv (no preceding Snake). groups = d_model when depthwise.
        int finalPad = _cfg.StemKernelSize / 2;
        int tFinal = t;
        int finalGroups = _depthwise ? dim : 1;
        Tensor latent = new(new TensorShape(batch, _cfg.LatentDim, tFinal), DType.F32);
        backend.Conv1d(latent, x, _finalProjW!, _finalProjB,
            stride: 1, padLeft: finalPad, padRight: finalPad, dilation: 1, groups: finalGroups);
        x.Dispose();
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

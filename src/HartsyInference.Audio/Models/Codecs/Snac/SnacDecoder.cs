using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Snac;

/// <summary>SNAC decoder, matching the official hubertsiuzdak/snac <c>Decoder</c> (snac/layers.py).</summary>
/// <remarks>
/// <para>The published checkpoints set <c>depthwise=true</c> and <c>noise=true</c>, which changes the
/// module layout vs a plain DAC decoder:</para>
/// <list type="bullet">
///   <item>Depthwise initial: <c>model.0</c> is a depthwise k7 conv (groups = input channels) and
///         <c>model.1</c> is a pointwise k1 conv to <c>DecoderDim</c>. The DecoderBlocks then start at
///         <c>model.2</c> (a plain decoder starts the blocks at <c>model.1</c>).</item>
///   <item>NoiseBlock: each DecoderBlock inserts a <c>NoiseBlock</c> at <c>block.2</c> (a bias-free k1 conv
///         whose output scales per-frame Gaussian noise that is added back), shifting the three residual
///         units to <c>block.3 / 4 / 5</c>.</item>
///   <item>Residual-unit first conv uses <c>groups = output_dim</c> (depthwise).</item>
///   <item>The decoder ends in <c>Tanh</c> (reconstruction is in [-1, 1]).</item>
/// </list>
///
/// PARITY-TODO: verify against real snac_24khz weights. The NoiseBlock is stochastic in the reference
/// (torch.randn); here it uses a fixed-seed Gaussian so decode is deterministic. The 32/44 kHz checkpoints
/// add a LocalMHA (attn_window_size 32) which is not yet wired (throws if requested).</remarks>
internal sealed unsafe class SnacDecoder
{
    private readonly SnacConfig _cfg;
    private readonly string _prefix;
    private readonly int _nStages;
    private readonly int[] _strides;
    private readonly bool _depthwise;
    private readonly bool _noise;
    private readonly int _blockBase;     // model index where the first DecoderBlock sits

    // Initial: depthwise (two convs) or plain (one conv).
    private Tensor? _initDwW, _initDwB;   // model.0 depthwise (depthwise mode only)
    private Tensor? _initW, _initB;       // model.0 plain conv, or model.1 pointwise in depthwise mode

    private readonly Tensor?[] _stageSnakeAlpha;
    private readonly Tensor?[] _stageUpW;
    private readonly Tensor?[] _stageUpB;
    private readonly Tensor?[] _noiseW;   // NoiseBlock .linear weight (bias-free), per stage
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
        _depthwise = cfg.Depthwise;
        _noise = cfg.Noise;

        // LocalMHA (attn_window_size, 32/44 kHz) is not wired yet — deferred to Forward so config/construction still
        // work (the decode path throws only when actually invoked). Phase 4 PARITY-TODO.

        // model.0 (depthwise) + model.1 (pointwise) push the blocks to model.2; plain decoder starts at model.1.
        _blockBase = _depthwise ? 2 : 1;

        _stageSnakeAlpha = new Tensor?[_nStages];
        _stageUpW = new Tensor?[_nStages];
        _stageUpB = new Tensor?[_nStages];
        _noiseW = new Tensor?[_nStages];
        _stageUnits = new SnacResidualUnit[_nStages][];

        int resBase = _noise ? 3 : 2;     // residual units start after Snake(0), Transpose(1), [Noise(2)]
        int dim = cfg.DecoderDim;
        for (int i = 0; i < _nStages; i++)
        {
            int outDim = dim / 2;
            int groups = _depthwise ? outDim : 1;
            int blockIdx = _blockBase + i;
            _stageUnits[i] = new SnacResidualUnit[cfg.ResidualDilations.Count];
            for (int j = 0; j < cfg.ResidualDilations.Count; j++)
            {
                _stageUnits[i][j] = new SnacResidualUnit(
                    prefix: $"{prefix}.model.{blockIdx}.block.{resBase + j}",
                    dim: outDim,
                    kernel: cfg.ResidualKernelSize,
                    dilation: cfg.ResidualDilations[j],
                    groups: groups);
            }
            dim = outDim;
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        if (_depthwise)
        {
            _initDwW = LoadFusedWeight(w, $"{_prefix}.model.0");
            _initDwB = WhisperOps.EnsureF32(w[$"{_prefix}.model.0.bias"]);
            _initW = LoadFusedWeight(w, $"{_prefix}.model.1");
            _initB = WhisperOps.EnsureF32(w[$"{_prefix}.model.1.bias"]);
        }
        else
        {
            _initW = LoadFusedWeight(w, $"{_prefix}.model.0");
            _initB = WhisperOps.EnsureF32(w[$"{_prefix}.model.0.bias"]);
        }

        int dim = _cfg.DecoderDim;
        for (int i = 0; i < _nStages; i++)
        {
            int blockIdx = _blockBase + i;
            _stageSnakeAlpha[i] = WhisperOps.EnsureF32(w[$"{_prefix}.model.{blockIdx}.block.0.alpha"]).Reshape(new TensorShape(dim));
            _stageUpW[i] = LoadFusedTransposeWeight(w, $"{_prefix}.model.{blockIdx}.block.1");
            _stageUpB[i] = WhisperOps.EnsureF32(w[$"{_prefix}.model.{blockIdx}.block.1.bias"]);
            if (_noise)
                _noiseW[i] = LoadFusedWeight(w, $"{_prefix}.model.{blockIdx}.block.2.linear");   // bias-free k1 conv

            for (int j = 0; j < _cfg.ResidualDilations.Count; j++)
                _stageUnits[i][j].LoadWeights(w);

            dim /= 2;
        }

        int finalSnakeIdx = _blockBase + _nStages;
        _finalSnakeAlpha = WhisperOps.EnsureF32(w[$"{_prefix}.model.{finalSnakeIdx}.alpha"]).Reshape(new TensorShape(dim));
        _finalConvW = LoadFusedWeight(w, $"{_prefix}.model.{finalSnakeIdx + 1}");
        _finalConvB = WhisperOps.EnsureF32(w[$"{_prefix}.model.{finalSnakeIdx + 1}.bias"]);
    }

    /// <param name="callSeed">Distinguishes separate <see cref="Forward"/> calls that decode overlapping windows of the same utterance so each window's <see cref="ApplyNoiseBlock"/> draws different noise; the default (<c>callSeed=0</c>) reduces to the same fixed-seed sequence this method always used, so non-streaming callers see byte-identical output.</param>
    /// <remarks>Used by Orpheus's streaming path (see <see cref="Pipelines.OrpheusPipeline"/>). Without per-call
    /// seed distinction, every window would replay the identical Gaussian sequence at the same relative offset,
    /// producing an audible periodic artifact that never shows up in a single whole-utterance decode.</remarks>
    public Tensor Forward(IBackend backend, Tensor latent, int batch, int tFrames, int callSeed = 0)
    {
        if (_cfg.AttnWindowSize is not null)
            throw new NotSupportedException("SNAC LocalMHA (attn_window_size) decode is not wired yet (Phase 4 PARITY-TODO).");
        if (_initW is null) throw new InvalidOperationException("SnacDecoder weights not loaded.");

        int initPad = _cfg.StemKernelSize / 2;
        int tInit = tFrames;     // k7 same padding keeps T
        Tensor x;
        if (_depthwise)
        {
            // Depthwise k7 (groups = latent channels), then pointwise k1 to DecoderDim.
            int latentCh = (int)latent.Shape[1];
            Tensor dw = new(new TensorShape(batch, latentCh, tInit), DType.F32);
            backend.Conv1d(dw, latent, _initDwW!, _initDwB,
                stride: 1, padLeft: initPad, padRight: initPad, dilation: 1, groups: latentCh);
            x = new Tensor(new TensorShape(batch, _cfg.DecoderDim, tInit), DType.F32);
            backend.Conv1d(x, dw, _initW!, _initB, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
            dw.Dispose();
        }
        else
        {
            x = new Tensor(new TensorShape(batch, _cfg.DecoderDim, tInit), DType.F32);
            backend.Conv1d(x, latent, _initW!, _initB,
                stride: 1, padLeft: initPad, padRight: initPad, dilation: 1, groups: 1);
        }

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
                stride: stride, padLeft: padLeft, padRight: padRight, dilation: 1, groups: 1);
            x.Dispose();
            x = up;
            t = tUp;
            dim = dimOut;

            if (_noise && _cfg.NoiseScale != 0f) ApplyNoiseBlock(backend, x, _noiseW[i]!, batch, dim, t, stageSeed: i, callSeed);

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

        // Decoder ends in Tanh (reference). PARITY-TODO: confirm downstream callers do not also clip.
        float* pp = (float*)pcm.DataPointer;
        long n = pcm.Shape.ElementCount;
        for (long k = 0; k < n; k++) pp[k] = MathF.Tanh(pp[k]);
        return pcm;
    }

    /// <summary>NoiseBlock: h = conv1x1(x); x += randn[B,1,T] * h (noise broadcast across channels). Reference uses torch.randn; we use a fixed-seed Gaussian so decode is reproducible (PARITY-TODO).</summary>
    private void ApplyNoiseBlock(IBackend backend, Tensor x, Tensor linearW, int batch, int dim, int t, int stageSeed, int callSeed)
    {
        Tensor h = new(new TensorShape(batch, dim, t), DType.F32);
        backend.Conv1d(h, x, linearW, null, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);

        Random rng = new(1234 + stageSeed + callSeed * 97);
        float scale = _cfg.NoiseScale;
        float* xp = (float*)x.DataPointer;
        float* hp = (float*)h.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int j = 0; j < t; j++)
            {
                float g = NextGaussian(rng) * scale;
                for (int c = 0; c < dim; c++)
                {
                    long idx = ((long)b * dim + c) * t + j;
                    xp[idx] += g * hp[idx];
                }
            }
        h.Dispose();
    }

    private static float NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_initDwW is not null) yield return _initDwW;
        if (_initDwB is not null) yield return _initDwB;
        if (_initW is not null) yield return _initW;
        if (_initB is not null) yield return _initB;
        for (int i = 0; i < _nStages; i++)
        {
            if (_stageSnakeAlpha[i] is not null) yield return _stageSnakeAlpha[i]!;
            if (_stageUpW[i] is not null) yield return _stageUpW[i]!;
            if (_stageUpB[i] is not null) yield return _stageUpB[i]!;
            if (_noiseW[i] is not null) yield return _noiseW[i]!;
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

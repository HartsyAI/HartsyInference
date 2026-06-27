using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Dac;

/// <summary>DAC decoder. Mirrors the descript-audio-codec decoder:
/// <code>
///   initial: WNConv1d(latent_dim → decoder_dim, k=7, padding=3)
///   for stride in decoder_rates:
///     DecoderBlock(d, d/2, stride)
///       = Snake1d(d) + WNConvTranspose1d(d → d/2, k=2*stride, stride=stride,
///                                         padding=ceil(stride/2),
///                                         output_padding=stride%2)
///         + 3 × ResidualUnit(d/2, dilation=[1,3,9])
///     d /= 2
///   Snake1d(d_min)
///   WNConv1d(d_min → channels=1, k=7, padding=3)
///   tanh
/// </code>
///
/// <para>Note <c>output_padding=stride%2</c> in the transpose conv: for odd-stride
/// stages (e.g. stride=5 in the 24 kHz / 16 kHz variants) PyTorch adds one extra zero
/// on the right of the raw conv output before trimming. We absorb that by passing
/// <c>padRight = ceil(stride/2) - (stride % 2)</c> to
/// <see cref="IBackend.ConvTranspose1d"/> — algebraic equivalent that fits our
/// <c>padLeft</c>/<c>padRight</c> signature without needing an explicit
/// <c>output_padding</c> argument.</para>
///
/// <para>Final <c>tanh</c> keeps reconstructed PCM safely inside <c>[-1, 1]</c>.</para></summary>
internal sealed class DacDecoder
{
    private readonly DacConfig _cfg;
    private readonly string _prefix;
    private readonly int _nStages;
    private readonly int[] _strides;

    private Tensor? _initialW;
    private Tensor? _initialB;
    private readonly Tensor?[] _stageSnakeAlpha;
    private readonly Tensor?[] _stageUpW;       // ConvTranspose1d weight [C_in, C_out, K]
    private readonly Tensor?[] _stageUpB;
    private readonly DacResidualUnit[][] _stageUnits;
    private Tensor? _finalSnakeAlpha;
    private Tensor? _finalConvW;
    private Tensor? _finalConvB;

    public DacDecoder(DacConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;
        _nStages = cfg.DecoderRates.Count;
        _strides = [.. cfg.DecoderRates];

        _stageSnakeAlpha = new Tensor?[_nStages];
        _stageUpW = new Tensor?[_nStages];
        _stageUpB = new Tensor?[_nStages];
        _stageUnits = new DacResidualUnit[_nStages][];

        int dim = cfg.DecoderDim;
        for (int i = 0; i < _nStages; i++)
        {
            int outDim = dim / 2;
            _stageUnits[i] = new DacResidualUnit[cfg.ResidualDilations.Count];
            for (int j = 0; j < cfg.ResidualDilations.Count; j++)
            {
                _stageUnits[i][j] = new DacResidualUnit(
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
            // DecoderBlock sequential: [Snake1d, WNConvTranspose1d, 3 × ResidualUnit].
            _stageSnakeAlpha[i] = WhisperOps.EnsureF32(w[$"{_prefix}.model.{i + 1}.block.0.alpha"]).Reshape(new TensorShape(dim));
            _stageUpW[i] = LoadFusedTransposeWeight(w, $"{_prefix}.model.{i + 1}.block.1");
            _stageUpB[i] = WhisperOps.EnsureF32(w[$"{_prefix}.model.{i + 1}.block.1.bias"]);

            for (int j = 0; j < _cfg.ResidualDilations.Count; j++)
                _stageUnits[i][j].LoadWeights(w);

            dim /= 2;
        }

        // Final snake + conv at decoder.model.{n_stages+1}.alpha and .{n_stages+2}.weight.
        _finalSnakeAlpha = WhisperOps.EnsureF32(w[$"{_prefix}.model.{_nStages + 1}.alpha"]).Reshape(new TensorShape(dim));
        _finalConvW = LoadFusedWeight(w, $"{_prefix}.model.{_nStages + 2}");
        _finalConvB = WhisperOps.EnsureF32(w[$"{_prefix}.model.{_nStages + 2}.bias"]);
    }

    /// <summary>Forward — <paramref name="latent"/> channels-first
    /// <c>[B, latent_dim, T_frames]</c> → <c>[B, 1, T_frames * hop]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, int batch, int tFrames)
    {
        if (_initialW is null) throw new InvalidOperationException("DacDecoder weights not loaded.");

        // Initial WNConv1d(latent_dim → decoder_dim, k=7, padding=3).
        int initPad = _cfg.StemKernelSize / 2;
        int tInit = tFrames + 2 * initPad - (_cfg.StemKernelSize - 1);
        Tensor x = new(new TensorShape(batch, _cfg.DecoderDim, tInit), DType.F32);
        backend.Conv1d(x, latent, _initialW!, _initialB,
            stride: 1, padLeft: initPad, padRight: initPad, dilation: 1, groups: 1);

        int t = tInit;
        int dim = _cfg.DecoderDim;

        // Decoder stages.
        for (int i = 0; i < _nStages; i++)
        {
            // Snake.
            Tensor snk = new(x.Shape, DType.F32);
            backend.Snake(snk, x, _stageSnakeAlpha[i]!, null);
            x.Dispose();
            x = snk;

            // WNConvTranspose1d(dim → dim/2). Either k=2*stride with descript's output_padding=s%2, or
            // explicit kernel sizes with padding=(k-stride)/2 and no output padding (Spark BiCodec).
            int stride = _strides[i];
            int padLeft, padRight, kUp;
            if (_cfg.DecoderKernelSizes is not null)
            {
                kUp = _cfg.DecoderKernelSizes[i];
                int padding = (kUp - stride) / 2;
                padLeft = padding;
                padRight = padding;
            }
            else
            {
                kUp = 2 * stride;
                int padding = (stride + 1) / 2;        // ceil(stride/2)
                int outputPadding = stride % 2;
                // PyTorch length = (T_in-1)*stride + K - 2*padding + output_padding; our backend uses
                // (T_in-1)*stride + K - padLeft - padRight, so padLeft + padRight = 2*padding - output_padding.
                padLeft = padding;
                padRight = padding - outputPadding;
            }
            int dimOut = dim / 2;
            int tUp = (t - 1) * stride + kUp - padLeft - padRight;
            Tensor up = new(new TensorShape(batch, dimOut, tUp), DType.F32);
            backend.ConvTranspose1d(up, x, _stageUpW[i]!, _stageUpB[i],
                stride: stride, padLeft: padLeft, padRight: padRight, dilation: 1, groups: 1);
            x.Dispose();
            x = up;
            t = tUp;
            dim = dimOut;

            // 3 × ResidualUnit.
            foreach (DacResidualUnit unit in _stageUnits[i])
            {
                Tensor next = unit.Forward(backend, x, batch, t);
                x.Dispose();
                x = next;
                t = (int)x.Shape[2];
            }
        }

        // Final snake + WNConv1d(d_min → 1, k=7, padding=3) + tanh.
        Tensor activated = new(x.Shape, DType.F32);
        backend.Snake(activated, x, _finalSnakeAlpha!, null);
        x.Dispose();

        int finalPad = _cfg.DecoderFinalKernelSize / 2;
        int tOut = t + 2 * finalPad - (_cfg.DecoderFinalKernelSize - 1);
        Tensor pcmRaw = new(new TensorShape(batch, _cfg.Channels, tOut), DType.F32);
        backend.Conv1d(pcmRaw, activated, _finalConvW!, _finalConvB,
            stride: 1, padLeft: finalPad, padRight: finalPad, dilation: 1, groups: 1);
        activated.Dispose();

        Tensor pcm = new(pcmRaw.Shape, DType.F32);
        backend.Tanh(pcm, pcmRaw);
        pcmRaw.Dispose();
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
            foreach (DacResidualUnit unit in _stageUnits[i])
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

    private Tensor LoadFusedTransposeWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor g = WhisperOps.EnsureF32(w[$"{prefix}.weight_g"]);
        Tensor v = WhisperOps.EnsureF32(w[$"{prefix}.weight_v"]);
        // descript/Spark norm over dim 0 (C_in) — weight_g is [C_in,1,1], same composition as a regular conv.
        return _cfg.TransposeWeightNormDim0 ? WeightNormFusion.Fuse(g, v) : WeightNormFusionT.Fuse(g, v);
    }
}

using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Oobleck;

/// <summary>Oobleck decoder (diffusers <c>OobleckDecoder</c>):
/// <code>
///   conv1: WNConv1d(latent_dim → channels·mult[-1], k=7, padding=3)
///   for stride in reversed(downsampling_ratios):
///     DecoderBlock = SnakeBeta(in) + WNConvTranspose1d(in → out, k=2·stride, stride, padding=ceil(s/2))
///                    + 3 × ResidualUnit(out, dilation=[1, 3, 9])
///   SnakeBeta(channels) + WNConv1d(channels → audio_channels, k=7, padding=3, bias=False)
/// </code>
/// All strides published so far are even, so each transpose conv exactly multiplies T by its stride
/// (T_out = T·s with padLeft = padRight = s/2) and the full stack expands T by the hop length.
/// No final tanh — Stable-Audio-family decoders emit unbounded PCM that callers clamp.</summary>
internal sealed class OobleckDecoder
{
    private readonly OobleckConfig _cfg;
    private readonly string _prefix;
    private readonly int[] _strides;
    private readonly int[] _dims;

    private Tensor? _stemW, _stemB;
    private Tensor?[] _blockSnakeAlpha = [];
    private Tensor?[] _blockSnakeBeta = [];
    private Tensor?[] _blockUpW = [];
    private Tensor?[] _blockUpB = [];
    private OobleckResidualUnit[][] _blockUnits = [];
    private Tensor? _finalSnakeAlpha, _finalSnakeBeta;
    private Tensor? _outW;

    public OobleckDecoder(OobleckConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;

        // strides = reversed downsampling ratios; dims walk channel multiples high → low:
        // dims[i] = channels · ([1] + multiples)[n - i], so dims = [ch·m[-1], …, ch·1, ch·1].
        int n = cfg.DownsamplingRatios.Length;
        _strides = new int[n];
        for (int i = 0; i < n; i++) _strides[i] = cfg.DownsamplingRatios[n - 1 - i];

        int[] mults = new int[cfg.ChannelMultiples.Length + 1];
        mults[0] = 1;
        cfg.ChannelMultiples.CopyTo(mults, 1);
        _dims = new int[n + 1];
        for (int i = 0; i <= n; i++) _dims[i] = cfg.DecoderChannels * mults[n - i];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // Real descript / ACE-Step 1.5 layout is a flat nn.Sequential:
        //   layers.0            = stem WNConv1d  (latent -> dims[0], k=7)
        //   layers.{1..n}       = DecoderBlock i  (Sequential: [Snake, WNConvTranspose1d, ResUnit×3])
        //   layers.{n+1}        = final Snake
        //   layers.{n+2}        = output WNConv1d (-> audio_channels, k=7, bias=False)
        // Inside a DecoderBlock: .layers.0 Snake, .layers.1 ConvTranspose, .layers.{2,3,4} ResUnits.
        _stemW = OobleckOps.LoadFusedWeight(w, $"{_prefix}.layers.0");
        _stemB = WhisperOps.EnsureF32(w[$"{_prefix}.layers.0.bias"]);

        int n = _strides.Length;
        _blockSnakeAlpha = new Tensor?[n];
        _blockSnakeBeta = new Tensor?[n];
        _blockUpW = new Tensor?[n];
        _blockUpB = new Tensor?[n];
        _blockUnits = new OobleckResidualUnit[n][];
        for (int i = 0; i < n; i++)
        {
            string blk = $"{_prefix}.layers.{i + 1}";
            (_blockSnakeAlpha[i], _blockSnakeBeta[i]) = OobleckOps.LoadSnake(w, $"{blk}.layers.0", _dims[i]);
            _blockUpW[i] = OobleckOps.LoadFusedTransposeWeight(w, $"{blk}.layers.1");
            _blockUpB[i] = WhisperOps.EnsureF32(w[$"{blk}.layers.1.bias"]);
            _blockUnits[i] = new OobleckResidualUnit[3];
            for (int j = 0; j < 3; j++)
            {
                _blockUnits[i][j] = new OobleckResidualUnit($"{blk}.layers.{j + 2}", _dims[i + 1], dilation: j == 0 ? 1 : j == 1 ? 3 : 9);
                _blockUnits[i][j].LoadWeights(w);
            }
        }

        (_finalSnakeAlpha, _finalSnakeBeta) = OobleckOps.LoadSnake(w, $"{_prefix}.layers.{n + 1}", _dims[^1]);
        _outW = OobleckOps.LoadFusedWeight(w, $"{_prefix}.layers.{n + 2}");   // bias=False upstream
    }

    /// <summary>Forward — latent <c>[B, latent_dim, T]</c> → PCM <c>[B, audio_channels, T · hop]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, int batch, int tFrames)
    {
        if (_stemW is null) throw new InvalidOperationException("OobleckDecoder weights not loaded.");

        Tensor x = new(new TensorShape(batch, _dims[0], tFrames), DType.F32);
        backend.Conv1d(x, latent, _stemW!, _stemB,
            stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);

        int t = tFrames;
        for (int i = 0; i < _strides.Length; i++)
        {
            Tensor snk = new(x.Shape, DType.F32);
            backend.Snake(snk, x, _blockSnakeAlpha[i]!, _blockSnakeBeta[i]);
            x.Dispose();

            // WNConvTranspose1d(k=2s, stride=s, padding=ceil(s/2), output_padding=0):
            // padLeft + padRight = 2·ceil(s/2); even s → exact T·s expansion.
            int stride = _strides[i];
            int pad = (stride + 1) / 2;
            int tUp = (t - 1) * stride + 2 * stride - 2 * pad;
            Tensor up = new(new TensorShape(batch, _dims[i + 1], tUp), DType.F32);
            backend.ConvTranspose1d(up, snk, _blockUpW[i]!, _blockUpB[i],
                stride: stride, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
            snk.Dispose();
            x = up;
            t = tUp;

            foreach (OobleckResidualUnit unit in _blockUnits[i])
            {
                Tensor next = unit.Forward(backend, x, batch, t);
                x.Dispose();
                x = next;
            }
        }

        Tensor act = new(x.Shape, DType.F32);
        backend.Snake(act, x, _finalSnakeAlpha!, _finalSnakeBeta);
        x.Dispose();

        Tensor pcm = new(new TensorShape(batch, _cfg.AudioChannels, t), DType.F32);
        backend.Conv1d(pcm, act, _outW!, null,
            stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);
        act.Dispose();
        return pcm;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] singles = [_stemW, _stemB, _finalSnakeAlpha, _finalSnakeBeta, _outW];
        foreach (Tensor? s in singles) if (s is not null) yield return s;
        for (int i = 0; i < _blockUnits.Length; i++)
        {
            Tensor?[] blk = [_blockSnakeAlpha[i], _blockSnakeBeta[i], _blockUpW[i], _blockUpB[i]];
            foreach (Tensor? s in blk) if (s is not null) yield return s;
            foreach (OobleckResidualUnit unit in _blockUnits[i])
                foreach (Tensor wt in unit.EnumerateWeights()) yield return wt;
        }
    }
}

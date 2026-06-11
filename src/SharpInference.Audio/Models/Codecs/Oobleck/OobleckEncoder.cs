using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Codecs.Oobleck;

/// <summary>Oobleck encoder (diffusers <c>OobleckEncoder</c>):
/// <code>
///   conv1: WNConv1d(audio_channels → hidden, k=7, padding=3)
///   for stride in downsampling_ratios:
///     EncoderBlock = 3 × ResidualUnit(in, dilation=[1, 3, 9]) + SnakeBeta(in)
///                    + WNConv1d(in → out, k=2·stride, stride, padding=ceil(s/2))
///   SnakeBeta(hidden·mult[-1]) + WNConv1d(hidden·mult[-1] → hidden, k=3, padding=1)
/// </code>
/// The output's <c>hidden</c> channels are the Gaussian parameters — mean in the first half,
/// softplus-scale in the second. <see cref="OobleckVae.EncodeMode"/> returns the mean
/// (diffusers' <c>latent_dist.mode()</c> / "argmax" sampling).</summary>
internal sealed class OobleckEncoder
{
    private readonly OobleckConfig _cfg;
    private readonly string _prefix;
    private readonly int[] _dims;

    private Tensor? _stemW, _stemB;
    private OobleckResidualUnit[][] _blockUnits = [];
    private Tensor?[] _blockSnakeAlpha = [];
    private Tensor?[] _blockSnakeBeta = [];
    private Tensor?[] _blockDownW = [];
    private Tensor?[] _blockDownB = [];
    private Tensor? _finalSnakeAlpha, _finalSnakeBeta;
    private Tensor? _outW, _outB;

    public OobleckEncoder(OobleckConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;

        int n = cfg.DownsamplingRatios.Length;
        int[] mults = new int[cfg.ChannelMultiples.Length + 1];
        mults[0] = 1;
        cfg.ChannelMultiples.CopyTo(mults, 1);
        _dims = new int[n + 1];
        for (int i = 0; i <= n; i++) _dims[i] = cfg.EncoderHiddenSize * mults[i];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _stemW = OobleckOps.LoadFusedWeight(w, $"{_prefix}.conv1");
        _stemB = WhisperOps.EnsureF32(w[$"{_prefix}.conv1.bias"]);

        int n = _cfg.DownsamplingRatios.Length;
        _blockUnits = new OobleckResidualUnit[n][];
        _blockSnakeAlpha = new Tensor?[n];
        _blockSnakeBeta = new Tensor?[n];
        _blockDownW = new Tensor?[n];
        _blockDownB = new Tensor?[n];
        for (int i = 0; i < n; i++)
        {
            string blk = $"{_prefix}.block.{i}";
            _blockUnits[i] = new OobleckResidualUnit[3];
            for (int j = 0; j < 3; j++)
            {
                _blockUnits[i][j] = new OobleckResidualUnit($"{blk}.res_unit{j + 1}", _dims[i], dilation: j == 0 ? 1 : j == 1 ? 3 : 9);
                _blockUnits[i][j].LoadWeights(w);
            }
            (_blockSnakeAlpha[i], _blockSnakeBeta[i]) = OobleckOps.LoadSnake(w, $"{blk}.snake1", _dims[i]);
            _blockDownW[i] = OobleckOps.LoadFusedWeight(w, $"{blk}.conv1");
            _blockDownB[i] = WhisperOps.EnsureF32(w[$"{blk}.conv1.bias"]);
        }

        (_finalSnakeAlpha, _finalSnakeBeta) = OobleckOps.LoadSnake(w, $"{_prefix}.snake1", _dims[^1]);
        _outW = OobleckOps.LoadFusedWeight(w, $"{_prefix}.conv2");
        _outB = WhisperOps.EnsureF32(w[$"{_prefix}.conv2.bias"]);
    }

    /// <summary>Forward — PCM <c>[B, audio_channels, T]</c> (T a multiple of the hop) →
    /// Gaussian parameters <c>[B, hidden, T / hop]</c> (mean ‖ scale).</summary>
    public Tensor Forward(IBackend backend, Tensor pcm, int batch, int tPcm)
    {
        if (_stemW is null) throw new InvalidOperationException("OobleckEncoder weights not loaded.");

        Tensor x = new(new TensorShape(batch, _dims[0], tPcm), DType.F32);
        backend.Conv1d(x, pcm, _stemW!, _stemB,
            stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);

        int t = tPcm;
        for (int i = 0; i < _cfg.DownsamplingRatios.Length; i++)
        {
            foreach (OobleckResidualUnit unit in _blockUnits[i])
            {
                Tensor next = unit.Forward(backend, x, batch, t);
                x.Dispose();
                x = next;
            }

            Tensor snk = new(x.Shape, DType.F32);
            backend.Snake(snk, x, _blockSnakeAlpha[i]!, _blockSnakeBeta[i]);
            x.Dispose();

            // WNConv1d(k=2s, stride=s, padding=ceil(s/2)); even s + T multiple of s → exact T/s.
            int stride = _cfg.DownsamplingRatios[i];
            int pad = (stride + 1) / 2;
            int tDown = (t + 2 * pad - 2 * stride) / stride + 1;
            Tensor down = new(new TensorShape(batch, _dims[i + 1], tDown), DType.F32);
            backend.Conv1d(down, snk, _blockDownW[i]!, _blockDownB[i],
                stride: stride, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
            snk.Dispose();
            x = down;
            t = tDown;
        }

        Tensor act = new(x.Shape, DType.F32);
        backend.Snake(act, x, _finalSnakeAlpha!, _finalSnakeBeta);
        x.Dispose();

        Tensor parameters = new(new TensorShape(batch, _cfg.EncoderHiddenSize, t), DType.F32);
        backend.Conv1d(parameters, act, _outW!, _outB,
            stride: 1, padLeft: 1, padRight: 1, dilation: 1, groups: 1);
        act.Dispose();
        return parameters;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] singles = [_stemW, _stemB, _finalSnakeAlpha, _finalSnakeBeta, _outW, _outB];
        foreach (Tensor? s in singles) if (s is not null) yield return s;
        for (int i = 0; i < _blockUnits.Length; i++)
        {
            foreach (OobleckResidualUnit unit in _blockUnits[i])
                foreach (Tensor wt in unit.EnumerateWeights()) yield return wt;
            Tensor?[] blk = [_blockSnakeAlpha[i], _blockSnakeBeta[i], _blockDownW[i], _blockDownB[i]];
            foreach (Tensor? s in blk) if (s is not null) yield return s;
        }
    }
}

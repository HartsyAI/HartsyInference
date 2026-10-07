using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.IndexTts;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts2;

/// <summary>IndexTTS-2's vocoder: the stock <c>nvidia/bigvgan_v2_22khz_80band_256x</c> generator, used
/// unmodified — a plain mel→waveform BigVGAN-v2 with NO speaker conditioning at all (unlike IndexTTS-1.5's
/// own, which bakes in a d-vector injected at every stage — see
/// <see cref="HartsyInference.Audio.Models.IndexTts.IndexTtsBigVganGenerator"/>'s remarks for that
/// correction). Reuses <see cref="AntiAliasedSnake"/> and
/// <see cref="HartsyInference.Audio.Models.IndexTts.IndexTtsBigVganResBlock"/> verbatim (both fully generic,
/// channel-count-parameterized, confirmed from the real checkpoint's own key names —
/// <c>resblocks.N.activations.N.act.alpha</c>/<c>convs1.N</c>/<c>convs2.N</c>, byte-identical naming to
/// IndexTTS-1.5's BigVGAN since both are the same upstream AMPBlock1 architecture family). The real
/// checkpoint has NO <c>cond_layer</c>/<c>conds.N</c> keys and NO <c>conv_post.bias</c> key (confirmed —
/// <c>use_bias_at_final: false</c> in the real <c>config.json</c>), and does not tanh its output
/// (<c>use_tanh_at_final: false</c>).</summary>
internal sealed unsafe class IndexTts2BigVganGenerator : IDisposable
{
    private readonly IndexTts2BigVganConfig _cfg;
    private readonly List<Tensor> _owned = [];
    private readonly int _numStages;
    private readonly Tensor?[] _upsW, _upsB;
    private readonly IndexTtsBigVganResBlock[][] _resblocks;   // [stage][kernelIdx]
    private Tensor? _convPreW, _convPreB, _convPostW;
    private AntiAliasedSnake? _activationPost;
    private int _disposed;

    public IndexTts2BigVganGenerator(IndexTts2BigVganConfig cfg)
    {
        _cfg = cfg;
        _numStages = cfg.UpsampleRates.Length;
        _upsW = new Tensor?[_numStages];
        _upsB = new Tensor?[_numStages];
        _resblocks = new IndexTtsBigVganResBlock[_numStages][];
        for (int i = 0; i < _numStages; i++)
        {
            int ch = cfg.UpsampleInitialChannel / (1 << (i + 1));
            _resblocks[i] = new IndexTtsBigVganResBlock[cfg.ResblockKernelSizes.Length];
            for (int j = 0; j < cfg.ResblockKernelSizes.Length; j++)
                _resblocks[i][j] = new IndexTtsBigVganResBlock(ch, cfg.ResblockKernelSizes[j], cfg.ResblockDilations[j]);
        }
    }

    /// <param name="prefix">e.g. <c>generator</c> — the real checkpoint's own top-level module name.</param>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _convPreW = Own(WeightNormFusion.Compose(w, $"{prefix}.conv_pre"));
        _convPreB = EnsureF32(w[$"{prefix}.conv_pre.bias"]);

        for (int i = 0; i < _numStages; i++)
        {
            _upsW[i] = Own(WeightNormFusion.Compose(w, $"{prefix}.ups.{i}.0"));
            _upsB[i] = EnsureF32(w[$"{prefix}.ups.{i}.0.bias"]);
            for (int j = 0; j < _resblocks[i].Length; j++)
                _resblocks[i][j].LoadWeights(w, $"{prefix}.resblocks.{i * _resblocks[i].Length + j}");
        }

        int finalCh = _cfg.UpsampleInitialChannel / (1 << _numStages);
        _activationPost = new AntiAliasedSnake(finalCh);
        _activationPost.LoadWeights(w, $"{prefix}.activation_post");
        _convPostW = Own(WeightNormFusion.Compose(w, $"{prefix}.conv_post"));
    }

    /// <summary>Synthesizes 22050 Hz PCM from an 80-band mel <c>[1, inChannels, T]</c> (channel-first — the
    /// same layout the S2Mel CFM's output already has, no transpose needed). Returns <c>[1, 1, T·hop]</c>
    /// where <c>hop = Π upsampleRates = 256</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor mel, int melLen)
    {
        if (_convPreW is null) throw new InvalidOperationException("IndexTts2BigVganGenerator weights not loaded.");

        Tensor x = new(new TensorShape(1, _cfg.UpsampleInitialChannel, melLen), DType.F32);
        backend.Conv1d(x, mel, _convPreW!, _convPreB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);

        int t = melLen;
        for (int i = 0; i < _numStages; i++)
        {
            int outCh = _cfg.UpsampleInitialChannel / (1 << (i + 1));
            int rate = _cfg.UpsampleRates[i];
            int kernel = _cfg.UpsampleKernelSizes[i];
            int pad = (kernel - rate) / 2;
            int tUp = (t - 1) * rate + kernel - 2 * pad;

            Tensor up = new(new TensorShape(1, outCh, tUp), DType.F32);
            backend.ConvTranspose1d(up, x, _upsW[i]!, _upsB[i], rate, pad, pad, 1, 1);
            x.Dispose();
            x = up;
            t = tUp;

            Tensor? sum = null;
            foreach (IndexTtsBigVganResBlock rb in _resblocks[i])
            {
                Tensor rbOut = rb.Forward(backend, x);
                if (sum is null) sum = rbOut;
                else { backend.Add(sum, sum, rbOut); rbOut.Dispose(); }
            }
            x.Dispose();
            x = sum!;
            float invK = 1f / _resblocks[i].Length;
            backend.Scale(x, x, invK);
        }

        Tensor activated = _activationPost!.Forward(backend, x);
        x.Dispose();

        Tensor outWave = new(new TensorShape(1, 1, (int)activated.Shape[2]), DType.F32);
        backend.Conv1d(outWave, activated, _convPostW!, null, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);
        activated.Dispose();
        // No tanh — real config.json: use_tanh_at_final: false (unlike IndexTTS-1.5's own BigVGAN).
        return outWave;
    }

    private static Tensor EnsureF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_convPreW, _convPreB, _convPostW];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        for (int i = 0; i < _numStages; i++)
        {
            yield return _upsW[i]!;
            yield return _upsB[i]!;
            foreach (IndexTtsBigVganResBlock rb in _resblocks[i]) foreach (Tensor t in rb.EnumerateWeights()) yield return t;
        }
        if (_activationPost is not null) foreach (Tensor t in _activationPost.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (IndexTtsBigVganResBlock[] stage in _resblocks) foreach (IndexTtsBigVganResBlock rb in stage) rb.Dispose();
        _activationPost?.Dispose();
        foreach (Tensor t in _owned) t.Dispose();
        GC.SuppressFinalize(this);
    }
}

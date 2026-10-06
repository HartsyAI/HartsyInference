using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.IndexTts;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>BigVGAN-v2 vocoder (mel to waveform) at ControlFoley's 44.1 kHz setting; the residual blocks and
/// anti-aliased SnakeBeta activations are the shared AMPBlock1 / <see cref="AntiAliasedSnake"/> implementation.</summary>
public sealed class ControlFoleyBigVgan : IDisposable
{
    private readonly ControlFoleyBigVganConfig _cfg;
    private readonly IndexTtsBigVganResBlock[][] _resblocks;
    private readonly Tensor?[] _upsW;
    private readonly Tensor?[] _upsB;
    private readonly List<Tensor> _owned = [];
    private Tensor? _convPreW;
    private Tensor? _convPreB;
    private Tensor? _convPostW;
    private Tensor? _convPostB;
    private AntiAliasedSnake? _activationPost;
    private int _disposed;

    public ControlFoleyBigVgan(ControlFoleyBigVganConfig cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        _cfg = cfg;
        int stages = cfg.UpsampleRates.Length;
        if (cfg.UpsampleKernelSizes.Length != stages) throw new ArgumentException("Upsample kernel/rate count mismatch.", nameof(cfg));
        if (cfg.ResblockDilations.Length != cfg.ResblockKernelSizes.Length)
            throw new ArgumentException("Resblock kernel/dilation count mismatch.", nameof(cfg));
        _upsW = new Tensor?[stages];
        _upsB = new Tensor?[stages];
        _resblocks = new IndexTtsBigVganResBlock[stages][];
        for (int i = 0; i < stages; i++)
        {
            int ch = cfg.UpsampleInitialChannel >> (i + 1);
            _resblocks[i] = new IndexTtsBigVganResBlock[cfg.ResblockKernelSizes.Length];
            for (int j = 0; j < _resblocks[i].Length; j++)
                _resblocks[i][j] = new IndexTtsBigVganResBlock(ch, cfg.ResblockKernelSizes[j], cfg.ResblockDilations[j]);
        }
    }

    /// <summary>Samples produced per mel frame.</summary>
    public int HopLength => _cfg.HopLength;

    /// <summary>Loads the released generator keys (<c>conv_pre</c>, <c>ups.i.0</c>, <c>resblocks.n</c>, <c>activation_post</c>,
    /// <c>conv_post</c>), with weight norm either fused or still split into <c>weight_g</c>/<c>weight_v</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        _convPreW = Own(WeightNormFusion.Compose(w, $"{prefix}conv_pre"));
        _convPreB = F32(w[$"{prefix}conv_pre.bias"]);
        int kernels = _cfg.ResblockKernelSizes.Length;
        for (int i = 0; i < _upsW.Length; i++)
        {
            _upsW[i] = Own(WeightNormFusion.Compose(w, $"{prefix}ups.{i}.0"));
            _upsB[i] = F32(w[$"{prefix}ups.{i}.0.bias"]);
            for (int j = 0; j < kernels; j++) _resblocks[i][j].LoadWeights(w, $"{prefix}resblocks.{i * kernels + j}");
        }
        _activationPost = new AntiAliasedSnake(_cfg.UpsampleInitialChannel >> _upsW.Length);
        _activationPost.LoadWeights(w, $"{prefix}activation_post");
        _convPostW = Own(WeightNormFusion.Compose(w, $"{prefix}conv_post"));
        _convPostB = _cfg.UseBiasAtFinal ? F32(w[$"{prefix}conv_post.bias"]) : null;
    }

    /// <summary>Vocodes a mel <c>[1, NumMels, T]</c> into <c>[1, 1, T * HopLength]</c>; the caller owns the result.
    /// <paramref name="tap"/> receives <c>conv_pre</c>, <c>stage{i}</c> and <c>act_post</c> activations.</summary>
    public Tensor Forward(IBackend backend, Tensor mel, Action<string, Tensor>? tap = null)
    {
        if (_convPreW is null || _activationPost is null) throw new InvalidOperationException("ControlFoleyBigVgan weights not loaded.");
        if (mel.Shape.Rank != 3 || (int)mel.Shape[0] != 1 || (int)mel.Shape[1] != _cfg.NumMels)
            throw new ArgumentException($"Expected mel [1, {_cfg.NumMels}, T]; got {mel.Shape}.", nameof(mel));
        int t = (int)mel.Shape[2];
        Tensor x = new(new TensorShape(1, _cfg.UpsampleInitialChannel, t), DType.F32);
        backend.Conv1d(x, mel, _convPreW, _convPreB, 1, 3, 3, 1, 1);
        tap?.Invoke("conv_pre", x);
        for (int i = 0; i < _upsW.Length; i++)
        {
            int rate = _cfg.UpsampleRates[i];
            int kernel = _cfg.UpsampleKernelSizes[i];
            int pad = (kernel - rate) / 2;
            int outCh = _cfg.UpsampleInitialChannel >> (i + 1);
            t = (t - 1) * rate + kernel - 2 * pad;
            Tensor up = new(new TensorShape(1, outCh, t), DType.F32);
            backend.ConvTranspose1d(up, x, _upsW[i]!, _upsB[i], rate, pad, pad, 1, 1);
            x.Dispose();
            Tensor? sum = null;
            foreach (IndexTtsBigVganResBlock block in _resblocks[i])
            {
                Tensor y = block.Forward(backend, up);
                if (sum is null)
                {
                    sum = y;
                    continue;
                }
                backend.Add(sum, sum, y);
                y.Dispose();
            }
            up.Dispose();
            backend.Scale(sum!, sum!, 1f / _resblocks[i].Length);
            x = sum!;
            tap?.Invoke($"stage{i}", x);
        }
        Tensor activated = _activationPost.Forward(backend, x);
        x.Dispose();
        tap?.Invoke("act_post", activated);
        Tensor wave = new(new TensorShape(1, 1, (int)activated.Shape[2]), DType.F32);
        backend.Conv1d(wave, activated, _convPostW!, _convPostB, 1, 3, 3, 1, 1);
        activated.Dispose();
        if (_cfg.UseTanhAtFinal) backend.Tanh(wave, wave);
        else backend.Clamp(wave, wave, -1f, 1f);
        return wave;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (IndexTtsBigVganResBlock[] stage in _resblocks)
            foreach (IndexTtsBigVganResBlock block in stage) block.Dispose();
        _activationPost?.Dispose();
        foreach (Tensor t in _owned) t.Dispose();
        GC.SuppressFinalize(this);
    }

    private static Tensor F32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }
}

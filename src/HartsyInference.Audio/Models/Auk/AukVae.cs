using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Models;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>AuK BigVGAN-flow VAE decoder: 24 kHz mono PCM from 64-d latents at 50 Hz (hop 480).</summary>
/// <remarks>
/// <code>
///   conv_pre k7 (symmetric) -> 6 x { causal ConvTranspose (k=2s) -> mean of 3 AMPBlock1 (k 3/7/11) }
///   -> anti-aliased SnakeBeta -> conv_post k7 (no bias, causal) -> clamp[-1, 1]
/// </code>
/// Reads the upstream layout (<c>conv_pre</c>, <c>ups.N.0</c>, <c>resblocks.N</c>, <c>activation_post</c>, <c>conv_post</c>) under an optional prefix; <c>flow.*</c> and <c>audio_encoder.*</c> keys are ignored. Batch size is 1.
/// </remarks>
public sealed class AukVae : IAudioLatentDecoder, IDisposable
{
    private readonly AukVaeConfig _config;
    private readonly List<Tensor> _owned = [];
    private Tensor? _preWeight, _preBias, _postWeight;
    private Tensor[] _upWeights = [], _upBiases = [];
    private AmpBlock[][] _blocks = [];
    private AntiAliasedSnake? _activationPost;

    public AukVae(AukVaeConfig config)
    {
        config.Validate();
        _config = config;
    }

    /// <summary>Latent channels.</summary>
    public int LatentDim => _config.LatentDim;

    /// <summary>Samples per latent frame.</summary>
    public int HopLength => _config.Hop;

    /// <summary>Loads every decoder tensor under <paramref name="prefix"/> (e.g. <c>"vae."</c>, empty for a bare VAE checkpoint); a missing key throws.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        Dispose();
        bool loaded = false;
        try
        {
            LoadAll(weights, prefix);
            loaded = true;
        }
        finally
        {
            if (!loaded) Dispose();
        }
    }

    private void LoadAll(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _preWeight = Own(WeightNormFusion.LoadFused(weights, $"{prefix}conv_pre"));
        _preBias = weights[$"{prefix}conv_pre.bias"];
        int n = _config.UpsampleRates.Length;
        _upWeights = new Tensor[n];
        _upBiases = new Tensor[n];
        _blocks = new AmpBlock[n][];
        for (int i = 0; i < n; i++)
        {
            _upWeights[i] = Own(WeightNormFusion.LoadFused(weights, $"{prefix}ups.{i}.0"));
            _upBiases[i] = weights[$"{prefix}ups.{i}.0.bias"];
            int ch = _config.InitialChannels >> (i + 1);
            _blocks[i] = new AmpBlock[_config.ResblockKernelSizes.Length];
            for (int j = 0; j < _blocks[i].Length; j++)
            {
                _blocks[i][j] = new AmpBlock(_config, ch, _config.ResblockKernelSizes[j]);
                _blocks[i][j].LoadWeights(weights, $"{prefix}resblocks.{i * _blocks[i].Length + j}", Own);
            }
        }
        _activationPost = new AntiAliasedSnake(_config.FinalChannels, 2, _config.AntiAliasKernel, _config.ActCausal);
        _activationPost.LoadWeights(weights, $"{prefix}activation_post");
        _postWeight = Own(WeightNormFusion.LoadFused(weights, $"{prefix}conv_post"));
    }

    /// <summary>Decodes channels-first latents <c>[1, D, T]</c> to PCM <c>[1, 1, T * hop]</c> clamped to [-1, 1].</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        if (_preWeight is null || _activationPost is null || _postWeight is null)
            throw new InvalidOperationException("AukVae weights not loaded.");
        if (latent.Shape.Rank != 3 || (int)latent.Shape[0] != 1 || (int)latent.Shape[1] != _config.LatentDim)
            throw new ArgumentException($"Expected latents [1, {_config.LatentDim}, T]; got {latent.Shape}.", nameof(latent));
        int t = (int)latent.Shape[2];
        Tensor x = new(new TensorShape(1, _config.InitialChannels, t), DType.F32);
        backend.Conv1d(x, latent, _preWeight, _preBias, 1, 3, 3, 1, 1);
        for (int i = 0; i < _config.UpsampleRates.Length; i++)
        {
            int stride = _config.UpsampleRates[i];
            int kernel = 2 * stride;
            int padRight = _config.Causal ? kernel - stride : (kernel - stride) / 2;
            int padLeft = _config.Causal ? 0 : padRight;
            int tUp = (t - 1) * stride + kernel - padLeft - padRight;
            Tensor up = new(new TensorShape(1, _config.InitialChannels >> (i + 1), tUp), DType.F32);
            backend.ConvTranspose1d(up, x, _upWeights[i], _upBiases[i], stride, padLeft, padRight, 1, 1);
            x.Dispose();
            t = tUp;
            Tensor sum = _blocks[i][0].Forward(backend, up);
            for (int j = 1; j < _blocks[i].Length; j++)
            {
                using Tensor next = _blocks[i][j].Forward(backend, up);
                backend.Add(sum, sum, next);
            }
            up.Dispose();
            backend.Scale(sum, sum, 1f / _blocks[i].Length);
            x = sum;
        }
        Tensor act = _activationPost.Forward(backend, x);
        x.Dispose();
        Tensor pcm = new(new TensorShape(1, 1, t), DType.F32);
        backend.Conv1d(pcm, act, _postWeight, null, 1, _config.Causal ? 6 : 3, _config.Causal ? 0 : 3, 1, 1);
        act.Dispose();
        backend.Clamp(pcm, pcm, -1f, 1f);
        return pcm;
    }

    /// <summary>Decodes time-major latents <c>[1, T, D]</c> (the DiT output layout) via a device transpose.</summary>
    public Tensor DecodeTimeMajor(IBackend backend, Tensor latent)
    {
        if (latent.Shape.Rank != 3 || (int)latent.Shape[0] != 1 || (int)latent.Shape[2] != _config.LatentDim)
            throw new ArgumentException($"Expected latents [1, T, {_config.LatentDim}]; got {latent.Shape}.", nameof(latent));
        int t = (int)latent.Shape[1];
        using Tensor channelsFirst = new(new TensorShape(1, _config.LatentDim, t), DType.F32);
        backend.Transpose2D(channelsFirst, latent, t, _config.LatentDim);
        return Decode(backend, channelsFirst);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_preWeight is null) yield break;
        yield return _preWeight;
        yield return _preBias!;
        for (int i = 0; i < _upWeights.Length; i++)
        {
            yield return _upWeights[i];
            yield return _upBiases[i];
            foreach (AmpBlock b in _blocks[i])
                foreach (Tensor t in b.EnumerateWeights()) yield return t;
        }
        foreach (Tensor t in _activationPost!.EnumerateWeights()) yield return t;
        yield return _postWeight!;
    }

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
        foreach (AmpBlock?[]? stage in _blocks)
            if (stage is not null)
                foreach (AmpBlock? b in stage) b?.Dispose();
        _blocks = [];
        _activationPost?.Dispose();
        _activationPost = null;
        _preWeight = _preBias = _postWeight = null;
        _upWeights = _upBiases = [];
    }

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    /// <summary>One AMPBlock1: three (snake, dilated conv, snake, conv) residual pairs.</summary>
    private sealed class AmpBlock(AukVaeConfig config, int channels, int kernel) : IDisposable
    {
        private readonly int _pairs = config.ResblockDilations.Length;
        private readonly AntiAliasedSnake[] _acts = new AntiAliasedSnake[config.ResblockDilations.Length * 2];
        private readonly Tensor[] _convWeights = new Tensor[config.ResblockDilations.Length * 2];
        private readonly Tensor[] _convBiases = new Tensor[config.ResblockDilations.Length * 2];

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, Func<Tensor, Tensor> own)
        {
            for (int p = 0; p < _pairs; p++)
            {
                for (int h = 0; h < 2; h++)
                {
                    int idx = p * 2 + h;
                    string conv = $"{prefix}.convs{h + 1}.{p}";
                    _convWeights[idx] = own(WeightNormFusion.LoadFused(w, conv));
                    _convBiases[idx] = w[$"{conv}.bias"];
                    _acts[idx] = new AntiAliasedSnake(channels, 2, config.AntiAliasKernel, config.ActCausal);
                    _acts[idx].LoadWeights(w, $"{prefix}.activations.{idx}");
                }
            }
        }

        public Tensor Forward(IBackend backend, Tensor input)
        {
            Tensor x = new(input.Shape, DType.F32);
            backend.CopyTo(x, input);
            for (int p = 0; p < _pairs; p++)
            {
                Tensor cur = _acts[p * 2].Forward(backend, x);
                cur = Conv(backend, cur, p * 2, config.ResblockDilations[p]);
                cur = Activate(backend, cur, p * 2 + 1);
                cur = Conv(backend, cur, p * 2 + 1, 1);
                backend.Add(x, x, cur);
                cur.Dispose();
            }
            return x;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            for (int i = 0; i < _convWeights.Length; i++)
            {
                yield return _convWeights[i];
                yield return _convBiases[i];
                foreach (Tensor t in _acts[i].EnumerateWeights()) yield return t;
            }
        }

        public void Dispose()
        {
            foreach (AntiAliasedSnake a in _acts) a?.Dispose();
        }

        private Tensor Activate(IBackend backend, Tensor consumed, int idx)
        {
            Tensor o = _acts[idx].Forward(backend, consumed);
            consumed.Dispose();
            return o;
        }

        private Tensor Conv(IBackend backend, Tensor consumed, int idx, int dilation)
        {
            int span = dilation * (kernel - 1);
            Tensor o = new(consumed.Shape, DType.F32);
            backend.Conv1d(o, consumed, _convWeights[idx], _convBiases[idx], 1,
                config.Causal ? span : span / 2, config.Causal ? 0 : span / 2, dilation, 1);
            consumed.Dispose();
            return o;
        }
    }
}

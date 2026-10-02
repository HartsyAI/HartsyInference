using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>AuK VAE encoder: PCM <c>[1, 1, L]</c> to normalized time-major latents <c>[1, T, D]</c>. Non-causal (symmetric padding) and stochastic.</summary>
/// <remarks>
/// <code>
///   conv k3 -> LeakyReLU(0.2) -> per stride f: conv(k=2f, stride f, pad f-1) -> ResStack -> LeakyReLU(0.2) -> conv k3 -> [mean | log_std]
///   ResStack layer i: x + conv(k3) (LeakyReLU(0.01) -> conv dilation 2^i -> LeakyReLU(0.01) -> conv)
/// </code>
/// Latents are <c>z = mean + noise * exp(log_std)</c> then <c>(z - global_mean) / sqrt(global_var)</c>. Each strided conv yields <c>floor((L - 2) / f) + 1</c> frames, which is exactly <c>L / hop</c> only when L is a multiple of the hop: callers trim reference audio to a multiple of 480 samples before encoding.
/// Reads <c>{prefix}audio_encoder.generator.*</c>; the latent statistics come from <see cref="AukVaeStats"/>.
/// </remarks>
public sealed class AukVaeEncoder(AukVaeConfig config, AukVaeStats stats) : IDisposable
{
    private readonly List<Tensor> _owned = [];
    private readonly List<Tensor> _borrowed = [];
    private Tensor? _stemWeight, _stemBias, _headWeight, _headBias;
    private Tensor[] _downWeights = [], _downBiases = [];
    private Tensor[][] _stackWeights = [], _stackBiases = [];

    /// <summary>Latent frames produced for <paramref name="samples"/> input samples.</summary>
    public static int OutputFrames(AukVaeConfig config, int samples)
    {
        int t = samples;
        if (t < 2) throw new ArgumentOutOfRangeException(nameof(samples), "The encoder needs at least 2 samples.");
        foreach (int f in config.DownsampleRates) t = (t - 2) / f + 1;
        return t;
    }

    /// <summary>Loads the encoder convolutions; a missing key throws.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        Dispose();
        string g = $"{prefix}audio_encoder.generator";
        _stemWeight = Own(WeightNormFusion.LoadFused(weights, $"{g}.0.layer"));
        _stemBias = Borrow(weights[$"{g}.0.layer.bias"]);
        int n = config.DownsampleRates.Length;
        _downWeights = new Tensor[n];
        _downBiases = new Tensor[n];
        _stackWeights = new Tensor[n][];
        _stackBiases = new Tensor[n][];
        for (int b = 0; b < n; b++)
        {
            int baseIdx = 2 + 3 * b;
            _downWeights[b] = Own(WeightNormFusion.LoadFused(weights, $"{g}.{baseIdx}.layer"));
            _downBiases[b] = Borrow(weights[$"{g}.{baseIdx}.layer.bias"]);
            _stackWeights[b] = new Tensor[config.EncoderStackLayers * 2];
            _stackBiases[b] = new Tensor[config.EncoderStackLayers * 2];
            for (int i = 0; i < config.EncoderStackLayers; i++)
            {
                for (int h = 0; h < 2; h++)
                {
                    string conv = $"{g}.{baseIdx + 1}.layers.{i}.{(h == 0 ? 1 : 3)}";
                    _stackWeights[b][i * 2 + h] = Own(WeightNormFusion.LoadFused(weights, conv));
                    _stackBiases[b][i * 2 + h] = Borrow(weights[$"{conv}.bias"]);
                }
            }
        }
        string head = $"{g}.{2 + 3 * n}.layer";
        _headWeight = Own(WeightNormFusion.LoadFused(weights, head));
        _headBias = Borrow(weights[$"{head}.bias"]);
    }

    /// <summary>Encodes with Gaussian noise drawn from <see cref="DeterministicRng"/> seeded by <paramref name="seed"/>; the same seed and input give identical latents.</summary>
    public Tensor Encode(IBackend backend, Tensor pcm, int seed)
    {
        (Tensor mean, Tensor logStd) = Distribution(backend, pcm);
        using Tensor noise = new(mean.Shape, DType.F32);
        uint state = DeterministicRng.Seed(seed);
        unsafe
        {
            float* p = (float*)noise.DataPointer;
            for (long i = 0; i < noise.ElementCount; i++) p[i] = DeterministicRng.NextGaussian(ref state);
        }
        return Finish(backend, mean, logStd, noise);
    }

    /// <summary>Encodes with caller-supplied standard-normal <paramref name="noise"/> of shape <c>[1, D, T]</c> (T from <see cref="OutputFrames"/>).</summary>
    public Tensor Encode(IBackend backend, Tensor pcm, Tensor noise)
    {
        (Tensor mean, Tensor logStd) = Distribution(backend, pcm);
        if (!noise.Shape.Equals(mean.Shape))
        {
            string expected = mean.Shape.ToString();
            mean.Dispose();
            logStd.Dispose();
            throw new ArgumentException($"Expected noise {expected}; got {noise.Shape}.", nameof(noise));
        }
        return Finish(backend, mean, logStd, noise);
    }

    public IEnumerable<Tensor> EnumerateWeights() => [.. _owned, .. _borrowed];

    public void Dispose()
    {
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
        _borrowed.Clear();
    }

    private (Tensor Mean, Tensor LogStd) Distribution(IBackend backend, Tensor pcm)
    {
        if (_stemWeight is null || _headWeight is null) throw new InvalidOperationException("AukVaeEncoder weights not loaded.");
        if (pcm.Shape.Rank != 3 || pcm.Shape[0] != 1 || pcm.Shape[1] != 1)
            throw new ArgumentException($"Expected PCM [1, 1, L]; got {pcm.Shape}.", nameof(pcm));
        int t = (int)pcm.Shape[2];
        OutputFrames(config, t);
        Tensor x = new(new TensorShape(1, config.DownsampleChannels[0], t), DType.F32);
        backend.Conv1d(x, pcm, _stemWeight, _stemBias, 1, 1, 1, 1, 1);
        backend.LeakyRelu(x, x, 0.2f);
        for (int b = 0; b < config.DownsampleRates.Length; b++)
        {
            int f = config.DownsampleRates[b];
            int c = config.DownsampleChannels[b + 1];
            int tOut = (t - 2) / f + 1;
            Tensor down = new(new TensorShape(1, c, tOut), DType.F32);
            backend.Conv1d(down, x, _downWeights[b], _downBiases[b], f, f - 1, f - 1, 1, 1);
            x.Dispose();
            x = down;
            t = tOut;
            for (int i = 0; i < config.EncoderStackLayers; i++) ResidualLayer(backend, x, b, i);
            backend.LeakyRelu(x, x, 0.2f);
        }
        Tensor stats = new(new TensorShape(1, 2 * config.LatentDim, t), DType.F32);
        backend.Conv1d(stats, x, _headWeight, _headBias, 1, 1, 1, 1, 1);
        x.Dispose();
        Tensor mean = new(new TensorShape(1, config.LatentDim, t), DType.F32);
        Tensor logStd = new(new TensorShape(1, config.LatentDim, t), DType.F32);
        backend.Split([mean, logStd], stats, 1);
        stats.Dispose();
        return (mean, logStd);
    }

    private void ResidualLayer(IBackend backend, Tensor x, int block, int layer)
    {
        int dilation = 1 << layer;
        using Tensor a = new(x.Shape, DType.F32);
        backend.LeakyRelu(a, x, 0.01f);
        using Tensor b = new(x.Shape, DType.F32);
        backend.Conv1d(b, a, _stackWeights[block][layer * 2], _stackBiases[block][layer * 2], 1, dilation, dilation, dilation, 1);
        backend.LeakyRelu(b, b, 0.01f);
        backend.Conv1d(a, b, _stackWeights[block][layer * 2 + 1], _stackBiases[block][layer * 2 + 1], 1, 1, 1, 1, 1);
        backend.Add(x, x, a);
    }

    private Tensor Finish(IBackend backend, Tensor mean, Tensor logStd, Tensor noise)
    {
        int d = config.LatentDim;
        int t = (int)mean.Shape[2];
        using Tensor z = new(mean.Shape, DType.F32);
        unsafe
        {
            float* m = (float*)mean.DataPointer, s = (float*)logStd.DataPointer, n = (float*)noise.DataPointer, o = (float*)z.DataPointer;
            for (long i = 0; i < mean.ElementCount; i++) o[i] = m[i] + n[i] * MathF.Exp(s[i]);
        }
        mean.Dispose();
        logStd.Dispose();
        using Tensor timeMajor = new(new TensorShape(1, t, d), DType.F32);
        backend.Transpose2D(timeMajor, z, d, t);
        return stats.Normalize(backend, timeMajor);
    }

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    private Tensor Borrow(Tensor t)
    {
        _borrowed.Add(t);
        return t;
    }
}

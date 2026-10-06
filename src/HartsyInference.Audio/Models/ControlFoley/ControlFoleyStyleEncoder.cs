using HartsyInference.Audio.Io;
using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HubertModel = HartsyInference.Audio.Models.Hubert.Hubert;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>ControlFoley's timbre feature (<c>FeaturesUtils.encode_audio_with_music_model</c>): the style tokens the
/// MusicGen-Style conditioner computes for a 32 kHz reference clip, i.e. <c>[ceil(frames / 15), 1536]</c> projections of the
/// first-codebook-quantised MERT/transformer features, zeroed past the clip. <see cref="EncodeTimbre"/> adds the mean over time
/// <c>generate()</c> applies.</summary>
public sealed class ControlFoleyStyleEncoder : IDisposable
{
    private const float NormEps = 1e-5f;
    private const float BatchNormEps = 1e-5f;
    private const float MaxPeriod = 10_000f;
    private const int MinMertSamples = 400;

    private readonly ControlFoleyStyleConfig _config;
    private readonly HubertModel _mert;
    private Layer[]? _layers;
    private Tensor? _embedW, _embedB, _outW, _outB, _bnScale, _bnShift;
    private Tensor[]? _codebooks;
    private bool _loaded;

    /// <summary>Creates an unloaded encoder; call <see cref="LoadWeights"/> before encoding.</summary>
    public ControlFoleyStyleEncoder(ControlFoleyStyleConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Dim % config.Heads != 0 || config.Dim % 2 != 0)
        {
            throw new ArgumentException("Dim must be even and divisible by Heads.", nameof(config));
        }

        if (config.EvalCodebooks < 1 || config.EvalCodebooks > config.Codebooks || config.Downsample < 1)
        {
            throw new ArgumentException("EvalCodebooks must be within 1..Codebooks and Downsample positive.", nameof(config));
        }

        _config = config;
        _mert = new HubertModel(config.Mert);
    }

    /// <summary>Geometry this instance was built for.</summary>
    public ControlFoleyStyleConfig Config => _config;

    /// <summary>Binds a state dict holding the MERT network under <c>mert.</c> (HuggingFace Hubert names with <c>weight_g</c> /
    /// <c>weight_v</c> positional convolution) and the <c>self_wav</c> conditioner under <c>style.</c> (audiocraft names).</summary>
    public unsafe void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ControlFoleyStyleConfig c = _config;
        _mert.LoadWeights(weights, "mert.");
        _embedW = ControlFoleyClap.Require(weights, "style.embed.weight", c.Dim, c.Mert.Hidden);
        _embedB = ControlFoleyClap.Require(weights, "style.embed.bias", c.Dim);
        _layers = new Layer[c.Layers];
        for (int i = 0; i < _layers.Length; i++)
        {
            string p = $"style.transformer.layers.{i}";
            _layers[i] = new Layer(
                ControlFoleyClap.Require(weights, $"{p}.norm1.weight", c.Dim), ControlFoleyClap.Require(weights, $"{p}.norm1.bias", c.Dim),
                ControlFoleyClap.Require(weights, $"{p}.self_attn.in_proj_weight", 3 * c.Dim, c.Dim),
                ControlFoleyClap.Require(weights, $"{p}.self_attn.out_proj.weight", c.Dim, c.Dim),
                ControlFoleyClap.Require(weights, $"{p}.norm2.weight", c.Dim), ControlFoleyClap.Require(weights, $"{p}.norm2.bias", c.Dim),
                ControlFoleyClap.Require(weights, $"{p}.linear1.weight", 4 * c.Dim, c.Dim),
                ControlFoleyClap.Require(weights, $"{p}.linear2.weight", c.Dim, 4 * c.Dim));
        }

        Tensor mean = ControlFoleyClap.Require(weights, "style.batch_norm.running_mean", c.Dim);
        Tensor variance = ControlFoleyClap.Require(weights, "style.batch_norm.running_var", c.Dim);
        _bnScale = new Tensor(new TensorShape(c.Dim), DType.F32);
        _bnShift = new Tensor(new TensorShape(c.Dim), DType.F32);
        float* m = (float*)mean.DataPointer, v = (float*)variance.DataPointer;
        float* scale = (float*)_bnScale.DataPointer, shift = (float*)_bnShift.DataPointer;
        for (int i = 0; i < c.Dim; i++)
        {
            scale[i] = 1f / MathF.Sqrt(v[i] + BatchNormEps);
            shift[i] = -m[i] * scale[i];
        }

        _codebooks = new Tensor[c.EvalCodebooks];
        for (int q = 0; q < _codebooks.Length; q++)
        {
            _codebooks[q] = ControlFoleyClap.Require(weights, $"style.rvq.vq.layers.{q}._codebook.embed", c.Bins, c.Dim);
        }

        _outW = ControlFoleyClap.Require(weights, "style.output_proj.weight", c.OutputDim, c.Dim);
        _outB = ControlFoleyClap.Require(weights, "style.output_proj.bias", c.OutputDim);
        _loaded = true;
    }

    /// <summary>Style tokens <c>[frames, OutputDim]</c> of a mono 32 kHz clip whose official excerpt length equals the clip
    /// (ControlFoley passes <c>duration = samples / 32000</c>); frames past the clip are zero, as the official mask makes them.</summary>
    public float[] Encode(IBackend backend, ReadOnlySpan<float> audio32k)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (!_loaded)
        {
            throw new InvalidOperationException("Call LoadWeights before encoding.");
        }

        ControlFoleyStyleConfig c = _config;
        float[] mert = EncodeMert(backend, audio32k, out int frames);
        float[] x = DacOps.Linear(backend, mert, _embedW!, frames, c.Mert.Hidden, c.Dim, _embedB);
        AddSinusoidalPositions(x, frames, c.Dim);
        foreach (Layer layer in _layers!)
        {
            x = layer.Forward(backend, x, frames, c);
        }

        NormalizeBatch(x, frames);
        float[] quantized = Quantize(x, frames);
        int kept = (frames + c.Downsample - 1) / c.Downsample;
        float[] strided = new float[kept * c.Dim];
        for (int i = 0; i < kept; i++)
        {
            Array.Copy(quantized, i * c.Downsample * c.Dim, strided, i * c.Dim, c.Dim);
        }

        float[] tokens = DacOps.Linear(backend, strided, _outW!, kept, c.Dim, c.OutputDim, _outB);
        float downsampling = (float)(c.SampleRate / 75.0 * c.Downsample);
        float validFrames = audio32k.Length / downsampling;
        for (int i = 0; i < kept; i++)
        {
            if (i >= validFrames)
            {
                Array.Clear(tokens, i * c.OutputDim, c.OutputDim);
            }
        }

        return tokens;
    }

    /// <summary>The timbre feature <c>[OutputDim]</c> <c>generate()</c> feeds the network: the style tokens averaged over time
    /// (masked frames included).</summary>
    public float[] EncodeTimbre(IBackend backend, ReadOnlySpan<float> audio32k)
    {
        float[] tokens = Encode(backend, audio32k);
        int width = _config.OutputDim, frames = tokens.Length / width;
        float[] mean = new float[width];
        for (int t = 0; t < frames; t++)
        {
            for (int d = 0; d < width; d++)
            {
                mean[d] += tokens[t * width + d];
            }
        }

        for (int d = 0; d < width; d++)
        {
            mean[d] /= frames;
        }

        return mean;
    }

    /// <summary>MERT <c>last_hidden_state</c> <c>[frames, hidden]</c> of the clip resampled to 24 kHz.</summary>
    internal unsafe float[] EncodeMert(IBackend backend, ReadOnlySpan<float> audio32k, out int frames)
    {
        float[] pcm = JuliusResampler.Resample(audio32k, _config.SampleRate, _config.MertSampleRate);
        if (pcm.Length < MinMertSamples)
        {
            throw new ArgumentException($"Reference audio is too short ({audio32k.Length} samples).", nameof(audio32k));
        }

        using Tensor input = new(new TensorShape(1, 1, pcm.Length), DType.F32);
        pcm.AsSpan().CopyTo(new Span<float>((void*)input.DataPointer, pcm.Length));
        using Tensor hidden = _mert.Forward(backend, input, pcm.Length);
        int width = _config.Mert.Hidden;
        frames = (int)hidden.Shape[2];
        float[] result = new float[frames * width];
        float* src = (float*)hidden.DataPointer;
        for (int t = 0; t < frames; t++)
        {
            for (int d = 0; d < width; d++)
            {
                result[t * width + d] = src[d * frames + t];
            }
        }

        return result;
    }

    /// <summary>Intermediate checks: the batch-normalised transformer output for a clip.</summary>
    internal float[] EncodePreQuantizer(IBackend backend, ReadOnlySpan<float> audio32k, out int frames)
    {
        ControlFoleyStyleConfig c = _config;
        float[] mert = EncodeMert(backend, audio32k, out frames);
        float[] x = DacOps.Linear(backend, mert, _embedW!, frames, c.Mert.Hidden, c.Dim, _embedB);
        AddSinusoidalPositions(x, frames, c.Dim);
        foreach (Layer layer in _layers!)
        {
            x = layer.Forward(backend, x, frames, c);
        }

        NormalizeBatch(x, frames);
        return x;
    }

    /// <summary>Releases nothing the caller owns; weights stay with the loader that produced them.</summary>
    public void Dispose() => _mert.Dispose();

    /// <summary><c>create_sin_embedding</c>: <c>[cos(phase), sin(phase)]</c> with <c>phase = pos / max_period^(i / (half - 1))</c>.</summary>
    private static void AddSinusoidalPositions(float[] x, int frames, int dim)
    {
        int half = dim / 2;
        for (int t = 0; t < frames; t++)
        {
            for (int i = 0; i < half; i++)
            {
                float phase = t / MathF.Pow(MaxPeriod, i / (float)(half - 1));
                x[t * dim + i] += MathF.Cos(phase);
                x[t * dim + half + i] += MathF.Sin(phase);
            }
        }
    }

    private unsafe void NormalizeBatch(float[] x, int frames)
    {
        int dim = _config.Dim;
        float* scale = (float*)_bnScale!.DataPointer, shift = (float*)_bnShift!.DataPointer;
        for (int t = 0; t < frames; t++)
        {
            for (int d = 0; d < dim; d++)
            {
                x[t * dim + d] = x[t * dim + d] * scale[d] + shift[d];
            }
        }
    }

    /// <summary>Residual vector quantisation with the first <see cref="ControlFoleyStyleConfig.EvalCodebooks"/> codebooks:
    /// each stage takes the nearest codeword by <c>2 x.e - |e|^2</c> and subtracts it from the residual.</summary>
    private unsafe float[] Quantize(float[] x, int frames)
    {
        int dim = _config.Dim, bins = _config.Bins;
        float[] residual = (float[])x.Clone();
        float[] quantized = new float[x.Length];
        foreach (Tensor codebook in _codebooks!)
        {
            float* e = (float*)codebook.DataPointer;
            float[] norms = new float[bins];
            for (int b = 0; b < bins; b++)
            {
                float sum = 0f;
                for (int d = 0; d < dim; d++)
                {
                    sum += e[b * dim + d] * e[b * dim + d];
                }

                norms[b] = sum;
            }

            for (int t = 0; t < frames; t++)
            {
                float best = float.NegativeInfinity;
                int index = 0;
                float xNorm = 0f;
                for (int d = 0; d < dim; d++)
                {
                    xNorm += residual[t * dim + d] * residual[t * dim + d];
                }

                for (int b = 0; b < bins; b++)
                {
                    float dot = 0f;
                    for (int d = 0; d < dim; d++)
                    {
                        dot += residual[t * dim + d] * e[b * dim + d];
                    }

                    float score = -(xNorm - 2f * dot + norms[b]);
                    if (score > best)
                    {
                        best = score;
                        index = b;
                    }
                }

                for (int d = 0; d < dim; d++)
                {
                    float word = e[index * dim + d];
                    quantized[t * dim + d] += word;
                    residual[t * dim + d] -= word;
                }
            }
        }

        return quantized;
    }

    private sealed record Layer(Tensor Norm1W, Tensor Norm1B, Tensor InW, Tensor OutW, Tensor Norm2W, Tensor Norm2B, Tensor Fc1W,
        Tensor Fc2W)
    {
        internal float[] Forward(IBackend backend, float[] x, int frames, ControlFoleyStyleConfig c)
        {
            int dim = c.Dim, heads = c.Heads, headDim = dim / heads;
            float[] normed = ControlFoleyClipTransformer.LayerNorm(backend, x, Norm1W, Norm1B, frames, dim, NormEps);
            float[] qkv = DacOps.Linear(backend, normed, InW, frames, dim, 3 * dim);
            float[] merged = new float[frames * dim];
            float[] scores = new float[frames];
            float scale = 1f / MathF.Sqrt(headDim);
            for (int h = 0; h < heads; h++)
            {
                for (int i = 0; i < frames; i++)
                {
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < frames; j++)
                    {
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                        {
                            dot += qkv[i * 3 * dim + h * headDim + d] * qkv[j * 3 * dim + dim + h * headDim + d];
                        }

                        scores[j] = dot * scale;
                        max = MathF.Max(max, scores[j]);
                    }

                    float sum = 0f;
                    for (int j = 0; j < frames; j++)
                    {
                        scores[j] = MathF.Exp(scores[j] - max);
                        sum += scores[j];
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        float acc = 0f;
                        for (int j = 0; j < frames; j++)
                        {
                            acc += scores[j] / sum * qkv[j * 3 * dim + 2 * dim + h * headDim + d];
                        }

                        merged[i * dim + h * headDim + d] = acc;
                    }
                }
            }

            float[] attn = DacOps.Linear(backend, merged, OutW, frames, dim, dim);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += attn[i];
            }

            float[] hidden = DacOps.Linear(backend, ControlFoleyClipTransformer.LayerNorm(backend, x, Norm2W, Norm2B, frames, dim, NormEps),
                Fc1W, frames, dim, 4 * dim);
            for (int i = 0; i < hidden.Length; i++)
            {
                hidden[i] = Activations.ErfGelu(hidden[i]);
            }

            float[] ff = DacOps.Linear(backend, hidden, Fc2W, frames, 4 * dim, dim);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += ff[i];
            }

            return x;
        }
    }
}

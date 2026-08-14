using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>Projects MiniMax Music 3's per-frame autoregressive hidden states onto the Flow-VAE latent timeline.
/// Each frame carries eight 4096-wide hidden states (the language model's, then one per residual-codebook step);
/// they are mixed with learned softmax weights, scaled, projected by a <c>k=3</c> convolution, and resampled from
/// the 25 Hz frame rate to the ~86 Hz latent rate with nearest-neighbour interpolation.</summary>
public sealed unsafe class MiniMaxMusic3ConditionEncoder : IDisposable
{
    /// <summary>Latent frames per autoregressive frame: <c>44100/24000 · 960/512</c>.</summary>
    public const double LatentsPerFrame = 44100.0 / 24000.0 * 960.0 / 512.0;

    private const int Layers = 8;
    private const int HiddenDim = 4096;
    private const int OutDim = 2048;
    private const int Kernel = 3;

    private readonly float[] _layerWeights = new float[Layers];
    private Tensor? _projWeight;
    private Tensor? _projBias;
    private float _layerScale = 1f;
    private int _disposed;

    /// <summary>Latent length for <paramref name="frames"/> autoregressive frames, matching the reference's
    /// truncating conversion.</summary>
    public static int LatentLength(int frames) => Math.Max(1, (int)(frames * LatentsPerFrame));

    /// <summary>Reads the four checkpoint tensors and pre-applies the softmax over the layer logits.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ReadOnlySpan<float> logits = weights["layer_weight_logits"].AsReadOnlySpan<float>();
        float max = float.NegativeInfinity;
        for (int i = 0; i < Layers; i++)
        {
            max = Math.Max(max, logits[i]);
        }
        float sum = 0f;
        for (int i = 0; i < Layers; i++)
        {
            _layerWeights[i] = MathF.Exp(logits[i] - max);
            sum += _layerWeights[i];
        }
        for (int i = 0; i < Layers; i++)
        {
            _layerWeights[i] /= sum;
        }
        _layerScale = weights["layer_scale"].AsReadOnlySpan<float>()[0];
        _projWeight = weights["proj.weight"];
        _projBias = weights["proj.bias"];
    }

    /// <summary>Encodes host-side frame hiddens <c>[frames, 8·4096]</c> starting at <paramref name="frameOffset"/>
    /// into a latent-aligned conditioning tensor <c>[1, latentLength, 2048]</c>. Caller owns the result.</summary>
    public Tensor Encode(IBackend backend, ReadOnlySpan<float> frameHiddens, int frameOffset, int frameCount)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_projWeight is null)
        {
            throw new InvalidOperationException($"{nameof(MiniMaxMusic3ConditionEncoder)}.{nameof(LoadWeights)} must run before encoding.");
        }
        if (frameCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(frameCount), frameCount, "frameCount must be positive.");
        }

        // The layer mix collapses 32768 channels to 4096 per frame, so it runs host-side on the frame hiddens
        // (which the autoregressive stage produced on the host) rather than uploading 8× the data.
        using Tensor mixed = new Tensor(new TensorShape(1, HiddenDim, frameCount), DType.F32);
        float* destination = (float*)mixed.DataPointer;
        fixed (float* source = frameHiddens)
        {
            for (int frame = 0; frame < frameCount; frame++)
            {
                float* row = source + (long)(frameOffset + frame) * Layers * HiddenDim;
                for (int channel = 0; channel < HiddenDim; channel++)
                {
                    float accumulator = 0f;
                    for (int layer = 0; layer < Layers; layer++)
                    {
                        accumulator += _layerWeights[layer] * row[layer * HiddenDim + channel];
                    }
                    destination[(long)channel * frameCount + frame] = _layerScale * accumulator;
                }
            }
        }

        using Tensor projected = new Tensor(new TensorShape(1, OutDim, frameCount), DType.F32);
        backend.Conv1d(projected, mixed, _projWeight!, _projBias, 1, Kernel / 2, Kernel / 2, 1, 1);

        int latentLength = LatentLength(frameCount);
        Tensor condition = new Tensor(new TensorShape(1, latentLength, OutDim), DType.F32);
        ReadOnlySpan<float> projectedValues = projected.AsReadOnlySpan<float>();
        float* conditionData = (float*)condition.DataPointer;
        for (int latent = 0; latent < latentLength; latent++)
        {
            // PyTorch's nearest-neighbour interpolate maps destination → floor(dst · in / out).
            int frame = (int)((long)latent * frameCount / latentLength);
            for (int channel = 0; channel < OutDim; channel++)
            {
                conditionData[(long)latent * OutDim + channel] = projectedValues[channel * frameCount + frame];
            }
        }
        return condition;
    }

    /// <summary>Every loaded weight, for <see cref="IBackend.PreloadWeights"/>/<see cref="IBackend.FreeWeights"/>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_projWeight is not null) { yield return _projWeight; }
        if (_projBias is not null) { yield return _projBias; }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        _projWeight = null;
        _projBias = null;
    }
}

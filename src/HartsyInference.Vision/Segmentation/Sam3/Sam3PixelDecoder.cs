using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>SAM 3 pixel decoder: each stage upsamples the running map 2x (nearest), adds the next finer level,
/// then applies a 3x3 convolution, GroupNorm and ReLU.</summary>
public sealed class Sam3PixelDecoder
{
    private const int GroupCount = 8;
    private const string Prefix = "pixel_decoder.";

    private readonly int _channels;
    private readonly float _eps;
    private readonly Tensor[] _convWeights;
    private readonly Tensor[] _convBiases;
    private readonly Tensor[] _normWeights;
    private readonly Tensor[] _normBiases;

    /// <summary>Loads the stages that run on the image path; the coarsest level seeds the decoder, one stage per finer level.</summary>
    public Sam3PixelDecoder(Sam3Config config, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        int stages = config.PixelDecoderStages - 1;
        if (stages < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(config), config.PixelDecoderStages,
                "The pixel decoder needs at least two declared stages.");
        }

        _channels = config.DModel;
        _eps = config.LayerNormEps;
        _convWeights = new Tensor[stages];
        _convBiases = new Tensor[stages];
        _normWeights = new Tensor[stages];
        _normBiases = new Tensor[stages];
        for (int i = 0; i < stages; i++)
        {
            string conv = $"{Prefix}conv_layers.{i}.";
            string norm = $"{Prefix}norms.{i}.";
            _convWeights[i] = Sam3WeightAccess.Require(weights, conv + "weight", _channels, _channels, 3, 3);
            _convBiases[i] = Sam3WeightAccess.Require(weights, conv + "bias", _channels);
            _normWeights[i] = Sam3WeightAccess.Require(weights, norm + "weight", _channels);
            _normBiases[i] = Sam3WeightAccess.Require(weights, norm + "bias", _channels);
        }
    }

    /// <summary>Fuses <paramref name="levels"/> ordered finest first into one map at the finest resolution; the caller owns the result.</summary>
    public Tensor Forward(IBackend backend, Tensor[] levels)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(levels);
        if (levels.Length != _convWeights.Length + 1)
        {
            throw new ArgumentException($"Pixel decoder expects {_convWeights.Length + 1} levels; got {levels.Length}.", nameof(levels));
        }

        for (int i = 0; i < levels.Length; i++)
        {
            Tensor level = levels[i];
            if (level.DType != DType.F32 || level.Shape.Rank != 4 || level.Shape[0] != 1 || level.Shape[1] != _channels)
            {
                throw new ArgumentException($"Pixel decoder level {i} must be F32 [1, {_channels}, H, W]; got {level.DType} {level.Shape}.", nameof(levels));
            }

            if (i > 0 && (2 * level.Shape[2] != levels[i - 1].Shape[2] || 2 * level.Shape[3] != levels[i - 1].Shape[3]))
            {
                throw new ArgumentException($"Pixel decoder level {i} must be half the size of level {i - 1}; got {level.Shape}.", nameof(levels));
            }
        }

        Tensor running = levels[levels.Length - 1];
        bool ownsRunning = false;
        for (int stage = 0; stage < _convWeights.Length; stage++)
        {
            Tensor fine = levels[levels.Length - 2 - stage];
            Tensor up = new Tensor(fine.Shape, DType.F32);
            backend.UpsampleNearest2D(up, running, 2, 2);

            Tensor fused = new Tensor(fine.Shape, DType.F32);
            backend.Add(fused, fine, up);
            up.Dispose();

            Tensor conv = new Tensor(fine.Shape, DType.F32);
            backend.Conv2D(conv, fused, _convWeights[stage], _convBiases[stage], 1, 1, 1, 1);
            fused.Dispose();

            Tensor norm = new Tensor(fine.Shape, DType.F32);
            backend.GroupNorm(norm, conv, _normWeights[stage], _normBiases[stage], GroupCount, _eps);
            conv.Dispose();
            backend.Clamp(norm, norm, 0f, float.MaxValue);

            if (ownsRunning)
            {
                running.Dispose();
            }

            running = norm;
            ownsRunning = true;
        }

        return running;
    }
}

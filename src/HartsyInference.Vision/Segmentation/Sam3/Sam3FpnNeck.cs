using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>SAM 3 simple-FPN neck. Builds the scales kept after scalp (4x, 2x, 1x); the 0.5x level is never computed.</summary>
public sealed class Sam3FpnNeck
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _grid;
    private readonly Branch[] _branches;

    /// <summary>Loads the lateral and output convolutions of every kept scale.</summary>
    /// <param name="config">Neck hyperparameters; scalp drops the finest <c>ScalpLevels</c> scales.</param>
    /// <param name="weights">Neck weights keyed relative to <c>...vision_backbone.</c>, for example <c>convs.0.conv_1x1.weight</c>.</param>
    public Sam3FpnNeck(Sam3Config config, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);

        int kept = config.NeckScaleFactors.Length - config.ScalpLevels;
        if (kept < 1)
            throw new ArgumentException("Scalp removes every neck level.", nameof(config));

        _inChannels = config.VitEmbedDim;
        _outChannels = config.DModel;
        _grid = config.PatchGrid;
        _branches = new Branch[kept];
        for (int i = 0; i < kept; i++)
        {
            _branches[i] = BuildBranch(config, weights, i, config.NeckScaleFactors[i]);
        }
    }

    /// <summary>Runs every kept scale on the trunk map and returns features and their sine positions, coarse to fine.</summary>
    /// <param name="backend">Backend that runs the ops.</param>
    /// <param name="trunkFeatures">F32 <c>[1, VitEmbedDim, grid, grid]</c> trunk output; not disposed.</param>
    /// <returns>Owned feature and position tensors; the caller disposes them.</returns>
    public Sam3VisionFeatures Forward(IBackend backend, Tensor trunkFeatures)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(trunkFeatures);
        if (trunkFeatures.DType != DType.F32 || trunkFeatures.Shape.Rank != 4 || trunkFeatures.Shape[0] != 1
            || trunkFeatures.Shape[1] != _inChannels || trunkFeatures.Shape[2] != _grid || trunkFeatures.Shape[3] != _grid)
        {
            throw new ArgumentException(
                $"Neck expects F32 trunk features [1,{_inChannels},{_grid},{_grid}]; got {trunkFeatures.DType} {trunkFeatures.Shape}.",
                nameof(trunkFeatures));
        }

        Tensor[] features = new Tensor[_branches.Length];
        Tensor[] positions = new Tensor[_branches.Length];
        for (int i = 0; i < _branches.Length; i++)
        {
            features[i] = RunBranch(backend, _branches[i], trunkFeatures);
            positions[i] = Sam3PositionEncoding.Create(_outChannels, (int)features[i].Shape[2], (int)features[i].Shape[3]);
        }

        return new Sam3VisionFeatures(features, positions);
    }

    private static Tensor RunBranch(IBackend backend, Branch branch, Tensor trunkFeatures)
    {
        Tensor current = trunkFeatures;
        for (int i = 0; i < branch.DeconvWeights.Length; i++)
        {
            Tensor upsampled = Deconv2x2(backend, current, branch.DeconvWeights[i], branch.DeconvBiases[i]);
            if (!ReferenceEquals(current, trunkFeatures))
            {
                current.Dispose();
            }

            current = upsampled;
            if (i == 0 && branch.GeluAfterFirstDeconv)
            {
                backend.GeluErf(current, current);
            }
        }

        Tensor lateral = Conv(backend, current, branch.Conv1Weight, branch.Conv1Bias);
        if (!ReferenceEquals(current, trunkFeatures))
        {
            current.Dispose();
        }

        Tensor fused = Conv(backend, lateral, branch.Conv3Weight, branch.Conv3Bias);
        lateral.Dispose();
        return fused;
    }

    private static Branch BuildBranch(Sam3Config config, IReadOnlyDictionary<string, Tensor> weights, int index, float scale)
    {
        int inDim = config.VitEmbedDim;
        int outDim = config.DModel;
        string prefix = $"convs.{index}.";

        if (scale == 4f)
        {
            int mid = inDim / 2;
            int low = inDim / 4;
            return new Branch(
                [
                    Sam3WeightAccess.Require(weights, prefix + "dconv_2x2_0.weight", inDim, mid, 2, 2),
                    Sam3WeightAccess.Require(weights, prefix + "dconv_2x2_1.weight", mid, low, 2, 2),
                ],
                [
                    Sam3WeightAccess.Require(weights, prefix + "dconv_2x2_0.bias", mid),
                    Sam3WeightAccess.Require(weights, prefix + "dconv_2x2_1.bias", low),
                ],
                geluAfterFirstDeconv: true,
                conv1Weight: Sam3WeightAccess.Require(weights, prefix + "conv_1x1.weight", outDim, low, 1, 1),
                conv1Bias: Sam3WeightAccess.Require(weights, prefix + "conv_1x1.bias", outDim),
                conv3Weight: Sam3WeightAccess.Require(weights, prefix + "conv_3x3.weight", outDim, outDim, 3, 3),
                conv3Bias: Sam3WeightAccess.Require(weights, prefix + "conv_3x3.bias", outDim));
        }

        if (scale == 2f)
        {
            int mid = inDim / 2;
            return new Branch(
                [Sam3WeightAccess.Require(weights, prefix + "dconv_2x2.weight", inDim, mid, 2, 2)],
                [Sam3WeightAccess.Require(weights, prefix + "dconv_2x2.bias", mid)],
                geluAfterFirstDeconv: false,
                conv1Weight: Sam3WeightAccess.Require(weights, prefix + "conv_1x1.weight", outDim, mid, 1, 1),
                conv1Bias: Sam3WeightAccess.Require(weights, prefix + "conv_1x1.bias", outDim),
                conv3Weight: Sam3WeightAccess.Require(weights, prefix + "conv_3x3.weight", outDim, outDim, 3, 3),
                conv3Bias: Sam3WeightAccess.Require(weights, prefix + "conv_3x3.bias", outDim));
        }

        if (scale == 1f)
        {
            return new Branch(
                [],
                [],
                geluAfterFirstDeconv: false,
                conv1Weight: Sam3WeightAccess.Require(weights, prefix + "conv_1x1.weight", outDim, inDim, 1, 1),
                conv1Bias: Sam3WeightAccess.Require(weights, prefix + "conv_1x1.bias", outDim),
                conv3Weight: Sam3WeightAccess.Require(weights, prefix + "conv_3x3.weight", outDim, outDim, 3, 3),
                conv3Bias: Sam3WeightAccess.Require(weights, prefix + "conv_3x3.bias", outDim));
        }

        throw new NotSupportedException(
            $"Neck scale {scale} is not supported; only 4, 2 and 1 are computed. Scale 0.5 is dropped by scalp.");
    }

    private static Tensor Deconv2x2(IBackend backend, Tensor input, Tensor weight, Tensor bias)
    {
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];
        int outChannels = (int)weight.Shape[1];
        Tensor output = new Tensor(new TensorShape(1, outChannels, 2 * height, 2 * width), DType.F32);
        backend.ConvTranspose2d(output, input, weight, bias, 2, 2, 0, 0);
        return output;
    }

    // Same-size convolution: the padding is kernel/2, which is 0 for 1x1 and 1 for 3x3.
    private static Tensor Conv(IBackend backend, Tensor input, Tensor weight, Tensor bias)
    {
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];
        int outChannels = (int)weight.Shape[0];
        int pad = (int)weight.Shape[2] / 2;
        Tensor output = new Tensor(new TensorShape(1, outChannels, height, width), DType.F32);
        backend.Conv2D(output, input, weight, bias, 1, 1, pad, pad);
        return output;
    }

    private sealed class Branch(
        Tensor[] deconvWeights,
        Tensor[] deconvBiases,
        bool geluAfterFirstDeconv,
        Tensor conv1Weight,
        Tensor conv1Bias,
        Tensor conv3Weight,
        Tensor conv3Bias)
    {
        public Tensor[] DeconvWeights { get; } = deconvWeights;

        public Tensor[] DeconvBiases { get; } = deconvBiases;

        public bool GeluAfterFirstDeconv { get; } = geluAfterFirstDeconv;

        public Tensor Conv1Weight { get; } = conv1Weight;

        public Tensor Conv1Bias { get; } = conv1Bias;

        public Tensor Conv3Weight { get; } = conv3Weight;

        public Tensor Conv3Bias { get; } = conv3Bias;
    }
}

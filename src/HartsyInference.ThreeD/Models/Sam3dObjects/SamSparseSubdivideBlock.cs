using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// <c>SparseSubdivideBlock3d</c> of SAM 3D's mesh decoder: <c>GroupNorm(32) · SiLU</c>, a subdivision that turns every voxel into its eight children (coordinates doubled, features copied),
/// <c>conv3 · GroupNorm(32) · SiLU · conv3</c>, plus the subdivided input through a skip (the identity, or a 1 x 1 x 1 convolution when the channel counts differ). The group norm takes
/// its statistics over all voxels, as upstream's does for a batch of one.
/// </summary>
internal sealed unsafe class SamSparseSubdivideBlock
{
    private const int Groups = 32;
    private const float Eps = 1e-5f;

    private readonly Tensor _norm1Weight, _norm1Bias, _norm2Weight, _norm2Bias;
    private readonly Tensor[] _conv1Slices, _conv2Slices;
    private readonly Tensor _conv1Bias, _conv2Bias;
    private readonly Tensor? _skipWeight, _skipBias;
    private readonly int _channels, _outChannels;

    public SamSparseSubdivideBlock(IReadOnlyDictionary<string, Tensor> weights, string prefix, int channels, int outChannels)
    {
        ArgumentNullException.ThrowIfNull(weights);
        if (channels % Groups != 0 || outChannels % Groups != 0)
            throw new ArgumentException($"Group norm of {Groups} groups needs channel counts that divide by {Groups}; got {channels} and {outChannels}.");
        _channels = channels;
        _outChannels = outChannels;
        _norm1Weight = SamWeights.Require(weights, prefix + ".act_layers.0.weight", channels);
        _norm1Bias = SamWeights.Require(weights, prefix + ".act_layers.0.bias", channels);
        _conv1Slices = SparseOps.ConvWeightSlices(SamWeights.Require(weights, prefix + ".out_layers.0.conv.weight", outChannels, 3, 3, 3, channels));
        _conv1Bias = SamWeights.Require(weights, prefix + ".out_layers.0.conv.bias", outChannels);
        _norm2Weight = SamWeights.Require(weights, prefix + ".out_layers.1.weight", outChannels);
        _norm2Bias = SamWeights.Require(weights, prefix + ".out_layers.1.bias", outChannels);
        _conv2Slices = SparseOps.ConvWeightSlices(SamWeights.Require(weights, prefix + ".out_layers.3.conv.weight", outChannels, 3, 3, 3, outChannels));
        _conv2Bias = SamWeights.Require(weights, prefix + ".out_layers.3.conv.bias", outChannels);
        if (outChannels != channels)
        {
            Tensor skip = SamWeights.Require(weights, prefix + ".skip_connection.conv.weight", outChannels, 1, 1, 1, channels);
            _skipWeight = skip.Reshape(new TensorShape(outChannels, channels));
            _skipBias = SamWeights.Require(weights, prefix + ".skip_connection.conv.bias", outChannels);
        }
    }

    /// <summary>Every tensor the block reads, for <c>IBackend.PreloadWeights</c> and <c>FreeWeights</c>.</summary>
    public IEnumerable<Tensor> Weights()
    {
        foreach (Tensor t in new[] { _norm1Weight, _norm1Bias, _norm2Weight, _norm2Bias, _conv1Bias, _conv2Bias })
            yield return t;
        foreach (Tensor t in _conv1Slices)
            yield return t;
        foreach (Tensor t in _conv2Slices)
            yield return t;
        if (_skipWeight is not null)
        {
            yield return _skipWeight;
            yield return _skipBias!;
        }
    }

    /// <summary>The block over <paramref name="x"/> (features <c>[1, N, channels]</c>); the result has <c>8 N</c> voxels at twice the resolution and <c>[1, 8 N, outChannels]</c> features. The input is not disposed.</summary>
    public SparseTensor Forward(IBackend backend, SparseTensor x)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (x.Channels != _channels)
            throw new ArgumentException($"The block takes {_channels} channels; got {x.Channels}.", nameof(x));

        Tensor normed = GroupNorm(x.Feats, _norm1Weight, _norm1Bias, _channels);
        Tensor activated = new(normed.Shape, DType.F32);
        backend.Silu(activated, normed);
        normed.Dispose();
        SparseTensor subdividedActivation = Subdivide(x.Replace(activated));
        activated.Dispose();
        SparseTensor subdividedInput = Subdivide(x);

        SparseTensor c1 = SparseOps.SubmanifoldConv3dSparse(backend, subdividedActivation, _conv1Slices, _conv1Bias);
        subdividedActivation.Feats.Dispose();
        Tensor normed2 = GroupNorm(c1.Feats, _norm2Weight, _norm2Bias, _outChannels);
        Tensor activated2 = new(normed2.Shape, DType.F32);
        backend.Silu(activated2, normed2);
        normed2.Dispose();
        SparseTensor c2 = SparseOps.SubmanifoldConv3dSparse(backend, c1.Replace(activated2), _conv2Slices, _conv2Bias);
        activated2.Dispose();
        c1.Feats.Dispose();

        int rows = c2.Count;
        Tensor result = new(new TensorShape(1, rows, _outChannels), DType.F32);
        if (_skipWeight is null)
        {
            backend.Add(result, SparseOps.As3D(c2.Feats), SparseOps.As3D(subdividedInput.Feats));
        }
        else
        {
            Tensor skip = new(new TensorShape(1, rows, _outChannels), DType.F32);
            backend.Linear(skip, SparseOps.As3D(subdividedInput.Feats), _skipWeight, _skipBias!);
            backend.Add(result, SparseOps.As3D(c2.Feats), skip);
            skip.Dispose();
        }

        c2.Feats.Dispose();
        subdividedInput.Feats.Dispose();
        return c2.Replace(result);
    }

    /// <summary>
    /// Every voxel becomes its eight children, in the row-major order of a 2 x 2 x 2 cube (<c>torch.nonzero</c> of ones): coordinates are doubled and offset by 0 or 1 per axis, the
    /// child index is <c>4 dx + 2 dy + dz</c>, and the features are copied. Row <c>8 i + k</c> is child <c>k</c> of voxel <c>i</c>.
    /// </summary>
    internal static SparseTensor Subdivide(SparseTensor x)
    {
        int n = x.Count, c = x.Channels;
        int[] coords = new int[n * 8 * 4];
        Tensor feats = new(new TensorShape(1, n * 8, c), DType.F32);
        float* source = (float*)x.Feats.DataPointer;
        float* target = (float*)feats.DataPointer;
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < 8; k++)
            {
                int row = i * 8 + k;
                coords[row * 4] = x.Coords[i * 4];
                coords[row * 4 + 1] = x.Coords[i * 4 + 1] * 2 + ((k >> 2) & 1);
                coords[row * 4 + 2] = x.Coords[i * 4 + 2] * 2 + ((k >> 1) & 1);
                coords[row * 4 + 3] = x.Coords[i * 4 + 3] * 2 + (k & 1);
                Buffer.MemoryCopy(source + (long)i * c, target + (long)row * c, (long)c * 4, (long)c * 4);
            }
        }

        return new SparseTensor(feats, coords, x.Resolution * 2);
    }

    /// <summary>
    /// <c>torch.nn.GroupNorm(32, C)</c> over a <c>[N, C]</c> feature matrix read as <c>[1, C, N]</c>: the mean and biased variance of each group of <c>C / 32</c> channels are taken over
    /// all voxels and channels of the group, in double precision, then the affine is applied per channel.
    /// </summary>
    internal static Tensor GroupNorm(Tensor feats, Tensor weight, Tensor bias, int channels)
    {
        long n = feats.ElementCount / channels;
        int perGroup = channels / Groups;
        Tensor result = new(new TensorShape(1, n, channels), DType.F32);
        float* x = (float*)feats.DataPointer;
        float* y = (float*)result.DataPointer;
        float* w = (float*)weight.DataPointer;
        float* b = (float*)bias.DataPointer;
        for (int g = 0; g < Groups; g++)
        {
            double sum = 0, squares = 0;
            for (long i = 0; i < n; i++)
            {
                for (int k = 0; k < perGroup; k++)
                {
                    double v = x[i * channels + g * perGroup + k];
                    sum += v;
                    squares += v * v;
                }
            }

            double count = (double)n * perGroup;
            double mean = sum / count;
            double variance = Math.Max(squares / count - mean * mean, 0);
            float inverse = (float)(1.0 / Math.Sqrt(variance + Eps));
            for (long i = 0; i < n; i++)
            {
                for (int k = 0; k < perGroup; k++)
                {
                    int ch = g * perGroup + k;
                    y[i * channels + ch] = (x[i * channels + ch] - (float)mean) * inverse * w[ch] + b[ch];
                }
            }
        }

        return result;
    }
}

using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Moge;

/// <summary>
/// The MoGe v1 convolutional head (upstream <c>Head</c>): the backbone's last blocks are projected by 1 x 1 convolutions and summed on the patch grid, then every upsample block
/// concatenates a two-channel view-plane UV grid (the aspect ratio's awareness), doubles the resolution with a 2 x 2 stride-2 transposed convolution and a replicate-padded 3 x 3
/// convolution, and runs residual blocks (GroupNorm, ReLU, 3 x 3 convolution, twice, with a group count of <c>hidden / 32</c> for the second norm). The result is resized to the
/// image, gets the UV grid once more and goes through one output block per output: 3 x 3 convolution, ReLU, 1 x 1 convolution, for the 3 point channels and the 1 mask channel.
/// All weights are F32, borrowed from the caller. Convolutions that upstream pads by replication pad explicitly first, as the backend's convolution pads with zeros.
/// </summary>
public sealed class MogeHead
{
    private const float NormEps = 1e-5f;

    private readonly MogeConfig _config;
    private readonly IReadOnlyDictionary<string, Tensor> _weights;
    private readonly string _prefix;
    private readonly int _backboneWidth;

    /// <param name="weights">The checkpoint's tensors by name (F32).</param>
    /// <param name="prefix">The head's prefix, <c>head.</c> in the released checkpoint.</param>
    /// <param name="backboneWidth">The width of the backbone tokens.</param>
    /// <exception cref="InvalidOperationException">A weight is missing, not F32 or of the wrong shape.</exception>
    public MogeHead(MogeConfig config, IReadOnlyDictionary<string, Tensor> weights, string prefix, int backboneWidth)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(prefix);
        _config = config;
        _weights = weights;
        _prefix = prefix;
        _backboneWidth = backboneWidth;
        Validate();
    }

    /// <summary>The widths each upsample block takes in (the UV channels included) and gives out.</summary>
    private IEnumerable<(int Index, int In, int Out)> UpsampleBlocks()
    {
        int previous = _config.DimProj;
        for (int i = 0; i < _config.DimUpsample.Length; i++)
        {
            yield return (i, previous + 2, _config.DimUpsample[i]);
            previous = _config.DimUpsample[i];
        }
    }

    private static readonly int[] OutputChannels = [3, 1];

    /// <summary>The head's tensors (every one of the checkpoint under the head prefix), for <c>IBackend.PreloadWeights</c> and <c>FreeWeights</c>.</summary>
    public IEnumerable<Tensor> EnumerateWeights() =>
        _weights.Where(entry => entry.Key.StartsWith(_prefix, StringComparison.Ordinal)).Select(entry => entry.Value);

    private Tensor Weight(string name, params long[] shape)
    {
        string key = _prefix + name;
        if (!_weights.TryGetValue(key, out Tensor? tensor))
            throw new InvalidOperationException($"MoGe weight '{key}' is missing.");
        if (tensor.DType != DType.F32)
            throw new InvalidOperationException($"MoGe weight '{key}' is {tensor.DType}; the head takes F32 weights.");
        if (!tensor.Shape.Equals(new TensorShape(shape)))
            throw new InvalidOperationException($"MoGe weight '{key}' has shape {tensor.Shape}, expected {new TensorShape(shape)}.");
        return tensor;
    }

    private void Validate()
    {
        for (int i = 0; i < _config.IntermediateLayers; i++)
        {
            Weight($"projects.{i}.weight", _config.DimProj, _backboneWidth, 1, 1);
            Weight($"projects.{i}.bias", _config.DimProj);
        }

        foreach ((int index, int inChannels, int outChannels) in UpsampleBlocks())
        {
            string block = $"upsample_blocks.{index}.";
            Weight(block + "0.0.weight", inChannels, outChannels, 2, 2);
            Weight(block + "0.0.bias", outChannels);
            Weight(block + "0.1.weight", outChannels, outChannels, 3, 3);
            Weight(block + "0.1.bias", outChannels);
            for (int r = 0; r < _config.NumResBlocks; r++)
                ValidateResBlock($"{block}{r + 1}.layers.", outChannels, _config.ResBlockHiddenMultiplier * outChannels);
        }

        int last = _config.DimUpsample[^1] + 2;
        for (int o = 0; o < OutputChannels.Length; o++)
        {
            string block = $"output_block.{o}.";
            Weight(block + "0.weight", _config.LastConvChannels, last, 3, 3);
            Weight(block + "0.bias", _config.LastConvChannels);
            Weight(block + "2.weight", OutputChannels[o], _config.LastConvChannels, _config.LastConvSize, _config.LastConvSize);
            Weight(block + "2.bias", OutputChannels[o]);
        }
    }

    private void ValidateResBlock(string prefix, int channels, int hidden)
    {
        if (hidden % 32 != 0)
            throw new InvalidOperationException($"A MoGe residual block of hidden width {hidden} has no whole group count (hidden / 32).");
        Weight(prefix + "0.weight", channels);
        Weight(prefix + "0.bias", channels);
        Weight(prefix + "2.weight", hidden, channels, 3, 3);
        Weight(prefix + "2.bias", hidden);
        Weight(prefix + "3.weight", hidden);
        Weight(prefix + "3.bias", hidden);
        Weight(prefix + "5.weight", channels, hidden, 3, 3);
        Weight(prefix + "5.bias", channels);
    }

    /// <summary>
    /// Runs the head. <paramref name="taps"/> are the backbone's last blocks after the final norm, class token dropped: F32 <c>[1, gridH · gridW, width]</c> each; the grid is
    /// <c>imageHeight / 14</c> by <c>imageWidth / 14</c> (rounded down). Returns the raw points <c>[1, 3, imageHeight, imageWidth]</c> and mask <c>[1, 1, imageHeight, imageWidth]</c>, before the remap.
    /// </summary>
    public (Tensor Points, Tensor Mask) Forward(IBackend backend, IReadOnlyList<Tensor> taps, int imageHeight, int imageWidth, int patchSize = 14)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(taps);
        if (taps.Count != _config.IntermediateLayers)
            throw new ArgumentException($"The head takes {_config.IntermediateLayers} taps; got {taps.Count}.", nameof(taps));
        int gridH = imageHeight / patchSize, gridW = imageWidth / patchSize;
        double aspect = (double)imageWidth / imageHeight;

        Tensor x = SumOfProjections(backend, taps, gridH, gridW);
        foreach ((int index, _, int outChannels) in UpsampleBlocks())
        {
            string block = $"upsample_blocks.{index}.";
            using Tensor withUv = WithUv(x, aspect);
            x.Dispose();
            int h = (int)withUv.Shape[2] * 2, w = (int)withUv.Shape[3] * 2;
            using Tensor up = new(new TensorShape(1, outChannels, h, w), DType.F32);
            backend.ConvTranspose2d(up, withUv, Weight(block + "0.0.weight", withUv.Shape[1], outChannels, 2, 2), Weight(block + "0.0.bias", outChannels), 2, 2, 0, 0);
            x = ConvReplicate(backend, up, Weight(block + "0.1.weight", outChannels, outChannels, 3, 3), Weight(block + "0.1.bias", outChannels));
            for (int r = 0; r < _config.NumResBlocks; r++)
            {
                Tensor next = ResBlock(backend, x, $"{block}{r + 1}.layers.", outChannels, _config.ResBlockHiddenMultiplier * outChannels);
                x.Dispose();
                x = next;
            }
        }

        using Tensor resized = new(new TensorShape(1, x.Shape[1], imageHeight, imageWidth), DType.F32);
        backend.InterpolateBilinear2D(resized, x, alignCorners: false);
        x.Dispose();
        using Tensor full = WithUv(resized, aspect);
        Tensor points = OutputBlock(backend, full, 0);
        try
        {
            return (points, OutputBlock(backend, full, 1));
        }
        catch
        {
            points.Dispose();
            throw;
        }
    }

    /// <summary><c>sum_i projects[i](feat_i.permute(0, 2, 1).unflatten(2, (gridH, gridW)))</c>.</summary>
    private Tensor SumOfProjections(IBackend backend, IReadOnlyList<Tensor> taps, int gridH, int gridW)
    {
        Tensor sum = new(new TensorShape(1, _config.DimProj, gridH, gridW), DType.F32);
        try
        {
            for (int i = 0; i < taps.Count; i++)
            {
                Tensor tap = taps[i];
                if (tap.DType != DType.F32 || tap.Shape.Rank != 3 || tap.Shape[0] != 1 || tap.Shape[1] != gridH * gridW || tap.Shape[2] != _backboneWidth)
                    throw new ArgumentException($"Tap {i} must be F32 [1, {gridH * gridW}, {_backboneWidth}]; got {tap.DType} {tap.Shape}.", nameof(taps));
                using Tensor channelsFirst = new(new TensorShape(_backboneWidth, gridH * gridW), DType.F32);
                using Tensor flat = tap.Reshape(new TensorShape(gridH * gridW, _backboneWidth));
                backend.Transpose2D(channelsFirst, flat, gridH * gridW, _backboneWidth);
                using Tensor grid = channelsFirst.Reshape(new TensorShape(1, _backboneWidth, gridH, gridW));
                using Tensor projected = new(new TensorShape(1, _config.DimProj, gridH, gridW), DType.F32);
                backend.Conv2D(projected, grid, Weight($"projects.{i}.weight", _config.DimProj, _backboneWidth, 1, 1), Weight($"projects.{i}.bias", _config.DimProj), 1, 1, 0, 0);
                if (i == 0)
                    projected.AsReadOnlySpan<float>().CopyTo(sum.AsSpan<float>());
                else
                    backend.Add(sum, sum, projected);
            }

            return sum;
        }
        catch
        {
            sum.Dispose();
            throw;
        }
    }

    /// <summary>One residual block: <c>GroupNorm(1), ReLU, conv3, GroupNorm(hidden / 32), ReLU, conv3</c> plus the input (the skip is the identity as the widths match).</summary>
    private Tensor ResBlock(IBackend backend, Tensor x, string prefix, int channels, int hidden)
    {
        using Tensor normed = new(x.Shape, DType.F32);
        backend.GroupNorm(normed, x, Weight(prefix + "0.weight", channels), Weight(prefix + "0.bias", channels), 1, NormEps);
        backend.LeakyRelu(normed, normed, 0f);
        using Tensor mid = ConvReplicate(backend, normed, Weight(prefix + "2.weight", hidden, channels, 3, 3), Weight(prefix + "2.bias", hidden));
        using Tensor midNormed = new(mid.Shape, DType.F32);
        backend.GroupNorm(midNormed, mid, Weight(prefix + "3.weight", hidden), Weight(prefix + "3.bias", hidden), hidden / 32, NormEps);
        backend.LeakyRelu(midNormed, midNormed, 0f);
        Tensor output = ConvReplicate(backend, midNormed, Weight(prefix + "5.weight", channels, hidden, 3, 3), Weight(prefix + "5.bias", channels));
        backend.Add(output, output, x);
        return output;
    }

    /// <summary>One output block: replicate-padded 3 x 3 convolution, ReLU, then the last convolution (replicate padding for a kernel above 1).</summary>
    private Tensor OutputBlock(IBackend backend, Tensor x, int output)
    {
        string block = $"output_block.{output}.";
        int outChannels = OutputChannels[output], k = _config.LastConvSize;
        using Tensor first = ConvReplicate(backend, x, Weight(block + "0.weight", _config.LastConvChannels, x.Shape[1], 3, 3), Weight(block + "0.bias", _config.LastConvChannels));
        backend.LeakyRelu(first, first, 0f);
        Tensor weight = Weight(block + "2.weight", outChannels, _config.LastConvChannels, k, k), bias = Weight(block + "2.bias", outChannels);
        if (k == 1)
        {
            Tensor result = new(new TensorShape(1, outChannels, first.Shape[2], first.Shape[3]), DType.F32);
            backend.Conv2D(result, first, weight, bias, 1, 1, 0, 0);
            return result;
        }

        return ConvReplicate(backend, first, weight, bias);
    }

    /// <summary>A stride-1 convolution that keeps the size and pads by replication (<c>padding_mode='replicate'</c>, padding <c>k / 2</c>).</summary>
    private static Tensor ConvReplicate(IBackend backend, Tensor x, Tensor weight, Tensor bias)
    {
        int k = (int)weight.Shape[2], pad = k / 2;
        long n = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];
        using Tensor padded = new(new TensorShape(n, c, h + 2 * pad, w + 2 * pad), DType.F32);
        backend.Pad(padded, x, new PadSpec(PadMode.Replicate, [0, 0, pad, pad], [0, 0, pad, pad]));
        Tensor output = new(new TensorShape(n, weight.Shape[0], h, w), DType.F32);
        backend.Conv2D(output, padded, weight, bias, 1, 1, 0, 0);
        return output;
    }

    /// <summary><c>cat([x, uv], dim=1)</c> with the view-plane UV grid of this resolution.</summary>
    private static Tensor WithUv(Tensor x, double aspect)
    {
        int c = (int)x.Shape[1], h = (int)x.Shape[2], w = (int)x.Shape[3];
        Tensor result = new(new TensorShape(1, c + 2, h, w), DType.F32);
        x.AsReadOnlySpan<float>().CopyTo(result.AsSpan<float>());
        Span<float> target = result.AsSpan<float>()[(c * h * w)..];
        ViewPlaneUv(target[..(h * w)], target[(h * w)..], h, w, aspect);
        return result;
    }

    /// <summary>
    /// <c>normalized_view_plane_uv</c>: u and v are linear ramps from <c>-span · (n - 1) / n</c> to <c>+span · (n - 1) / n</c> along x and y, where the spans are
    /// <c>aspect / sqrt(1 + aspect²)</c> and <c>1 / sqrt(1 + aspect²)</c> (the half-diagonal is 1).
    /// </summary>
    public static void ViewPlaneUv(Span<float> u, Span<float> v, int height, int width, double aspect)
    {
        double spanX = aspect / Math.Sqrt(1 + aspect * aspect), spanY = 1 / Math.Sqrt(1 + aspect * aspect);
        for (int y = 0; y < height; y++)
        {
            double vv = Ramp(-spanY * (height - 1) / height, spanY * (height - 1) / height, height, y);
            for (int x = 0; x < width; x++)
            {
                u[y * width + x] = (float)Ramp(-spanX * (width - 1) / width, spanX * (width - 1) / width, width, x);
                v[y * width + x] = (float)vv;
            }
        }
    }

    private static double Ramp(double start, double end, int count, int index) => count == 1 ? start : start + (end - start) * index / (count - 1);
}

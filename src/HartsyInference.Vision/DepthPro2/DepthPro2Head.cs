using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.DepthPro2;

/// <summary>
/// Depth Pro 2's dense head, from the backbone taps to focal-normalised inverse depth (upstream <c>MultiViewEncoder.reassemble</c>, <c>DPTDecoder</c> and
/// <c>create_inverse_depth_head</c>). Reassemble: each tap's patch grid goes through a 1 x 1 convolution to its pyramid width, then a 4 x 4 stride-4 or 2 x 2
/// stride-2 transposed convolution, nothing, or a 3 x 3 stride-2 convolution, for 1/4, 1/8, 1/16 and 1/32 of the input. Decoder: a 3 x 3 convolution without
/// bias to the decoder width per level, then fusion blocks from the coarsest level up, each adding a residual-refined skip, refining, doubling the size with a
/// 2 x 2 stride-2 transposed convolution (where DPT resizes bilinearly) and mixing with a 1 x 1 convolution. Head: 3 x 3 convolution to half the width, 2 x 2
/// stride-2 transposed convolution, 3 x 3 convolution to 32, ReLU, 1 x 1 convolution to 1, ReLU. Every convolution zero-pads. F32 throughout (the reference
/// runs its heads in float32, and <c>ConvTranspose2d</c> is F32-only on every backend); weights are borrowed from the caller.
/// </summary>
public sealed class DepthPro2Head
{
    private const int HeadWidth = 32;

    private readonly IReadOnlyDictionary<string, Tensor> _weights;
    private readonly int[] _dims;
    private readonly int _decoder, _backboneWidth;

    /// <param name="dimsReassemble">The four pyramid widths, highest resolution first.</param>
    /// <param name="dimDecoder">The decoder width.</param>
    /// <param name="backboneWidth">The width of the backbone tokens.</param>
    /// <param name="weights">The checkpoint's tensors by their upstream names (<c>encoder.reassemble.*</c>, <c>decoder.*</c>, <c>head.*</c>), F32.</param>
    /// <exception cref="InvalidOperationException">A weight is missing, not F32 or of the wrong shape.</exception>
    public DepthPro2Head(IReadOnlyList<int> dimsReassemble, int dimDecoder, int backboneWidth, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(dimsReassemble);
        ArgumentNullException.ThrowIfNull(weights);
        if (dimsReassemble.Count != 4)
            throw new ArgumentException($"Depth Pro 2 reassembles four levels; got {dimsReassemble.Count} widths.", nameof(dimsReassemble));
        if (dimDecoder < 2 || dimDecoder % 2 != 0 || backboneWidth <= 0)
            throw new ArgumentException($"The decoder width must be even and positive and the backbone width positive; got {dimDecoder} and {backboneWidth}.");
        _dims = [.. dimsReassemble];
        _decoder = dimDecoder;
        _backboneWidth = backboneWidth;
        _weights = weights;
        Validate();
    }

    /// <summary>The head's tensors, for <c>IBackend.PreloadWeights</c> and <c>FreeWeights</c>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (KeyValuePair<string, Tensor> entry in _weights)
            if (entry.Key.StartsWith("encoder.reassemble.", StringComparison.Ordinal) || entry.Key.StartsWith("decoder.", StringComparison.Ordinal)
                || entry.Key.StartsWith("head.", StringComparison.Ordinal))
                yield return entry.Value;
    }

    /// <summary>
    /// Inverse depth <c>[1, 1, 16·gridH, 16·gridW]</c> from the four tapped token sequences (<c>[1, prefixTokens + gridH·gridW, width]</c> each, F32).
    /// <paramref name="observe"/> sees each stage's borrowed output (<c>reassemble0..3</c>, <c>conv0..3</c>, <c>fusion3..0</c>), for parity checks.
    /// </summary>
    public Tensor Forward(IBackend backend, IReadOnlyList<Tensor> taps, int prefixTokens, int gridH, int gridW, Action<string, Tensor>? observe = null)
    {
        Tensor[] levels = Reassemble(backend, taps, prefixTokens, gridH, gridW);
        try
        {
            for (int i = 0; i < levels.Length; i++) observe?.Invoke($"reassemble{i}", levels[i]);
            using Tensor decoded = Decode(backend, levels, observe);
            return InverseDepth(backend, decoded);
        }
        finally
        {
            foreach (Tensor level in levels) level.Dispose();
        }
    }

    /// <summary>The four pyramid levels, highest resolution first; the caller owns them.</summary>
    public Tensor[] Reassemble(IBackend backend, IReadOnlyList<Tensor> taps, int prefixTokens, int gridH, int gridW)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(taps);
        if (taps.Count != 4) throw new ArgumentException($"The head takes four taps; got {taps.Count}.", nameof(taps));
        int patches = gridH * gridW;
        Tensor[] levels = new Tensor[4];
        int made = 0;
        try
        {
            for (int i = 0; i < 4; i++)
            {
                Tensor tap = taps[i];
                if (tap.DType != DType.F32 || tap.Shape.Rank != 3 || tap.Shape[0] != 1 || tap.Shape[1] != prefixTokens + patches || tap.Shape[2] != _backboneWidth)
                    throw new ArgumentException($"Tap {i} must be F32 [1, {prefixTokens + patches}, {_backboneWidth}]; got {tap.DType} {tap.Shape}.", nameof(taps));
                using Tensor rows = tap.Reshape(new TensorShape(prefixTokens + patches, _backboneWidth));
                using Tensor patchRows = rows.SliceRows(prefixTokens, patches);
                using Tensor channelsFirst = new(new TensorShape(_backboneWidth, patches), DType.F32);
                backend.Transpose2D(channelsFirst, patchRows, patches, _backboneWidth);
                using Tensor grid = channelsFirst.Reshape(new TensorShape(1, _backboneWidth, gridH, gridW));
                string p = $"encoder.reassemble.{i}.";
                Tensor projected = Conv(backend, grid, W(p + "0.weight", _dims[i], _backboneWidth, 1, 1), W(p + "0.bias", _dims[i]), 1, 0);
                levels[i] = i switch
                {
                    0 => ThenDispose(projected, ConvTranspose(backend, projected, W(p + "1.weight", _dims[0], _dims[0], 4, 4), W(p + "1.bias", _dims[0]), 4)),
                    1 => ThenDispose(projected, ConvTranspose(backend, projected, W(p + "1.weight", _dims[1], _dims[1], 2, 2), W(p + "1.bias", _dims[1]), 2)),
                    2 => projected,
                    _ => ThenDispose(projected, Conv(backend, projected, W(p + "1.weight", _dims[3], _dims[3], 3, 3), W(p + "1.bias", _dims[3]), 2, 1)),
                };
                made++;
            }

            return levels;
        }
        catch
        {
            for (int i = 0; i < made; i++) levels[i].Dispose();
            throw;
        }
    }

    /// <summary>The decoder: per-level 3 x 3 convolutions and the fusion blocks, coarsest first; the result is at half the input resolution.</summary>
    public Tensor Decode(IBackend backend, IReadOnlyList<Tensor> levels, Action<string, Tensor>? observe = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(levels);
        if (levels.Count != 4) throw new ArgumentException($"The decoder takes four levels; got {levels.Count}.", nameof(levels));
        Tensor? features = null;
        try
        {
            for (int i = 3; i >= 0; i--)
            {
                using Tensor conv = Conv(backend, levels[i], W($"decoder.convs.{i}.weight", _decoder, _dims[i], 3, 3), null, 1, 1);
                observe?.Invoke($"conv{i}", conv);
                Tensor next = Fusion(backend, i, features, conv);
                features?.Dispose();
                features = next;
                observe?.Invoke($"fusion{i}", features);
            }

            return features!;
        }
        catch
        {
            features?.Dispose();
            throw;
        }
    }

    /// <summary>The inverse-depth head on the decoder output <c>[1, dimDecoder, h, w]</c>: <c>[1, 1, 2h, 2w]</c>, zero where infinitely far.</summary>
    public Tensor InverseDepth(IBackend backend, Tensor decoded)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(decoded);
        int half = _decoder / 2;
        using Tensor first = Conv(backend, decoded, W("head.0.weight", half, _decoder, 3, 3), W("head.0.bias", half), 1, 1);
        using Tensor up = ConvTranspose(backend, first, W("head.1.weight", half, half, 2, 2), W("head.1.bias", half), 2);
        using Tensor narrow = Conv(backend, up, W("head.2.weight", HeadWidth, half, 3, 3), W("head.2.bias", HeadWidth), 1, 1);
        backend.LeakyRelu(narrow, narrow, 0f);
        Tensor output = Conv(backend, narrow, W("head.4.weight", 1, HeadWidth, 1, 1), W("head.4.bias", 1), 1, 0);
        backend.LeakyRelu(output, output, 0f);
        return output;
    }

    // FeatureFusionBlock2d: x + resnet1(skip) when there is an earlier stage, resnet2, the 2 x 2 transposed convolution (no bias), the 1 x 1 convolution.
    private Tensor Fusion(IBackend backend, int index, Tensor? x, Tensor skip)
    {
        string p = $"decoder.fusions.{index}.";
        Tensor refinedInput = skip;
        using Tensor? merged = x is null ? null : Residual(backend, skip, p + "resnet1.");
        if (merged is not null)
        {
            backend.Add(merged, merged, x!);
            refinedInput = merged;
        }

        using Tensor refined = Residual(backend, refinedInput, p + "resnet2.");
        using Tensor up = ConvTranspose(backend, refined, W(p + "deconv.weight", _decoder, _decoder, 2, 2), null, 2);
        return Conv(backend, up, W(p + "out_conv.weight", _decoder, _decoder, 1, 1), W(p + "out_conv.bias", _decoder), 1, 0);
    }

    // x + conv(relu(conv(relu(x)))), both 3 x 3 with bias.
    private Tensor Residual(IBackend backend, Tensor x, string prefix)
    {
        int d = _decoder;
        using Tensor act = new(x.Shape, DType.F32);
        backend.LeakyRelu(act, x, 0f);
        using Tensor mid = Conv(backend, act, W(prefix + "residual.1.weight", d, d, 3, 3), W(prefix + "residual.1.bias", d), 1, 1);
        backend.LeakyRelu(mid, mid, 0f);
        Tensor output = Conv(backend, mid, W(prefix + "residual.3.weight", d, d, 3, 3), W(prefix + "residual.3.bias", d), 1, 1);
        backend.Add(output, output, x);
        return output;
    }

    private static Tensor Conv(IBackend backend, Tensor x, Tensor weight, Tensor? bias, int stride, int pad)
    {
        long k = weight.Shape[2], h = (x.Shape[2] + 2 * pad - k) / stride + 1, w = (x.Shape[3] + 2 * pad - k) / stride + 1;
        Tensor output = new(new TensorShape(x.Shape[0], weight.Shape[0], h, w), DType.F32);
        backend.Conv2D(output, x, weight, bias, stride, stride, pad, pad);
        return output;
    }

    // Kernel equal to stride, no padding: each input pixel becomes one k x k output tile.
    private static Tensor ConvTranspose(IBackend backend, Tensor x, Tensor weight, Tensor? bias, int k)
    {
        Tensor output = new(new TensorShape(x.Shape[0], weight.Shape[1], x.Shape[2] * k, x.Shape[3] * k), DType.F32);
        backend.ConvTranspose2d(output, x, weight, bias, k, k, 0, 0);
        return output;
    }

    private static Tensor ThenDispose(Tensor consumed, Tensor result)
    {
        consumed.Dispose();
        return result;
    }

    private Tensor W(string key, params long[] shape)
    {
        if (!_weights.TryGetValue(key, out Tensor? tensor))
            throw new InvalidOperationException($"Depth Pro 2 weight '{key}' is missing.");
        if (tensor.DType != DType.F32)
            throw new InvalidOperationException($"Depth Pro 2 weight '{key}' is {tensor.DType}; the head takes F32 weights.");
        if (!tensor.Shape.Equals(new TensorShape(shape)))
            throw new InvalidOperationException($"Depth Pro 2 weight '{key}' has shape {tensor.Shape}, expected {new TensorShape(shape)}.");
        return tensor;
    }

    private void Validate()
    {
        int b = _backboneWidth, d = _decoder, half = _decoder / 2;
        for (int i = 0; i < 4; i++)
        {
            string p = $"encoder.reassemble.{i}.";
            W(p + "0.weight", _dims[i], b, 1, 1);
            W(p + "0.bias", _dims[i]);
            if (i != 2)
            {
                int k = i switch { 0 => 4, 1 => 2, _ => 3 };
                W(p + "1.weight", _dims[i], _dims[i], k, k);
                W(p + "1.bias", _dims[i]);
            }

            W($"decoder.convs.{i}.weight", d, _dims[i], 3, 3);
            string f = $"decoder.fusions.{i}.";
            string[] blocks = i < 3 ? ["resnet1.", "resnet2."] : ["resnet2."];
            foreach (string block in blocks)
            {
                W(f + block + "residual.1.weight", d, d, 3, 3);
                W(f + block + "residual.1.bias", d);
                W(f + block + "residual.3.weight", d, d, 3, 3);
                W(f + block + "residual.3.bias", d);
            }

            W(f + "deconv.weight", d, d, 2, 2);
            W(f + "out_conv.weight", d, d, 1, 1);
            W(f + "out_conv.bias", d);
        }

        W("head.0.weight", half, d, 3, 3);
        W("head.0.bias", half);
        W("head.1.weight", half, half, 2, 2);
        W("head.1.bias", half);
        W("head.2.weight", HeadWidth, half, 3, 3);
        W("head.2.bias", HeadWidth);
        W("head.4.weight", 1, HeadWidth, 1, 1);
        W("head.4.bias", 1);
    }
}

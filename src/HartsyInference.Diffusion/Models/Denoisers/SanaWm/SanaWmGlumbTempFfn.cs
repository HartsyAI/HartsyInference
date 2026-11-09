using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>SANA-WM GLUMBConvTemp feed-forward. Tokens <c>[B, T·H·W, C]</c> (T-major) go through a gated 2-D conv per
/// frame, then a temporal residual <c>y + Conv2d(C, C, (3, 1))(y)</c> over T, where <c>y</c> is the point-conv output.
/// Activations and weights share one dtype, F32 or BF16. The weights are borrowed and must outlive the FFN.</summary>
public sealed class SanaWmGlumbTempFfn
{
    private const int TemporalKernel = 3;
    private const int DepthwiseKernel = 3;

    private readonly int _channels;
    private readonly int _hidden;
    private readonly DType _dtype;
    private readonly Tensor _invWeightValue;
    private readonly Tensor _invWeightGate;
    private readonly Tensor _invBiasValue;
    private readonly Tensor _invBiasGate;
    private readonly Tensor _depthWeightValue;
    private readonly Tensor _depthWeightGate;
    private readonly Tensor _depthBiasValue;
    private readonly Tensor _depthBiasGate;
    private readonly Tensor _pointWeight;
    private readonly Tensor _temporalWeight;

    /// <summary>Builds the FFN from upstream-layout weights: inverted <c>[2H, C, 1, 1]</c> with bias <c>[2H]</c>, depthwise
    /// <c>[2H, 1, 3, 3]</c> with bias <c>[2H]</c>, point <c>[C, H, 1, 1]</c> (no bias), temporal <c>[C, C, 3, 1]</c> (no bias).</summary>
    /// <remarks>The inverted and depthwise halves are cut into row views, so the value and gate branches never touch a
    /// channel-split activation view.</remarks>
    public SanaWmGlumbTempFfn(Tensor invWeight, Tensor invBias, Tensor depthWeight, Tensor depthBias, Tensor pointWeight, Tensor temporalWeight)
    {
        ArgumentNullException.ThrowIfNull(invWeight);
        ArgumentNullException.ThrowIfNull(invBias);
        ArgumentNullException.ThrowIfNull(depthWeight);
        ArgumentNullException.ThrowIfNull(depthBias);
        ArgumentNullException.ThrowIfNull(pointWeight);
        ArgumentNullException.ThrowIfNull(temporalWeight);

        if (invWeight.Shape.Rank != 4 || invWeight.Shape[2] != 1 || invWeight.Shape[3] != 1 || invWeight.Shape[0] % 2 != 0)
            throw new ArgumentException($"SANA-WM FFN inverted weight must be [2H, C, 1, 1]; got {invWeight.Shape}.", nameof(invWeight));
        _channels = (int)invWeight.Shape[1];
        _hidden = (int)invWeight.Shape[0] / 2;
        _dtype = invWeight.DType;

        RequireShape(invBias, [2L * _hidden], nameof(invBias));
        RequireShape(depthWeight, [2L * _hidden, 1, DepthwiseKernel, DepthwiseKernel], nameof(depthWeight));
        RequireShape(depthBias, [2L * _hidden], nameof(depthBias));
        RequireShape(pointWeight, [_channels, _hidden, 1, 1], nameof(pointWeight));
        RequireShape(temporalWeight, [_channels, _channels, TemporalKernel, 1], nameof(temporalWeight));
        RequireDtype(invBias, nameof(invBias));
        RequireDtype(depthWeight, nameof(depthWeight));
        RequireDtype(depthBias, nameof(depthBias));
        RequireDtype(pointWeight, nameof(pointWeight));
        RequireDtype(temporalWeight, nameof(temporalWeight));
        if (_dtype != DType.F32 && _dtype != DType.BF16)
            throw new ArgumentException($"SANA-WM FFN weights must be F32 or BF16; got {_dtype.Name}.", nameof(invWeight));

        int hidden = _hidden;
        _invWeightValue = invWeight.SliceRows(0, hidden);
        _invWeightGate = invWeight.SliceRows(hidden, hidden);
        _invBiasValue = invBias.SliceRows(0, hidden);
        _invBiasGate = invBias.SliceRows(hidden, hidden);
        _depthWeightValue = depthWeight.SliceRows(0, hidden);
        _depthWeightGate = depthWeight.SliceRows(hidden, hidden);
        _depthBiasValue = depthBias.SliceRows(0, hidden);
        _depthBiasGate = depthBias.SliceRows(hidden, hidden);
        _pointWeight = pointWeight;
        _temporalWeight = temporalWeight;
    }

    /// <summary>Runs the FFN on tokens <c>[B, frames·height·width, C]</c>; returns a new owned <c>[B, N, C]</c> tensor in the input dtype.</summary>
    /// <exception cref="NotSupportedException">BF16 input on a backend without BF16 support.</exception>
    public Tensor Forward(IBackend backend, Tensor x, int frames, int height, int width)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(x);
        ValidateInput(backend, x, frames, height, width);

        int batch = (int)x.Shape[0];
        int rows = batch * frames;
        int area = height * width;
        Tensor output = new(new TensorShape(batch, (long)frames * area, _channels), _dtype);
        bool completed = false;
        try
        {
            // No Reshape views: CUDA binds a result to the tensor object the op was given, not to a view of it.
            using Tensor channelFirst = new(new TensorShape(rows, _channels, height, width), _dtype);
            backend.Transpose2D(channelFirst, x, area, _channels);

            using Tensor point = SpatialPath(backend, channelFirst, rows, height, width);
            using Tensor frameMajor = TemporalPath(backend, point, batch, frames, area);

            backend.Transpose2D(output, frameMajor, _channels, area);
            completed = true;
            return output;
        }
        finally
        {
            if (!completed) output.Dispose();
        }
    }

    /// <summary>Gated per-frame conv: point-conv output <c>[rows, C, H, W]</c> of <c>value · SiLU(gate)</c>.</summary>
    private Tensor SpatialPath(IBackend backend, Tensor channelFirst, int rows, int height, int width)
    {
        TensorShape hiddenShape = new(rows, _hidden, height, width);
        using Tensor value = GatedBranch(backend, channelFirst, _invWeightValue, _invBiasValue, _depthWeightValue, _depthBiasValue, hiddenShape);
        using Tensor gate = GatedBranch(backend, channelFirst, _invWeightGate, _invBiasGate, _depthWeightGate, _depthBiasGate, hiddenShape);
        backend.Silu(gate, gate);
        backend.Mul(value, value, gate);

        Tensor point = new(new TensorShape(rows, _channels, height, width), _dtype);
        backend.Conv2D(point, value, _pointWeight, null, 1, 1, 0, 0);
        return point;
    }

    /// <summary>One half of the inverted and depthwise stage: <c>depthwise(SiLU(conv1x1(x)))</c>, no activation after the depthwise.
    /// The inverted buffer is released here, so only one half of it is alive at a time.</summary>
    private Tensor GatedBranch(IBackend backend, Tensor channelFirst, Tensor invWeight, Tensor invBias, Tensor depthWeight, Tensor depthBias, TensorShape shape)
    {
        using Tensor inverted = new(shape, _dtype);
        backend.Conv2D(inverted, channelFirst, invWeight, invBias, 1, 1, 0, 0);
        backend.Silu(inverted, inverted);

        // BF16 depthwise output on a BF16 input (P2): the largest intermediate stays at 2 bytes per element.
        Tensor depth = new(shape, _dtype);
        backend.Conv2dDepthwise(depth, inverted, depthWeight, depthBias, 1, 1, DepthwiseKernel / 2, DepthwiseKernel / 2);
        return depth;
    }

    /// <summary>Temporal residual over T for each (H, W) position: <c>[B·T, C, H, W]</c> point output in, <c>[B, T, C, HW]</c> out.</summary>
    private Tensor TemporalPath(IBackend backend, Tensor point, int batch, int frames, int area)
    {
        using Tensor timeMajor = new(new TensorShape(batch, _channels, frames, area), _dtype);
        backend.Permute0213(timeMajor, point, frames, _channels, area);

        using Tensor residual = new(new TensorShape(batch, _channels, frames, area), _dtype);
        backend.Conv2D(residual, timeMajor, _temporalWeight, null, 1, 1, TemporalKernel / 2, 0);
        backend.Add(residual, residual, timeMajor);

        Tensor frameMajor = new(new TensorShape(batch, frames, _channels, area), _dtype);
        backend.Permute0213(frameMajor, residual, _channels, frames, area);
        return frameMajor;
    }

    private void ValidateInput(IBackend backend, Tensor x, int frames, int height, int width)
    {
        if (x.Shape.Rank != 3)
            throw new ArgumentException($"SANA-WM FFN input must be [B, N, C]; got {x.Shape}.", nameof(x));
        if (frames <= 0 || height <= 0 || width <= 0)
            throw new ArgumentOutOfRangeException(nameof(frames), "SANA-WM FFN grid extents must be positive.");
        if (x.Shape[1] != (long)frames * height * width)
            throw new ArgumentException($"SANA-WM FFN token count {x.Shape[1]} does not equal T·H·W = {frames}·{height}·{width}.", nameof(x));
        if (x.Shape[2] != _channels)
            throw new ArgumentException($"SANA-WM FFN channel count {x.Shape[2]} does not match the weights ({_channels}).", nameof(x));
        if (x.DType != _dtype)
            throw new ArgumentException($"SANA-WM FFN input dtype {x.DType.Name} does not match the weights ({_dtype.Name}).", nameof(x));
        if (_dtype == DType.BF16 && !backend.Capabilities.SupportsBF16)
            throw new NotSupportedException($"SANA-WM FFN BF16 activations need a BF16-capable backend; {backend.Capabilities.Name} is not.");

        long largestIntermediate = (long)x.Shape[0] * frames * _hidden * height * width;
        if (largestIntermediate > int.MaxValue)
            throw new NotSupportedException($"SANA-WM FFN intermediate of {largestIntermediate} elements exceeds the 32-bit conv index range.");
    }

    private static void RequireShape(Tensor tensor, ReadOnlySpan<long> expected, string name)
    {
        if (tensor.Shape.Rank != expected.Length)
            throw new ArgumentException($"SANA-WM FFN {name} must have rank {expected.Length}; got {tensor.Shape}.", name);
        for (int i = 0; i < expected.Length; i++)
        {
            if (tensor.Shape[i] != expected[i])
                throw new ArgumentException($"SANA-WM FFN {name} has shape {tensor.Shape}; expected dimension {i} = {expected[i]}.", name);
        }
    }

    private void RequireDtype(Tensor tensor, string name)
    {
        if (tensor.DType != _dtype)
            throw new ArgumentException($"SANA-WM FFN {name} dtype {tensor.DType.Name} does not match the inverted weight ({_dtype.Name}).", name);
    }
}

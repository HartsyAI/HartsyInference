using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>SANA-WM GLUMBConvTemp feed-forward. Tokens <c>[B, T·H·W, C]</c> (T-major) go through a gated 2-D conv per
/// frame, then a temporal residual <c>y + Conv2d(C, C, (3, 1))(y)</c> over T, where <c>y</c> is the point-conv output.
/// Activations and conv kernels share one dtype, F32 or BF16; biases are that dtype or F32. The weights are borrowed and must
/// outlive the FFN.</summary>
/// <remarks>The gated conv is per frame, so it runs over chunks of frames: the hidden buffers, the largest in the block, never
/// hold more than the chunk budget.</remarks>
public sealed class SanaWmGlumbTempFfn
{
    /// <summary>Default budget of one hidden buffer, in elements: 64 MiB of BF16.</summary>
    public const long DefaultHiddenChunkElements = 1L << 25;

    private const int TemporalKernel = 3;
    private const int DepthwiseKernel = 3;

    private readonly int _channels;
    private readonly int _hidden;
    private readonly DType _dtype;
    private readonly long _hiddenChunkElements;
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
        : this(invWeight, invBias, depthWeight, depthBias, pointWeight, temporalWeight, DefaultHiddenChunkElements)
    {
    }

    /// <param name="hiddenChunkElements">Largest hidden buffer, in elements; a chunk holds at least one frame whatever the budget.</param>
    internal SanaWmGlumbTempFfn(Tensor invWeight, Tensor invBias, Tensor depthWeight, Tensor depthBias, Tensor pointWeight,
        Tensor temporalWeight, long hiddenChunkElements)
    {
        ArgumentNullException.ThrowIfNull(invWeight);
        ArgumentNullException.ThrowIfNull(invBias);
        ArgumentNullException.ThrowIfNull(depthWeight);
        ArgumentNullException.ThrowIfNull(depthBias);
        ArgumentNullException.ThrowIfNull(pointWeight);
        ArgumentNullException.ThrowIfNull(temporalWeight);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenChunkElements);

        if (invWeight.Shape.Rank != 4 || invWeight.Shape[2] != 1 || invWeight.Shape[3] != 1 || invWeight.Shape[0] % 2 != 0)
            throw new ArgumentException($"SANA-WM FFN inverted weight must be [2H, C, 1, 1]; got {invWeight.Shape}.", nameof(invWeight));
        _channels = (int)invWeight.Shape[1];
        _hidden = (int)invWeight.Shape[0] / 2;
        _dtype = invWeight.DType;
        _hiddenChunkElements = hiddenChunkElements;

        RequireShape(invBias, [2L * _hidden], nameof(invBias));
        RequireShape(depthWeight, [2L * _hidden, 1, DepthwiseKernel, DepthwiseKernel], nameof(depthWeight));
        RequireShape(depthBias, [2L * _hidden], nameof(depthBias));
        RequireShape(pointWeight, [_channels, _hidden, 1, 1], nameof(pointWeight));
        RequireShape(temporalWeight, [_channels, _channels, TemporalKernel, 1], nameof(temporalWeight));
        RequireBiasDtype(invBias, nameof(invBias));
        RequireDtype(depthWeight, nameof(depthWeight));
        RequireBiasDtype(depthBias, nameof(depthBias));
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

    /// <summary>The activation and conv-kernel dtype.</summary>
    public DType DType => _dtype;

    /// <summary>The tensors <see cref="Forward"/> reads (the half views of the inverted and depthwise weights, not their parents),
    /// for preloading on a device backend.</summary>
    public IEnumerable<Tensor> EnumerateWeights() =>
        [_invWeightValue, _invWeightGate, _invBiasValue, _invBiasGate, _depthWeightValue, _depthWeightGate, _depthBiasValue,
            _depthBiasGate, _pointWeight, _temporalWeight];

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
        using SanaWmScratch scratch = new();
        Tensor channelFirst = scratch.New(new TensorShape(rows, _channels, height, width), _dtype);
        backend.Transpose2D(channelFirst, x, area, _channels);
        Tensor point = scratch.Track(SpatialPath(backend, channelFirst, rows, height, width));
        scratch.Release(channelFirst);
        Tensor frameMajor = scratch.Track(TemporalPath(backend, point, batch, frames, area));
        scratch.Release(point);
        Tensor output = scratch.New(new TensorShape(batch, (long)frames * area, _channels), _dtype);
        backend.Transpose2D(output, frameMajor, _channels, area);
        return scratch.Detach(output);
    }

    /// <summary>Frames per chunk of the gated conv: as many as fit the hidden budget, at least one.</summary>
    internal int ChunkFrames(int rows, int area) => (int)Math.Clamp(_hiddenChunkElements / ((long)_hidden * area), 1, rows);

    /// <summary>Gated per-frame conv over chunks of frames: point-conv output <c>[rows, C, H, W]</c> of <c>value · SiLU(gate)</c>.</summary>
    private Tensor SpatialPath(IBackend backend, Tensor channelFirst, int rows, int height, int width)
    {
        int area = height * width;
        int chunk = ChunkFrames(rows, area);
        if (chunk >= rows)
            return GatedPoint(backend, channelFirst, rows, height, width);

        long frameElements = (long)_channels * area;
        using SanaWmScratch scratch = new();
        Tensor point = scratch.New(new TensorShape(rows, _channels, height, width), _dtype);
        for (int start = 0; start < rows; start += chunk)
        {
            int count = Math.Min(chunk, rows - start);
            using SanaWmScratch step = new();
            Tensor input = step.Track(SanaWmTensors.Slice(backend, channelFirst, start * frameElements,
                new TensorShape(count, _channels, height, width)));
            Tensor output = step.Track(GatedPoint(backend, input, count, height, width));
            SanaWmTensors.Write(backend, point, output, start * frameElements);
        }

        return scratch.Detach(point);
    }

    /// <summary><c>point(value · SiLU(gate))</c> for <paramref name="rows"/> frames <c>[rows, C, H, W]</c>.</summary>
    private Tensor GatedPoint(IBackend backend, Tensor channelFirst, int rows, int height, int width)
    {
        TensorShape hiddenShape = new(rows, _hidden, height, width);
        using Tensor value = GatedBranch(backend, channelFirst, _invWeightValue, _invBiasValue, _depthWeightValue, _depthBiasValue, hiddenShape);
        using Tensor gate = GatedBranch(backend, channelFirst, _invWeightGate, _invBiasGate, _depthWeightGate, _depthBiasGate, hiddenShape);
        backend.Silu(gate, gate);
        backend.Mul(value, value, gate);

        Tensor point = new(new TensorShape(rows, _channels, height, width), _dtype);
        try
        {
            backend.Conv2D(point, value, _pointWeight, null, 1, 1, 0, 0);
            return point;
        }
        catch
        {
            point.Dispose();
            throw;
        }
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
        try
        {
            backend.Conv2dDepthwise(depth, inverted, depthWeight, depthBias, 1, 1, DepthwiseKernel / 2, DepthwiseKernel / 2);
            return depth;
        }
        catch
        {
            depth.Dispose();
            throw;
        }
    }

    /// <summary>Temporal residual over T for each (H, W) position: <c>[B·T, C, H, W]</c> point output in, <c>[B, T, C, HW]</c> out.</summary>
    private Tensor TemporalPath(IBackend backend, Tensor point, int batch, int frames, int area)
    {
        using SanaWmScratch scratch = new();
        Tensor timeMajor = scratch.New(new TensorShape(batch, _channels, frames, area), _dtype);
        backend.Permute0213(timeMajor, point, frames, _channels, area);

        Tensor residual = scratch.New(new TensorShape(batch, _channels, frames, area), _dtype);
        backend.Conv2D(residual, timeMajor, _temporalWeight, null, 1, 1, TemporalKernel / 2, 0);
        backend.Add(residual, residual, timeMajor);
        scratch.Release(timeMajor);

        Tensor frameMajor = scratch.New(new TensorShape(batch, frames, _channels, area), _dtype);
        backend.Permute0213(frameMajor, residual, _channels, frames, area);
        return scratch.Detach(frameMajor);
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

        int area = height * width;
        long largestIntermediate = (long)ChunkFrames((int)x.Shape[0] * frames, area) * _hidden * area;
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

    private void RequireBiasDtype(Tensor tensor, string name)
    {
        if (tensor.DType != _dtype && tensor.DType != DType.F32)
            throw new ArgumentException($"SANA-WM FFN {name} dtype {tensor.DType.Name} is neither F32 nor the weight dtype ({_dtype.Name}).", name);
    }
}

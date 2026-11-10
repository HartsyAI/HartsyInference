using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Streaming;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>SANA-WM GLUMBConvTemp feed-forward. Tokens <c>[B, T·H·W, C]</c> (T-major) go through a gated 2-D conv per
/// frame, then a temporal residual <c>y + Conv2d(C, C, (3, 1))(y)</c> over T, where <c>y</c> is the point-conv output.
/// Activations and conv kernels share one dtype, F32 or BF16; biases are that dtype or F32. The weights are borrowed and must
/// outlive the FFN. The two tensors the FFN builds itself, the tap-major depthwise kernel and a cast depthwise bias, are its own,
/// and <see cref="Dispose"/> releases them. In a chunk-causal stream (upstream <c>CachedGLUMBConvTemp</c>) the temporal convolution
/// reads the last frame of the previous chunk on its left and zeros on its right.</summary>
/// <remarks>The gated conv runs on the tokens as they are (channels-last, as upstream's <c>permute</c> view feeds cuDNN): the 1×1
/// convolutions are GEMMs over the token rows and the depthwise convolution and GLU are one fused channels-last op, so no
/// activation is transposed. It is per frame, so it runs over chunks of frames: the hidden buffers, the largest in the block,
/// never hold more than the chunk budget.</remarks>
public sealed class SanaWmGlumbTempFfn : IDisposable
{
    /// <summary>Default budget of one hidden buffer, in elements: 64 MiB of BF16.</summary>
    public const long DefaultHiddenChunkElements = 1L << 25;

    private const int TemporalKernel = 3;
    private const int DepthwiseKernel = 3;

    private readonly int _channels;
    private readonly int _hidden;
    private readonly DType _dtype;
    private readonly long _hiddenChunkElements;
    private readonly Tensor _invWeight;
    private readonly Tensor _invBias;
    private readonly Tensor _depthTaps;
    private readonly Tensor _depthBias;
    private readonly Tensor _pointWeight;
    private readonly Tensor _temporalWeight;
    private readonly List<Tensor> _owned = [];
    private int _disposed;

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

        // The 1×1 kernels as GEMM weights; views over the borrowed tensors, so no copy. The depthwise bias is F32 for the fused op.
        _invWeight = invWeight.Reshape(new TensorShape(2L * _hidden, _channels));
        _invBias = invBias;
        Tensor taps = TapMajor(depthWeight, 2 * _hidden);
        Tensor bias;
        try
        {
            bias = depthBias.DType == DType.F32 ? depthBias : depthBias.CastTo(DType.F32);
        }
        catch
        {
            taps.Dispose();
            throw;
        }

        _depthTaps = taps;
        _owned.Add(taps);
        _depthBias = bias;
        if (!ReferenceEquals(bias, depthBias))
            _owned.Add(bias);
        _pointWeight = pointWeight.Reshape(new TensorShape(_channels, _hidden));
        _temporalWeight = temporalWeight;
    }

    /// <summary>The activation and conv-kernel dtype.</summary>
    public DType DType => _dtype;

    /// <summary>The tensors <see cref="Forward"/> reads (the GEMM views of the 1×1 kernels, not their parents), for preloading on a
    /// device backend.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => [_invWeight, _invBias, _depthTaps, _depthBias, _pointWeight, _temporalWeight];

    /// <summary>Releases the tap-major depthwise kernel and the cast depthwise bias the FFN built. The weights it was given are
    /// not touched, and a device copy of any tensor <see cref="EnumerateWeights"/> lists must be freed first. Safe to call twice.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        SanaWmTensors.DisposeAll(_owned);
    }

    /// <summary>Runs the FFN on tokens <c>[B, frames·height·width, C]</c>; returns a new owned <c>[B, N, C]</c> tensor in the input dtype.</summary>
    /// <exception cref="NotSupportedException">BF16 input on a backend without BF16 support.</exception>
    public Tensor Forward(IBackend backend, Tensor x, int frames, int height, int width, string profileTag = "ffn") =>
        Forward(backend, x, frames, height, width, profileTag, null, commit: false);

    /// <summary><see cref="Forward(IBackend, Tensor, int, int, int, string)"/> for one chunk of a chunk-causal stream of batch 1.</summary>
    /// <param name="tail">The point-convolution output of the frame before the chunk, which the temporal convolution reads as
    /// its left context; null runs the clip form, zero-padded on both sides.</param>
    /// <param name="commit">Stages this chunk's last frame into <paramref name="tail"/>.</param>
    internal Tensor Forward(IBackend backend, Tensor x, int frames, int height, int width, string profileTag, FrameTailCache? tail,
        bool commit)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(x);
        ValidateInput(backend, x, frames, height, width);
        if (tail is not null && x.Shape[0] != 1)
            throw new ArgumentException($"A chunk-causal stream runs one batch item per forward; got batch {x.Shape[0]}.", nameof(x));

        int batch = (int)x.Shape[0];
        int rows = batch * frames;
        int area = height * width;
        using SanaWmScratch scratch = new();
        Tensor point;
        using (StageProfiler.Begin(backend, profileTag, "mlp.spatial"))
            point = scratch.Track(SpatialPath(backend, x, rows, height, width));
        Tensor output;
        using (StageProfiler.Begin(backend, profileTag, "mlp.temporal"))
        {
            output = tail?.Current is Tensor previous
                ? scratch.Track(TemporalPathWithContext(backend, point, previous, tail.HeldFrames, frames, area))
                : scratch.Track(TemporalPath(backend, point, batch, frames, area));
            if (commit)
                tail?.Stage(point);
        }

        scratch.Release(point);
        return scratch.Detach(output);
    }

    /// <summary>Frames per chunk of the gated conv: as many as fit the hidden budget, at least one.</summary>
    internal int ChunkFrames(int rows, int area) => (int)Math.Clamp(_hiddenChunkElements / ((long)_hidden * area), 1, rows);

    /// <summary>Gated per-frame conv over chunks of frames on the token rows: point-conv output <c>[rows·H·W, C]</c> of
    /// <c>value · SiLU(gate)</c>.</summary>
    private Tensor SpatialPath(IBackend backend, Tensor tokens, int rows, int height, int width)
    {
        int area = height * width;
        int chunk = ChunkFrames(rows, area);
        if (chunk >= rows)
            return GatedPoint(backend, tokens, rows, height, width);

        long frameElements = (long)_channels * area;
        using SanaWmScratch scratch = new();
        Tensor point = scratch.New(new TensorShape((long)rows * area, _channels), _dtype);
        for (int start = 0; start < rows; start += chunk)
        {
            int count = Math.Min(chunk, rows - start);
            using SanaWmScratch step = new();
            Tensor input = step.Track(SanaWmTensors.Slice(backend, tokens, start * frameElements, new TensorShape((long)count * area, _channels)));
            Tensor output = step.Track(GatedPoint(backend, input, count, height, width));
            SanaWmTensors.Write(backend, point, output, start * frameElements);
        }

        return scratch.Detach(point);
    }

    /// <summary><c>point(value · SiLU(gate))</c> for <paramref name="rows"/> frames of token rows <c>[rows·H·W, C]</c>: the inverted
    /// GEMM and its SiLU, the fused channels-last depthwise conv and GLU, then the point GEMM.</summary>
    private Tensor GatedPoint(IBackend backend, Tensor tokens, int rows, int height, int width)
    {
        long pixels = (long)rows * height * width;
        Tensor gated;
        using (Tensor inverted = new(new TensorShape(pixels, 2L * _hidden), _dtype))
        {
            // The inverted conv's SiLU is applied as the fused depthwise op reads its output.
            backend.Linear(inverted, tokens, _invWeight, _invBias);
            gated = new Tensor(new TensorShape(pixels, _hidden), _dtype);
            try
            {
                backend.DepthwiseConvGlu(gated, inverted, _depthTaps, _depthBias, rows, height, width, DepthwiseKernel, inputSilu: true);
            }
            catch
            {
                gated.Dispose();
                throw;
            }
        }

        using (gated)
        {
            Tensor point = new(new TensorShape(pixels, _channels), _dtype);
            try
            {
                backend.Linear(point, gated, _pointWeight, null);
                return point;
            }
            catch
            {
                point.Dispose();
                throw;
            }
        }
    }

    /// <summary>Temporal residual over T for each (H, W) position: token rows <c>[B·T·H·W, C]</c> in, <c>[B, T·H·W, C]</c> out.
    /// The conv reads <c>[B, C, T, HW]</c>, one transpose of each batch item's rows.</summary>
    private Tensor TemporalPath(IBackend backend, Tensor point, int batch, int frames, int area)
    {
        int tokens = frames * area;
        using SanaWmScratch scratch = new();
        Tensor timeMajor = scratch.New(new TensorShape(batch, _channels, frames, area), _dtype);
        backend.Transpose2D(timeMajor, point, tokens, _channels);

        Tensor residual = scratch.New(new TensorShape(batch, _channels, frames, area), _dtype);
        backend.Conv2D(residual, timeMajor, _temporalWeight, null, 1, 1, TemporalKernel / 2, 0);
        backend.Add(residual, residual, timeMajor);
        scratch.Release(timeMajor);

        Tensor output = scratch.New(new TensorShape(batch, tokens, _channels), _dtype);
        backend.Transpose2D(output, residual, _channels, tokens);
        return scratch.Detach(output);
    }

    /// <summary><see cref="TemporalPath"/> for a chunk that follows <paramref name="contextFrames"/> earlier frames: the
    /// convolution runs over <c>[context ⊕ chunk]</c> and its output is read at the chunk's frames, so the first frame sees the
    /// previous chunk's last frame and the last frame sees zeros. Token rows <c>[T·H·W, C]</c> in, <c>[1, T·H·W, C]</c> out.</summary>
    private Tensor TemporalPathWithContext(IBackend backend, Tensor point, Tensor context, int contextFrames, int frames, int area)
    {
        int tokens = frames * area;
        int joinedFrames = contextFrames + frames;
        int joinedTokens = joinedFrames * area;
        using SanaWmScratch scratch = new();
        Tensor joined = scratch.Track(StreamingTensors.Concat(backend, [context, point], 0));
        Tensor timeMajor = scratch.New(new TensorShape(1, _channels, joinedFrames, area), _dtype);
        backend.Transpose2D(timeMajor, joined, joinedTokens, _channels);
        scratch.Release(joined);

        Tensor convolved = scratch.New(new TensorShape(1, _channels, joinedFrames, area), _dtype);
        backend.Conv2D(convolved, timeMajor, _temporalWeight, null, 1, 1, TemporalKernel / 2, 0);
        scratch.Release(timeMajor);
        Tensor convolvedRows = scratch.New(new TensorShape(joinedTokens, _channels), _dtype);
        backend.Transpose2D(convolvedRows, convolved, _channels, joinedTokens);
        scratch.Release(convolved);

        Tensor residual = scratch.Track(SanaWmTensors.Slice(backend, convolvedRows, (long)contextFrames * area * _channels,
            new TensorShape(1, tokens, _channels)));
        scratch.Release(convolvedRows);
        Tensor output = scratch.New(new TensorShape(1, tokens, _channels), _dtype);
        backend.Add(output, residual, point);
        return scratch.Detach(output);
    }

    /// <summary>The depthwise kernel <c>[channels, 1, k, k]</c> (F32 or BF16) as F32 tap-major taps <c>[k, k, channels]</c>.</summary>
    private static Tensor TapMajor(Tensor depthWeight, int channels)
    {
        const int area = DepthwiseKernel * DepthwiseKernel;
        Tensor f32 = depthWeight.DType == DType.F32 ? depthWeight : depthWeight.CastTo(DType.F32);
        try
        {
            Tensor taps = new(new TensorShape(DepthwiseKernel, DepthwiseKernel, channels), DType.F32);
            ReadOnlySpan<float> source = f32.AsReadOnlySpan<float>();
            Span<float> target = taps.AsSpan<float>();
            for (int c = 0; c < channels; c++)
            {
                for (int t = 0; t < area; t++)
                    target[t * channels + c] = source[c * area + t];
            }

            return taps;
        }
        finally
        {
            if (!ReferenceEquals(f32, depthWeight))
                f32.Dispose();
        }
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

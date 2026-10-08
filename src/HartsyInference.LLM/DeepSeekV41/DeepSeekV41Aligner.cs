using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Host reference for the V4.1 aligner: zero-pads the tower's patch grid to a multiple of the downsample ratio, folds each ratio x ratio block of features into one row, and maps it to the language model's width through two projections with an exact GELU between.</summary>
/// <remarks>Follows upstream <c>Aligner.forward</c> in float32. The padding is added after the tower's final norm, so padded cells are exact zeros. A row holds its block channel-major, then block row, then block column
/// (<c>F.unfold</c>'s order); rows run over the block grid in row-major order. The aligner owns the weight tensors it is given and disposes them.</remarks>
public sealed class DeepSeekV41Aligner : IDisposable
{
    private readonly IBackend _backend;
    private readonly DeepSeekV41AlignerWeights _weights;
    private readonly int _dim;
    private readonly int _ratio;
    private bool _disposed;

    /// <param name="backend">Provides the linear and GELU ops.</param>
    /// <param name="config">Vision tower dimensions (feature width and downsample ratio).</param>
    /// <param name="outputDim">The language model's hidden width, which the aligner emits.</param>
    /// <param name="weights">F32 weights shaped by <paramref name="config"/> and <paramref name="outputDim"/>; the aligner takes ownership.</param>
    /// <exception cref="HartsyInferenceException">The config is not runnable or a weight is missing, not F32 or mis-shaped.</exception>
    public DeepSeekV41Aligner(IBackend backend, DeepSeekV41VisionConfig config, int outputDim, DeepSeekV41AlignerWeights weights)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentOutOfRangeException.ThrowIfLessThan(outputDim, 1);
        config.Validate();
        weights.Validate(config, outputDim);
        _backend = backend;
        _weights = weights;
        _dim = config.HiddenSize;
        _ratio = config.DownsampleRatio;
        OutputDim = outputDim;
    }

    /// <summary>Width of each emitted row.</summary>
    public int OutputDim { get; }

    /// <summary>Diagnostic tap called with a stage name and a copy of its values: <c>unfold</c> (the folded rows) and <c>hidden</c> (after the GELU). Null (the default) copies nothing.</summary>
    public Action<string, float[]>? Probe { get; set; }

    /// <summary>The language-side token grid an <paramref name="gridHeight"/> x <paramref name="gridWidth"/> patch grid folds into: each side divided by <paramref name="ratio"/>, rounded up.</summary>
    public static (int Height, int Width) TokenGrid(int gridHeight, int gridWidth, int ratio)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(gridHeight, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(gridWidth, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(ratio, 1);
        return ((gridHeight + ratio - 1) / ratio, (gridWidth + ratio - 1) / ratio);
    }

    /// <summary>Aligns a tower output into language-model embeddings, <c>[tokenHeight * tokenWidth, OutputDim]</c> row-major.</summary>
    /// <param name="features">The tower's normed features, <c>[gridHeight * gridWidth, dim]</c>.</param>
    /// <param name="gridHeight">Patch rows.</param>
    /// <param name="gridWidth">Patch columns.</param>
    public float[] Forward(ReadOnlySpan<float> features, int gridHeight, int gridWidth)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        (int tokenHeight, int tokenWidth) = TokenGrid(gridHeight, gridWidth, _ratio);
        if (features.Length != (long)gridHeight * gridWidth * _dim)
            throw new ArgumentException($"Expected {gridHeight} x {gridWidth} features of {_dim} values, got {features.Length}.", nameof(features));
        int tokens = tokenHeight * tokenWidth, rowWidth = _dim * _ratio * _ratio;

        using Tensor folded = new(new TensorShape(tokens, rowWidth), DType.F32);
        Unfold(features, gridHeight, gridWidth, tokenHeight, tokenWidth, folded.AsSpan<float>());
        Emit("unfold", folded);
        using Tensor projected = new(new TensorShape(tokens, OutputDim), DType.F32);
        _backend.Linear(projected, folded, _weights.W1, _weights.B1);
        using Tensor hidden = new(new TensorShape(tokens, OutputDim), DType.F32);
        _backend.GeluErf(hidden, projected);
        Emit("hidden", hidden);
        using Tensor output = new(new TensorShape(tokens, OutputDim), DType.F32);
        _backend.Linear(output, hidden, _weights.W2, _weights.B2);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _weights.Dispose();
    }

    // No backend op gathers from a token-major grid with zero padding folded in, so this is a plain loop.
    private void Unfold(ReadOnlySpan<float> features, int gridHeight, int gridWidth, int tokenHeight, int tokenWidth, Span<float> folded)
    {
        int ratio = _ratio, dim = _dim, block = ratio * ratio, rowWidth = dim * block;
        for (int bh = 0; bh < tokenHeight; bh++)
            for (int bw = 0; bw < tokenWidth; bw++)
            {
                Span<float> row = folded.Slice((bh * tokenWidth + bw) * rowWidth, rowWidth);
                for (int ky = 0; ky < ratio; ky++)
                    for (int kx = 0; kx < ratio; kx++)
                    {
                        int y = bh * ratio + ky, x = bw * ratio + kx, slot = ky * ratio + kx;
                        if (y >= gridHeight || x >= gridWidth)
                        {
                            for (int c = 0; c < dim; c++) row[c * block + slot] = 0f;
                            continue;
                        }
                        ReadOnlySpan<float> cell = features.Slice((y * gridWidth + x) * dim, dim);
                        for (int c = 0; c < dim; c++) row[c * block + slot] = cell[c];
                    }
            }
    }

    private void Emit(string stage, Tensor values) => Probe?.Invoke(stage, values.AsReadOnlySpan<float>().ToArray());
}

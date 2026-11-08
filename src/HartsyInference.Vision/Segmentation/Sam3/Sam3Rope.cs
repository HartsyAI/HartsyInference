using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Axial 2D RoPE tables for the SAM 3 ViT, built once per configuration.</summary>
/// <remarks>Windowed blocks rotate by the position inside a window, so RoPE runs on full-grid tokens before
/// windows are gathered. Global blocks rotate by grid position scaled by window/grid. Pairs are interleaved
/// as (x[2j], x[2j+1]); the first half of the pairs rotate by x and the second half by y.</remarks>
public sealed class Sam3Rope : IDisposable
{
    private readonly int _tokens;
    private readonly int _headDim;
    private readonly Tensor _windowCos;
    private readonly Tensor _windowSin;
    private readonly Tensor _globalCos;
    private readonly Tensor _globalSin;

    /// <summary>Builds the windowed and global cos/sin tables for a square token grid.</summary>
    /// <param name="grid">Token grid side.</param>
    /// <param name="window">Window side, also the RoPE pretraining size.</param>
    /// <param name="headDim">Per-head width; a positive multiple of 4.</param>
    /// <param name="theta">Frequency base.</param>
    public Sam3Rope(int grid, int window, int headDim, float theta)
    {
        if (grid <= 0)
            throw new ArgumentOutOfRangeException(nameof(grid), grid, "Grid side must be positive.");
        if (window <= 0 || window > grid)
            throw new ArgumentOutOfRangeException(nameof(window), window, "Window must be in [1, grid].");
        if (headDim <= 0 || headDim % 4 != 0)
            throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "Head dim must be a positive multiple of 4.");
        if (theta <= 0f || float.IsNaN(theta))
            throw new ArgumentOutOfRangeException(nameof(theta), theta, "RoPE theta must be positive.");

        _tokens = grid * grid;
        _headDim = headDim;
        _windowCos = new Tensor(new TensorShape(_tokens, headDim), DType.F32);
        _windowSin = new Tensor(new TensorShape(_tokens, headDim), DType.F32);
        _globalCos = new Tensor(new TensorShape(_tokens, headDim), DType.F32);
        _globalSin = new Tensor(new TensorShape(_tokens, headDim), DType.F32);

        Fill(_windowCos, _windowSin, grid, window, 1f, headDim, theta);
        Fill(_globalCos, _globalSin, grid, grid, (float)window / grid, headDim, theta);
    }

    /// <summary>Rotates <paramref name="query"/> and <paramref name="key"/> in place.</summary>
    /// <param name="backend">Backend that runs the rotation.</param>
    /// <param name="query">Token-major F32 <c>[1, tokens, heads, headDim]</c>.</param>
    /// <param name="key">Token-major F32 <c>[1, tokens, heads, headDim]</c>.</param>
    /// <param name="globalAttention">True for global blocks, false for windowed blocks.</param>
    public void Apply(IBackend backend, Tensor query, Tensor key, bool globalAttention)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ValidateTokenMajor(query, nameof(query));
        ValidateTokenMajor(key, nameof(key));

        Tensor cos = globalAttention ? _globalCos : _windowCos;
        Tensor sin = globalAttention ? _globalSin : _windowSin;
        backend.ApplyRopeInterleaved(query, cos, sin);
        backend.ApplyRopeInterleaved(key, cos, sin);
    }

    /// <summary>Releases the cos/sin tables.</summary>
    public void Dispose()
    {
        _windowCos.Dispose();
        _windowSin.Dispose();
        _globalCos.Dispose();
        _globalSin.Dispose();
    }

    private void ValidateTokenMajor(Tensor tensor, string name)
    {
        if (tensor.DType != DType.F32 || tensor.Shape.Rank != 4 || tensor.Shape[0] != 1
            || tensor.Shape[1] != _tokens || tensor.Shape[3] != _headDim)
        {
            throw new ArgumentException(
                $"RoPE expects F32 token-major [1,{_tokens},heads,{_headDim}] for {name}; got {tensor.DType} {tensor.Shape}.", name);
        }
    }

    // Frequencies follow base^(-4k/headDim): the first quarter of pairs use x, the rest use y.
    private static void Fill(Tensor cos, Tensor sin, int grid, int modulus, float scale, int headDim, float theta)
    {
        int half = headDim / 2;
        int quarter = headDim / 4;
        Span<float> frequencies = stackalloc float[quarter];
        for (int k = 0; k < quarter; k++)
        {
            frequencies[k] = (float)(1.0 / Math.Pow(theta, 4.0 * k / headDim));
        }

        Span<float> cosTable = cos.AsSpan<float>();
        Span<float> sinTable = sin.AsSpan<float>();
        int tokens = grid * grid;
        for (int token = 0; token < tokens; token++)
        {
            float tx = (token % grid % modulus) * scale;
            float ty = (token / grid % modulus) * scale;
            int rowStart = token * headDim;
            for (int j = 0; j < half; j++)
            {
                float position = j < quarter ? tx : ty;
                float angle = position * frequencies[j % quarter];
                cosTable[rowStart + j] = MathF.Cos(angle);
                sinTable[rowStart + j] = MathF.Sin(angle);
            }

            // The rotation loop reads only the first half; identity keeps any extra lanes inert.
            for (int j = half; j < headDim; j++)
            {
                cosTable[rowStart + j] = 1f;
                sinTable[rowStart + j] = 0f;
            }
        }
    }
}

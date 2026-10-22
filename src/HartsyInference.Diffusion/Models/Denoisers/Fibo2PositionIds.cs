using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Fibo 2.0 RoPE position ids, F32 <c>[tokens, 3]</c> over the axes (frame, row, col). Gist token <c>i</c> (1-based)
/// is <c>(i, 0, 0)</c>; the image sits on frame <c>MaxGist + 1</c> in row-major order; reference image <c>k</c> sits on
/// frame <c>MaxGist + 2 + k</c>.</summary>
public static unsafe class Fibo2PositionIds
{
    /// <summary>Gist ids <c>(i, 0, 0)</c>, <c>i</c> in <c>1..count</c>.</summary>
    public static Tensor Gist(Fibo2Config config, int count)
    {
        ArgumentNullException.ThrowIfNull(config);
        ValidateGistCount(config, count);
        Tensor ids = new(new TensorShape(count, 3), DType.F32);
        WriteGist(ids, 0, count);
        return ids;
    }

    /// <summary>Image ids on frame <c>MaxGist + 1</c>, row-major over the <c>gridH × gridW</c> token grid.</summary>
    public static Tensor Image(Fibo2Config config, int gridH, int gridW)
    {
        ArgumentNullException.ThrowIfNull(config);
        Tensor ids = new(new TensorShape(ValidateGrid(gridH, gridW), 3), DType.F32);
        WriteGrid(ids, 0, config.MaxGist + 1, gridH, gridW);
        return ids;
    }

    /// <summary>Reference image <paramref name="index"/> ids on frame <c>MaxGist + 2 + index</c>.</summary>
    public static Tensor Reference(Fibo2Config config, int index, int gridH, int gridW)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (index < 0)
            throw new ArgumentOutOfRangeException(nameof(index), "Reference index cannot be negative.");
        Tensor ids = new(new TensorShape(ValidateGrid(gridH, gridW), 3), DType.F32);
        WriteGrid(ids, 0, config.MaxGist + 2 + index, gridH, gridW);
        return ids;
    }

    /// <summary>Stream ids in token order: the gist tokens, then the image tokens.</summary>
    public static Tensor Combined(Fibo2Config config, int gistCount, int gridH, int gridW)
    {
        ArgumentNullException.ThrowIfNull(config);
        ValidateGistCount(config, gistCount);
        int imageLen = ValidateGrid(gridH, gridW);
        Tensor ids = new(new TensorShape(gistCount + imageLen, 3), DType.F32);
        WriteGist(ids, 0, gistCount);
        WriteGrid(ids, gistCount, config.MaxGist + 1, gridH, gridW);
        return ids;
    }

    private static void WriteGist(Tensor ids, int firstRow, int count)
    {
        float* p = (float*)ids.DataPointer;
        for (int i = 0; i < count; i++)
        {
            int row = (firstRow + i) * 3;
            p[row] = i + 1;
            p[row + 1] = 0f;
            p[row + 2] = 0f;
        }
    }

    private static void WriteGrid(Tensor ids, int firstRow, int frame, int gridH, int gridW)
    {
        float* p = (float*)ids.DataPointer;
        for (int y = 0; y < gridH; y++)
        {
            for (int x = 0; x < gridW; x++)
            {
                int row = (firstRow + y * gridW + x) * 3;
                p[row] = frame;
                p[row + 1] = y;
                p[row + 2] = x;
            }
        }
    }

    private static int ValidateGrid(int gridH, int gridW)
    {
        if (gridH <= 0 || gridW <= 0)
            throw new ArgumentOutOfRangeException(nameof(gridH), $"Token grid {gridH}x{gridW} must be positive.");
        return gridH * gridW;
    }

    private static void ValidateGistCount(Fibo2Config config, int count)
    {
        if (count < 1 || count > config.MaxGist)
            throw new ArgumentOutOfRangeException(nameof(count), $"Gist count {count} is outside 1..{config.MaxGist}.");
    }
}

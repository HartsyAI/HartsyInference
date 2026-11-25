using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer.Dsa;

/// <summary>A layer with no indexer: every query attends its last <c>window</c> cached rows, itself included (Hy4 Flash sliding layers).</summary>
public sealed class DsaWindowSelector : IDsaKeySelector
{
    private readonly int _window;

    /// <param name="window">Rows attended, the query own row included.</param>
    public DsaWindowSelector(int window)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(window, 1);
        _window = window;
    }

    /// <inheritdoc />
    public int KeyWidth => 0;

    /// <inheritdoc />
    public void Append(ReadOnlySpan<float> x, int tokens, int startPos, Tensor keyStore)
    {
    }

    /// <inheritdoc />
    public DsaKeySelection Select(ReadOnlySpan<float> x, ReadOnlySpan<float> qLatent, ReadOnlySpan<int> positions, int length, Tensor keyStore)
    {
        int tokens = positions.Length;
        int[] rows = new int[tokens * _window], counts = new int[tokens];
        Array.Fill(rows, -1);
        for (int t = 0; t < tokens; t++)
        {
            int first = Math.Max(positions[t] + 1 - _window, 0), count = positions[t] + 1 - first;
            for (int i = 0; i < count; i++) rows[t * _window + i] = first + i;
            counts[t] = count;
        }
        return new DsaKeySelection(rows, _window, counts);
    }

    /// <inheritdoc />
    public void Dispose()
    {
    }
}

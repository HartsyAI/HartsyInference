using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer.Sparse;

namespace HartsyInference.LLM.Transformer.Dsa;

/// <summary>The ERNIE-5.5 and Kimi selection: a lightning-indexer top-k over the keys before each query trailing window, then that window
/// (vLLM-ERNIE <c>merge_swa_topk</c>). Owns its <see cref="DsaIndexer"/>.</summary>
public sealed class DsaIndexerSelector : IDsaKeySelector
{
    private readonly IBackend _backend;
    private readonly DsaIndexer _indexer;
    private readonly int _window;

    /// <param name="backend">Runs the window merge.</param>
    /// <param name="indexer">The indexer; disposed with this selector.</param>
    /// <param name="window">Newest keys every query attends besides the indexer picks; zero means none.</param>
    public DsaIndexerSelector(IBackend backend, DsaIndexer indexer, int window)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(indexer);
        ArgumentOutOfRangeException.ThrowIfNegative(window);
        _backend = backend;
        _indexer = indexer;
        _window = window;
    }

    /// <inheritdoc />
    public int KeyWidth => _indexer.Settings.HeadDim;

    /// <inheritdoc />
    public void Append(ReadOnlySpan<float> x, int tokens, int startPos, Tensor keyStore) => _indexer.AppendKeys(x, tokens, startPos, keyStore);

    /// <inheritdoc />
    public DsaKeySelection Select(ReadOnlySpan<float> x, ReadOnlySpan<float> qLatent, ReadOnlySpan<int> positions, int length, Tensor keyStore)
    {
        int tokens = positions.Length;
        int[] limits = new int[tokens];
        for (int t = 0; t < tokens; t++) limits[t] = SparseWindowMerge.IndexerLimit(positions[t], _window);
        int[] topk = _indexer.Select(x, qLatent, positions, limits, keyStore, out int topkCols);
        int[] rows = MergeWindow(topk, topkCols, positions, out int rowCols, out int rowWidth);
        int[] counts = new int[tokens];
        for (int t = 0; t < tokens; t++)
        {
            int seq = positions[t] + 1, picked = Math.Min(Math.Max(seq - _window, 0), rowCols);
            counts[t] = picked + Math.Min(seq, _window);
            // A NaN indexer score ranks below the masked tail and can leave a -1 among the picks; name it here, not in GatherRows.
            for (int i = 0; i < picked; i++)
            {
                if ((uint)rows[t * rowWidth + i] >= (uint)(seq - _window))
                    throw new InvalidOperationException($"DSA top-k entry {rows[t * rowWidth + i]} for position {positions[t]} is "
                        + $"outside [0, {seq - _window}); the indexer scores held a NaN.");
            }
        }
        return new DsaKeySelection(rows, rowWidth, counts);
    }

    // Every query row [topk | window | -1 ...] through IBackend.MergeWindowTopK. With no indexer columns yet (no query is past its
    // window) a one-column placeholder stands in; no query takes from it, since each takes min(max(seq - window, 0), cols) = 0.
    private int[] MergeWindow(int[] topk, int topkCols, ReadOnlySpan<int> positions, out int cols, out int width)
    {
        int tokens = positions.Length;
        cols = Math.Max(topkCols, 1);
        width = cols + _window;
        int[] source = topk;
        if (topkCols == 0)
        {
            source = new int[tokens];
            Array.Fill(source, -1);
        }
        using Tensor topkT = HostMath.Tensor(source, tokens, cols);
        using Tensor positionT = HostMath.Tensor(positions, tokens);
        using Tensor rows = new(new TensorShape(tokens, width), DType.I32);
        _backend.MergeWindowTopK(rows, topkT, positionT, _window);
        return rows.AsReadOnlySpan<int>().ToArray();
    }

    /// <inheritdoc />
    public void Dispose() => _indexer.Dispose();
}

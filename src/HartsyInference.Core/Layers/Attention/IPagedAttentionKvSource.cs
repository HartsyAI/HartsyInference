using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Layers.Attention;

/// <summary>A key/value store that keeps its rows in pages and can also present them contiguously. An attention call over
/// it appends to the pages, then reads them through the page table where the backend has a kernel for that
/// (<see cref="IBackend.AdmitsPagedAttention"/>) and through the contiguous buffers everywhere else.</summary>
public interface IPagedAttentionKvSource : IAttentionKvSource
{
    /// <summary>Appends head-major K/V <c>[1, kvHeads, tNew, D]</c> to <paramref name="layer"/>'s pages only. The contiguous
    /// buffers fall behind until <see cref="AppendKvContiguous"/> or a buffer read brings them up to date.</summary>
    void AppendKvPages(IBackend backend, int layer, Tensor k, Tensor v);

    /// <summary>Adds the rows just passed to <see cref="AppendKvPages"/> to <paramref name="layer"/>'s contiguous buffers,
    /// straight from <paramref name="k"/> and <paramref name="v"/>: the cheap way to keep the buffers current for a call
    /// that is about to read them.</summary>
    void AppendKvContiguous(IBackend backend, int layer, Tensor k, Tensor v);

    /// <summary>The page buffers and page table that hold <paramref name="layer"/>'s rows, for one call; false when the
    /// layer has written nothing. The tensors are the source's and are valid until its next append, rollback or reset.</summary>
    bool TryGetPages(int layer, out Tensor keyPages, out Tensor valuePages, out AttentionKvPages pages);

    /// <summary>A call just read <paramref name="layer"/> through its pages. The source may release that layer's contiguous
    /// buffers; a later buffer read rebuilds them from the pages.</summary>
    void PagesServed(int layer);
}

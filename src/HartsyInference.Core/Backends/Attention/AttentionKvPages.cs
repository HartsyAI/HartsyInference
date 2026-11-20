using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>Keys and values reached through a page table (docs/ATTENTION_CONTRACT.md §2): the call's key and value operands
/// are one layer's page buffers, <c>[1, kvHeads, pages × PageSize, D]</c>, and row <c>r</c> of the sequence is buffer row
/// <c>Table[r / PageSize] × PageSize + r mod PageSize</c>.</summary>
/// <param name="Table">I32, one page index per <paramref name="PageSize"/> rows of the sequence, in order. It covers at least
/// the call's valid key length. The owner replaces the tensor when the table changes, so a reader never sees it edited in
/// place, and a backend may keep its device copy between calls.</param>
/// <param name="PageSize">Rows per page.</param>
/// <param name="HighestPage">The largest page index in <paramref name="Table"/>, stated by the owner so a call is checked
/// against the page buffers without reading the table back from a device.</param>
public readonly record struct AttentionKvPages(Tensor Table, int PageSize, int HighestPage)
{
    /// <summary>Pages the table lists.</summary>
    public int Pages => (int)Table.ElementCount;
}

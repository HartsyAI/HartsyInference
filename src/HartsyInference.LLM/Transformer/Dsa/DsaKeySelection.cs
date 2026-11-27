namespace HartsyInference.LLM.Transformer.Dsa;

/// <summary>The cache rows each query of a chunk attends: <c>Rows</c> is <c>[tokens, Width]</c> cache slots, ascending, and only the first <c>Counts[t]</c> entries of row <c>t</c> are valid.</summary>
/// <param name="Rows">Slots, row-major.</param>
/// <param name="Width">Entries per row.</param>
/// <param name="Counts">Valid entries per query.</param>
public sealed record DsaKeySelection(int[] Rows, int Width, int[] Counts);

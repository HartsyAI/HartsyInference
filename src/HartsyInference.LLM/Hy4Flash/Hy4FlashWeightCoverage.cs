using System.Globalization;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>What a checkpoint's tensors map to: the key-level coverage shared with the other families, and the shape, scale and dtype checks Hy4 adds on top.</summary>
/// <param name="Keys">Mapped, scale-companion, skipped and unmapped keys and missing parameters. The speculative drafter's tensors are counted as skipped MTP keys.</param>
/// <param name="ShapeMismatches">Tensors whose shape differs from the config's.</param>
/// <param name="ScaleMismatches">FP8 weights without a correctly shaped scale grid, or a grid beside a weight that is not FP8.</param>
/// <param name="DtypeMismatches">Tensors whose storage dtype is not one the layout allows (F32 for the hyper-connections, sinks, indexer pooling and router bias; I32 for the Engram token map).</param>
public sealed record Hy4FlashWeightCoverage(CheckpointKeyCoverage Keys, IReadOnlyList<string> ShapeMismatches, IReadOnlyList<string> ScaleMismatches,
    IReadOnlyList<string> DtypeMismatches)
{
    /// <summary>Fails when any key feeds nothing, any parameter has no key, or any shape, scale grid or dtype disagrees, naming the first few of each. Call this rather than <c>Keys.RequireComplete</c>, which checks only the keys.</summary>
    /// <exception cref="InvalidDataException">The checkpoint and the config do not match.</exception>
    public void RequireComplete()
    {
        if (Keys.Unmapped.Count == 0 && Keys.Missing.Count == 0 && ShapeMismatches.Count == 0 && ScaleMismatches.Count == 0 && DtypeMismatches.Count == 0) return;
        throw new InvalidDataException(string.Create(CultureInfo.InvariantCulture,
            $"The Hy4 Flash checkpoint does not match the config: {Keys.Unmapped.Count} unmapped key(s) [{string.Join(", ", Keys.Unmapped.Take(6))}], "
            + $"{Keys.Missing.Count} missing [{string.Join(", ", Keys.Missing.Take(6))}], {ShapeMismatches.Count} shape mismatch(es) "
            + $"[{string.Join("; ", ShapeMismatches.Take(4))}], {ScaleMismatches.Count} scale mismatch(es) [{string.Join("; ", ScaleMismatches.Take(4))}], "
            + $"{DtypeMismatches.Count} dtype mismatch(es) [{string.Join("; ", DtypeMismatches.Take(4))}]."));
    }
}

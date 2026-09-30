using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Quant;

/// <summary>Pairs quantized weights with their scale companions from a header inventory alone, before any data is mapped, and refuses anything unpaired or ambiguous.</summary>
/// <remarks>Every problem found is reported in one exception, so a bad checkpoint is fixed in one pass rather than one key at a time.</remarks>
public static class QuantCompanionBinder
{
    private const int MlxGroupSize = 64;
    private const int MaxListedProblems = 12;

    private static readonly BlockGeometry[] Fp8Geometries = [new(32, 32), new(128, 128), new(1, 32), new(1, 16)];
    private static readonly BlockGeometry[] Fp4Geometries = [new(1, 32), new(1, 16)];
    private static readonly BlockGeometry[] Nvfp4Geometries = [new(1, 16)];
    private static readonly (int Bits, QuantEncoding Encoding)[] MlxWidths = [(4, QuantEncoding.AffineInt4), (8, QuantEncoding.AffineInt8)];
    private static readonly string[] GenericSuffixes = [".scale", ".scales", ".biases"];

    /// <summary>Binds every block-scaled weight in <paramref name="inventory"/> for the given producer's naming.</summary>
    /// <exception cref="HartsyInferenceException">Any weight lacks a companion, a companion has no weight, or geometry/dtype does not fit.</exception>
    public static QuantBindingSet Bind(IReadOnlyDictionary<string, TensorLocation> inventory, QuantFlavor flavor)
    {
        ArgumentNullException.ThrowIfNull(inventory);
        BindingRun run = new(inventory);
        foreach (string key in inventory.Keys.Order(StringComparer.Ordinal))
        {
            switch (flavor)
            {
                case QuantFlavor.Official:
                case QuantFlavor.AmdQuark:
                    BindScaledWeight(run, key, flavor);
                    break;
                case QuantFlavor.NvidiaNvfp4:
                    BindNvfp4(run, key);
                    break;
                case QuantFlavor.Mlx:
                    BindMlx(run, key);
                    break;
                case QuantFlavor.Exl3:
                    BindExl3(run, key);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(flavor), flavor, null);
            }
        }
        FindOrphans(run, CompanionSuffixes(flavor));
        if (run.Problems.Count > 0)
        {
            string listed = string.Join("; ", run.Problems.Take(MaxListedProblems));
            string more = run.Problems.Count > MaxListedProblems ? $"; and {run.Problems.Count - MaxListedProblems} more" : "";
            throw new HartsyInferenceException($"{flavor} quant companions do not bind ({run.Problems.Count} problems): {listed}{more}.");
        }
        return new QuantBindingSet(run.Bindings, run.PerTensorFp8);
    }

    private static string[] CompanionSuffixes(QuantFlavor flavor) => flavor switch
    {
        QuantFlavor.Official => [".scale", ".weight_scale_inv"],
        QuantFlavor.AmdQuark => [".weight_scale"],
        QuantFlavor.NvidiaNvfp4 => [".scale", ".weight_scale", ".weight_scale_2", ".input_scale"],
        QuantFlavor.Mlx => [".scales", ".biases"],
        _ => [".suh", ".svh", ".mcg"],
    };

    private static void BindScaledWeight(BindingRun run, string key, QuantFlavor flavor)
    {
        if (!key.EndsWith(".weight", StringComparison.Ordinal)) return;
        TensorLocation weight = run.Inventory[key];
        bool isFp8 = weight.DType == DType.F8E4M3;
        bool isFp4 = weight.DType == DType.I8 || (flavor == QuantFlavor.AmdQuark && weight.DType == DType.U8);
        if (!isFp8 && !isFp4) return;

        string baseName = key[..^".weight".Length];
        string[] suffixes = flavor == QuantFlavor.Official ? [".scale", ".weight_scale_inv"] : [".weight_scale"];
        List<string> present = suffixes.Select(s => baseName + s).Where(run.Inventory.ContainsKey).ToList();
        if (present.Count != 1)
        {
            run.Problems.Add(present.Count == 0
                ? $"'{key}' ({weight.DType.Name}) has no scale companion ({string.Join(" / ", suffixes.Select(s => baseName + s))})"
                : $"'{key}' has {present.Count} scale companions ({string.Join(", ", present)}); which one applies is ambiguous");
            return;
        }

        TensorLocation scale = run.Inventory[present[0]];
        run.Claimed.Add(present[0]);
        // Quark's FP8 attention scales are F8_E8M0; only its packed FP4 weights carry raw U8 E8M0 bytes.
        if (flavor == QuantFlavor.AmdQuark && isFp4 && scale.DType != DType.U8)
        {
            run.Problems.Add($"'{present[0]}' is {scale.DType.Name}; Quark stores E8M0 scales as raw U8");
            return;
        }
        BindBlockScaled(run, key, weight, present[0], scale, isFp4 ? QuantEncoding.Mxfp4E8M0 : QuantEncoding.Fp8E4M3BlockE8M0,
            isFp4 ? Fp4Geometries : Fp8Geometries);
    }

    private static void BindBlockScaled(BindingRun run, string key, TensorLocation weight, string scaleKey,
        TensorLocation scale, QuantEncoding encoding, BlockGeometry[] candidates)
    {
        if (scale.DType != DType.F8E8M0 && scale.DType != DType.U8)
        {
            run.Problems.Add($"'{scaleKey}' is {scale.DType.Name}; the E8M0 codecs decode only F8_E8M0 or raw U8 scales");
            return;
        }
        if (!TryLogicalShape(run, key, weight, encoding, out long rows, out long cols)) return;
        if (!TryInferGeometry(run, scaleKey, scale, rows, cols, candidates, out BlockGeometry geometry)) return;
        run.Bindings[key] = new QuantBinding(key, encoding, geometry, scale.DType, rows, cols, scaleKey);
    }

    private static void BindNvfp4(BindingRun run, string key)
    {
        if (!key.EndsWith(".weight", StringComparison.Ordinal)) return;
        TensorLocation weight = run.Inventory[key];
        string baseName = key[..^".weight".Length];
        string scaleKey = baseName + ".weight_scale", scale2Key = baseName + ".weight_scale_2", inputKey = baseName + ".input_scale";
        run.Inventory.TryGetValue(scaleKey, out TensorLocation? scale);
        run.Inventory.TryGetValue(scale2Key, out TensorLocation? scale2);
        bool hasInput = run.Inventory.TryGetValue(inputKey, out TensorLocation? input);

        // A scalar '.weight_scale' marks a per-tensor FP8 weight; without one the FP8 weight is official-layout.
        if (weight.DType == DType.F8E4M3 && scale is not null)
        {
            if (scale.Shape.ElementCount != 1)
            {
                run.Problems.Add($"fp8 weight '{key}' has a non-scalar scale {scale.Shape}; per-tensor only");
            }
            else
            {
                run.Claimed.Add(scaleKey);
                if (hasInput) run.Claimed.Add(inputKey);
                if (scale2 is not null) run.Claimed.Add(scale2Key);
                run.PerTensorFp8.Add(key);
            }
            return;
        }
        // The NVFP4 checkpoint keeps everything but the routed experts in the official layout.
        if (weight.DType != DType.U8)
        {
            BindScaledWeight(run, key, QuantFlavor.Official);
            return;
        }

        if (scale is null)
        {
            run.Problems.Add($"NVFP4 weight '{key}' has no '{scaleKey}'");
            return;
        }
        run.Claimed.Add(scaleKey);
        if (scale2 is null)
        {
            run.Problems.Add($"NVFP4 weight '{key}' has no '{scale2Key}'");
            return;
        }
        run.Claimed.Add(scale2Key);
        if (hasInput) run.Claimed.Add(inputKey);
        if (scale.DType != DType.F8E4M3)
        {
            run.Problems.Add($"'{scaleKey}' is {scale.DType.Name}; NVFP4 block scales are F8_E4M3");
            return;
        }
        if (scale2.DType != DType.F32 || scale2.Shape.ElementCount != 1)
        {
            run.Problems.Add($"'{scale2Key}' is {scale2.DType.Name} {scale2.Shape}; NVFP4 needs a scalar F32");
            return;
        }
        if (!TryLogicalShape(run, key, weight, QuantEncoding.Nvfp4, out long rows, out long cols)) return;
        if (!TryInferGeometry(run, scaleKey, scale, rows, cols, Nvfp4Geometries, out BlockGeometry geometry)) return;
        run.Bindings[key] = new QuantBinding(key, QuantEncoding.Nvfp4, geometry, scale.DType, rows, cols, scaleKey, scale2Key,
            hasInput ? inputKey : null);
    }

    private static void BindMlx(BindingRun run, string key)
    {
        if (!key.EndsWith(".weight", StringComparison.Ordinal) || run.Inventory[key].DType != DType.U32) return;
        TensorLocation weight = run.Inventory[key];
        string baseName = key[..^".weight".Length];
        string scalesKey = baseName + ".scales", biasesKey = baseName + ".biases";
        bool hasScales = run.Inventory.TryGetValue(scalesKey, out TensorLocation? scales);
        bool hasBiases = run.Inventory.TryGetValue(biasesKey, out TensorLocation? biases);
        if (hasScales) run.Claimed.Add(scalesKey);
        if (hasBiases) run.Claimed.Add(biasesKey);
        if (!hasScales || !hasBiases)
        {
            run.Problems.Add($"MLX weight '{key}' is missing {(hasScales ? biasesKey : scalesKey)}");
            return;
        }
        if (weight.Shape.Rank != 2 || scales!.Shape.Rank != 2 || biases!.Shape != scales.Shape || biases.DType != scales.DType)
        {
            run.Problems.Add(
                $"MLX '{key}' {weight.Shape} needs rank-2 scales and biases of one shape and dtype; got {scales!.Shape} and {biases!.Shape}");
            return;
        }
        if (scales.DType != DType.F32)
        {
            run.Problems.Add($"MLX '{scalesKey}' is {scales.DType.Name}; the affine codecs read F32 scales and biases");
            return;
        }
        // The conversion is mixed precision with no per-tensor bits in config.json, so read the width off the shapes.
        long rows = weight.Shape[0], packedWidth = weight.Shape[1];
        foreach ((int bits, QuantEncoding encoding) in MlxWidths)
        {
            long cols = packedWidth * 32 / bits;
            if (scales.Shape[0] != rows || scales.Shape[1] * MlxGroupSize != cols) continue;
            run.Bindings[key] = new QuantBinding(key, encoding, new BlockGeometry(1, MlxGroupSize), scales.DType, rows, cols, scalesKey,
                BiasKey: biasesKey);
            return;
        }
        run.Problems.Add(
            $"MLX '{key}' packed width {packedWidth} with scales {scales.Shape} is not group size {MlxGroupSize} at 4 bits "
            + $"(in={packedWidth * 8}) or 8 bits (in={packedWidth * 4})");
    }

    private static void BindExl3(BindingRun run, string key)
    {
        if (!key.EndsWith(".trellis", StringComparison.Ordinal)) return;
        TensorLocation trellis = run.Inventory[key];
        string baseName = key[..^".trellis".Length];
        string[] names = [baseName + ".suh", baseName + ".svh", baseName + ".mcg"];
        List<string> missing = names.Where(n => !run.Inventory.ContainsKey(n)).ToList();
        foreach (string n in names) run.Claimed.Add(n);
        if (missing.Count > 0)
        {
            run.Problems.Add($"EXL3 weight '{key}' is missing {string.Join(", ", missing)}");
            return;
        }
        if (trellis.Shape.Rank != 3 || trellis.Shape[2] % 16 != 0)
        {
            run.Problems.Add($"EXL3 '{key}' has trellis shape {trellis.Shape}; expected [in/16, out/16, 16*bits]");
            return;
        }
        // Unverified against real EXL3 files: rows/cols are taken from the svh/suh element counts.
        long rows = run.Inventory[names[1]].Shape.ElementCount, cols = run.Inventory[names[0]].Shape.ElementCount;
        run.Bindings[key] = new QuantBinding(key, QuantEncoding.Exl3Trellis, new BlockGeometry(16, 16), trellis.DType, rows, cols, null,
            Exl3: new Exl3Keys(names[0], names[1], names[2], (int)(trellis.Shape[2] / 16)));
    }

    private static bool TryLogicalShape(BindingRun run, string key, TensorLocation weight, QuantEncoding encoding, out long rows, out long cols)
    {
        rows = cols = 0;
        if (weight.Shape.Rank != 2)
        {
            run.Problems.Add($"'{key}' ({encoding}) has shape {weight.Shape}; block-scaled weights are rank 2");
            return false;
        }
        rows = weight.Shape[0];
        cols = weight.Shape[1] * (encoding is QuantEncoding.Mxfp4E8M0 or QuantEncoding.Nvfp4 ? 2 : 1);
        return true;
    }

    private static bool TryInferGeometry(BindingRun run, string scaleKey, TensorLocation scale, long rows, long cols,
        BlockGeometry[] candidates, out BlockGeometry geometry)
    {
        geometry = default;
        if (scale.Shape.Rank != 2)
        {
            run.Problems.Add($"'{scaleKey}' has shape {scale.Shape}; block scales are rank 2");
            return false;
        }
        List<BlockGeometry> fits = candidates.Where(g => g.ScaleShape(rows, cols) == (scale.Shape[0], scale.Shape[1])).ToList();
        if (fits.Count >= 1 && fits.All(g => SamePartition(g, fits[0], rows, cols)))
        {
            geometry = fits[0];
            return true;
        }
        string tried = string.Join(", ", candidates.Select(g => $"{g} -> {g.ScaleShape(rows, cols)}"));
        run.Problems.Add(fits.Count == 0
            ? $"'{scaleKey}' {scale.Shape} matches no geometry for a {rows}x{cols} weight ({tried})"
            : $"'{scaleKey}' {scale.Shape} fits {string.Join(" and ", fits)} for a {rows}x{cols} weight; geometry is ambiguous");
        return false;
    }

    // Geometries that clamp to the same block on this matrix tile it identically, so the first candidate is as good as any.
    private static bool SamePartition(BlockGeometry a, BlockGeometry b, long rows, long cols) =>
        Math.Min(a.BlockRows, rows) == Math.Min(b.BlockRows, rows) && Math.Min(a.BlockCols, cols) == Math.Min(b.BlockCols, cols);

    private static bool HasSiblingWeight(BindingRun run, string key) =>
        run.Inventory.ContainsKey(key[..key.LastIndexOf('.')] + ".weight");

    private static void FindOrphans(BindingRun run, string[] suffixes)
    {
        foreach (string key in run.Inventory.Keys.Order(StringComparer.Ordinal))
        {
            if (run.Claimed.Contains(key) || !suffixes.Any(s => key.EndsWith(s, StringComparison.Ordinal))) continue;
            // Generic names (.scale, .scales, .biases) also belong to norms and unquantized layers; flag them only beside a weight.
            if (GenericSuffixes.Any(s => key.EndsWith(s, StringComparison.Ordinal)) && !HasSiblingWeight(run, key)) continue;
            run.Problems.Add($"companion '{key}' has no quantized weight to pair with");
        }
    }

    private sealed class BindingRun(IReadOnlyDictionary<string, TensorLocation> inventory)
    {
        public IReadOnlyDictionary<string, TensorLocation> Inventory { get; } = inventory;
        public Dictionary<string, QuantBinding> Bindings { get; } = new(StringComparer.Ordinal);
        public List<string> PerTensorFp8 { get; } = new();
        public HashSet<string> Claimed { get; } = new(StringComparer.Ordinal);
        public List<string> Problems { get; } = new();
    }
}

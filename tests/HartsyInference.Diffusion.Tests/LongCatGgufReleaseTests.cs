using System.Runtime.InteropServices;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A LongCat GGUF repack against the BF16 release it came from, block 0 and every shared tensor: each part decodes to the release's rows
/// within its quant type's bound, each unquantized tensor is the release's value, and the whole file loads. Runs on the CPU and only when
/// HARTSY_LONGCAT_GGUF names a repack and HARTSY_LONGCAT_BF16_INDEX an index whose shards hold block 0 and the shared tensors.</summary>
public sealed class LongCatGgufReleaseTests(ITestOutputHelper output)
{
    /// <summary>Relative RMS bound per type, and the per-element bound: for the block types a multiple of the block's stored scale (plus, for the
    /// offset types, its offset's F16 rounding); for the k-quants a share of the largest reference magnitude in the 256-value block. Q8_0 is the
    /// exception: its levels are rounded with the F32 scale and decoded with the F16 one, so its bound is half a step plus 127.5 half-spacings of
    /// the stored F16 scale. That is about 0.56 steps for a normal scale and more than a step for the subnormal scales of small blocks.</summary>
    private static readonly Dictionary<string, (double RelRms, double Element)> Bounds = new(StringComparer.Ordinal)
    {
        ["Q8_0"] = (0.02, 0.5), ["Q4_0"] = (0.2, 1.01), ["Q5_0"] = (0.1, 1.01), ["Q4_1"] = (0.15, 0.51), ["Q5_1"] = (0.08, 0.52),
        ["Q2_K"] = (0.5, 0.75), ["Q3_K"] = (0.25, 0.5), ["Q4_K"] = (0.15, 0.25), ["Q5_K"] = (0.08, 0.15), ["Q6_K"] = (0.05, 0.07),
    };

    [Trait("Category", "Integration")]
    [RealWeightsFact("HARTSY_LONGCAT_GGUF", "HARTSY_LONGCAT_BF16_INDEX")]
    public void TheRepackIsTheReleaseWithinEachTypesBound()
    {
        string path = Environment.GetEnvironmentVariable("HARTSY_LONGCAT_GGUF")!;
        string index = Environment.GetEnvironmentVariable("HARTSY_LONGCAT_BF16_INDEX")!;
        using CheckpointSource source = CheckpointSource.Open(path);
        using ShardedSafeTensorSet release = ShardedSafeTensorSet.OpenIndex(Path.GetDirectoryName(index)!, null, Path.GetFileName(index));
        Dictionary<string, Totals> types = new(StringComparer.Ordinal);
        List<string> failures = [];
        HashSet<string> covered = new(StringComparer.Ordinal);
        int equal = 0, controls = 0;
        foreach ((string key, Tensor part) in source.Weights.OrderBy(entry => entry.Key, StringComparer.Ordinal))
        {
            if (key.StartsWith("blocks.", StringComparison.Ordinal) && !key.StartsWith("blocks.0.", StringComparison.Ordinal))
                continue;
            (string name, int place) = LongCatGgufConverter.EngineName(key);
            Assert.True(release.Inventory.ContainsKey(name), $"'{key}' maps to '{name}', which the release does not hold.");
            covered.Add(name);
            float[] whole = Floats(release.GetTensor(name));
            int count = checked((int)part.ElementCount);
            ReadOnlySpan<float> reference = whole.AsSpan(Math.Max(place, 0) * count, count);
            float[] got = Floats(part);
            if (!part.DType.IsQuantized)
            {
                if (got.AsSpan().SequenceEqual(reference))
                    equal++;
                else
                    failures.Add($"{key}: unquantized and not the release's values");
                continue;
            }
            (double relRms, double elementBound) = Bounds[part.DType.Name];
            double error = RelRms(got, reference);
            (long violations, double worst) = ElementCheck(part, got, reference, elementBound);
            if (!(error <= relRms) || violations != 0)
                failures.Add($"{key} {part.DType.Name}: relRMS {error:E3} (bound {relRms}), {violations} elements over, "
                    + $"worst {worst:F3} of the bound");
            Totals total = types.TryGetValue(part.DType.Name, out Totals seen) ? seen : new Totals(0, 0, 0, 0);
            types[part.DType.Name] = new Totals(total.Parts + 1, Math.Max(total.RelRms, error), total.Violations + violations,
                Math.Max(total.Worst, worst));
            output.WriteLine($"{key,-48} {part.DType.Name,-5} relRMS {error:E3}  worst element {worst:F3} of bound  over {violations}");
            // The order of a joined weight: its first part against the second part's rows has to miss the bound.
            if (place == 0)
            {
                double swapped = RelRms(got, whole.AsSpan(count, count));
                if (!(swapped > relRms))
                    failures.Add($"{key}: matches the second part's rows too (relRMS {swapped:E3}), so the order is not shown");
                controls++;
            }
        }
        foreach ((string type, Totals total) in types.OrderBy(entry => entry.Key, StringComparer.Ordinal))
            output.WriteLine($"type {type,-5} parts {total.Parts,2}  largest relRMS {total.RelRms:E3}  "
                + $"elements over {total.Violations}  worst {total.Worst:F3}");
        output.WriteLine($"names covered {covered.Count} of {release.Inventory.Count}; unquantized bit-equal {equal}; order controls {controls}");

        string folder = Path.GetDirectoryName(path)!;
        if (Directory.GetFiles(folder, "*.gguf").Length == 1)
        {
            using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(folder);
            IEnumerable<string> held = checkpoint.Weights.Values.GroupBy(t => t.DType.Name).OrderBy(g => g.Key, StringComparer.Ordinal)
                .Select(g => $"{g.Count()} {g.Key}");
            output.WriteLine($"loaded: depth {checkpoint.Config.Depth}, {checkpoint.Weights.Count} tensors, {string.Join(", ", held)}");
            Assert.Equal(48, checkpoint.Config.Depth);
        }
        else
        {
            output.WriteLine("whole-file load skipped: the folder holds more than one repack");
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
        Assert.Equal(release.Inventory.Count, covered.Count);
        Assert.Equal(2, controls);
    }

    private readonly record struct Totals(int Parts, double RelRms, long Violations, double Worst);

    private static float[] Floats(Tensor tensor)
    {
        if (tensor.DType == DType.F32)
            return tensor.AsReadOnlySpan<float>().ToArray();
        if (tensor.DType == DType.BF16)
        {
            ReadOnlySpan<ushort> bits = tensor.AsReadOnlySpan<ushort>();
            float[] values = new float[bits.Length];
            for (int i = 0; i < values.Length; i++)
                values[i] = BFloat16Bits.ToFloat(bits[i]);
            return values;
        }
        using Tensor wide = GgufDequantizer.Dequantize(tensor, DType.F32);
        return wide.AsReadOnlySpan<float>().ToArray();
    }

    private static double RelRms(ReadOnlySpan<float> got, ReadOnlySpan<float> reference)
    {
        double error = 0, power = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = (double)got[i] - reference[i];
            error += d * d;
            power += (double)reference[i] * reference[i];
        }
        return Math.Sqrt(error / power);
    }

    /// <summary>The elements past their bound, and the largest error as a share of the bound.</summary>
    private static (long Violations, double Worst) ElementCheck(Tensor part, float[] got, ReadOnlySpan<float> reference, double factor)
    {
        DType dtype = part.DType;
        bool kQuant = dtype.Name.EndsWith("_K", StringComparison.Ordinal);
        bool offset = dtype == DType.Q4_1 || dtype == DType.Q5_1;
        ReadOnlySpan<byte> bytes = part.AsReadOnlySpan<byte>();
        long violations = 0;
        double worst = 0;
        int size = dtype.BlockElementCount;
        for (int start = 0, block = 0; start < got.Length; start += size, block++)
        {
            double bound;
            if (kQuant)
            {
                float largest = 0f;
                for (int i = start; i < start + size; i++)
                    largest = MathF.Max(largest, MathF.Abs(reference[i]));
                bound = factor * largest;
            }
            else
            {
                // A legacy block opens with its F16 scale d; the offset types follow it with their F16 offset m.
                ReadOnlySpan<byte> head = bytes[(block * dtype.BlockByteSize)..];
                Half scale = Half.Abs(MemoryMarshal.Read<Half>(head));
                bound = factor * (double)scale
                    + (offset ? Math.Abs((double)MemoryMarshal.Read<Half>(head[2..])) / 1024 : 0)
                    + (dtype == DType.Q8_0 ? 127.5 * ((double)Half.BitIncrement(scale) - (double)scale) / 2 : 0);
            }
            for (int i = start; i < start + size; i++)
            {
                double error = Math.Abs((double)got[i] - reference[i]);
                if (error > bound)
                    violations++;
                if (bound > 0)
                    worst = Math.Max(worst, error / bound);
            }
        }
        return (violations, worst);
    }
}

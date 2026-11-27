using System.Text.Json;
using System.Text.RegularExpressions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Compares one INT8 weight of each linear type of a per-row INT8 LongCat release with the BF16 weight of the same name in the dense
/// release it came from. Only the shards holding the compared tensors are opened, so a folder that holds just those shards (or files of the same
/// names holding just those tensors) is enough.</summary>
/// <remarks>Two bounds, per element, with <c>s</c> the row's scale, <c>q</c> the stored int8 and <c>w</c> the BF16 weight. The release bound:
/// <c>|q·s − w| ≤ (0.5 + 1e-4)·s + 2^-8·|w|</c>, half a quantization step, a slack for the F32 division that made <c>q</c>, and the BF16 rounding
/// of <c>w</c> in case the INT8 weights were made from F32 ones. The engine value: the F32 weight the CPU widening gives is exactly <c>q·s</c> in
/// F32 with its low 16 bits dropped, the codec's truncation to BF16.</remarks>
internal static class LongCatInt8ReleaseCheck
{
    /// <summary>The block index in a key; every block's linear of one name is the same type.</summary>
    private static readonly Regex Block = new(@"^blocks\.(\d+)\.", RegexOptions.Compiled);

    /// <summary>One linear type's compared weight. <paramref name="MaxStepError"/> is <c>|q·s − w| / s</c> at its largest,
    /// <paramref name="ScaleRuleError"/> the largest relative difference between <c>127·s</c> and the row's largest <c>|w|</c>.</summary>
    internal sealed record LinearReport(string Type, string Key, TensorShape Shape, double MaxStepError, double ScaleRuleError, double RelRms,
        long Minus128, long ZeroScaleRows, double MaxExcess, long ReleaseViolations, long EngineViolations)
    {
        public override string ToString() => FormattableString.Invariant(
            $"{Type,-40} {Key,-42} {Shape} step {MaxStepError:F4} scale-rule {ScaleRuleError:E2} relRMS {RelRms:E3} q=-128 {Minus128} ")
            + FormattableString.Invariant(
                $"zero-rows {ZeroScaleRows} max-excess {MaxExcess:E2} release-violations {ReleaseViolations} engine-violations {EngineViolations}");
    }

    /// <summary>Compares the lowest-numbered weight of every quantized linear type listed by <paramref name="int8Index"/> with its namesake
    /// under <paramref name="denseIndex"/>. Throws when a compared weight, its scale or its dense namesake is missing or misshapen.</summary>
    public static List<LinearReport> Run(string int8Index, string denseIndex)
    {
        Dictionary<string, string> int8Map = WeightMap(int8Index), denseMap = WeightMap(denseIndex);
        SortedDictionary<string, (int Block, string Name)> types = new(StringComparer.Ordinal);
        foreach (string key in int8Map.Keys.Where(static k => k.EndsWith(LongCatInt8Converter.QuantizedSuffix, StringComparison.Ordinal)))
        {
            string name = key[..^LongCatInt8Converter.QuantizedSuffix.Length];
            Match match = Block.Match(name);
            int block = match.Success ? int.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture) : -1;
            string type = Block.Replace(name, "blocks.*.");
            if (!types.TryGetValue(type, out (int Block, string Name) first) || block < first.Block)
                types[type] = (block, name);
        }
        if (types.Count == 0)
            throw new InvalidDataException($"'{int8Index}' lists no {LongCatInt8Converter.QuantizedSuffix} weight.");

        string[] names = [.. types.Values.Select(static t => t.Name)];
        using ShardedSafeTensorSet int8 = Open(int8Index, names.SelectMany(name => new[]
        {
            Shard(int8Map, name + LongCatInt8Converter.QuantizedSuffix), Shard(int8Map, name + LongCatInt8Converter.ScaleSuffix),
        }));
        using ShardedSafeTensorSet dense = Open(denseIndex, names.Select(name => Shard(denseMap, name + ".weight")));
        return [.. types.Select(t => Compare(t.Key, t.Value.Name, int8, dense))];
    }

    private static LinearReport Compare(string type, string name, ShardedSafeTensorSet int8, ShardedSafeTensorSet dense)
    {
        Tensor quantized = int8.GetTensor(name + LongCatInt8Converter.QuantizedSuffix);
        Tensor scales = int8.GetTensor(name + LongCatInt8Converter.ScaleSuffix);
        Tensor weight = dense.GetTensor(name + ".weight");
        if (weight.DType != DType.BF16 || !weight.Shape.Equals(quantized.Shape))
            throw new InvalidDataException($"'{name}': the dense weight is {weight.DType} {weight.Shape}; the INT8 one is {quantized.Shape}.");

        // The engine's own path: the converter, then the widening a backend without a resident int8 path takes.
        Dictionary<string, Tensor> converted = LongCatInt8Converter.Convert(new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            [name + LongCatInt8Converter.QuantizedSuffix] = quantized,
            [name + LongCatInt8Converter.ScaleSuffix] = scales,
        });
        using QuantizedWeightPolicy.PreparedWeights prepared =
            QuantizedWeightPolicy.PrepareFor(converted, static _ => false, "longcat-int8-release-check", DType.F32);

        long rows = quantized.Shape[0], cols = quantized.Shape[1];
        if (scales.ElementCount != rows)
            throw new InvalidDataException($"'{name}': {scales.ElementCount} scales for {rows} rows.");
        ReadOnlySpan<byte> q = quantized.AsReadOnlySpan<byte>();
        ReadOnlySpan<float> s = scales.AsReadOnlySpan<float>(), engine = converted[name + ".weight"].AsReadOnlySpan<float>();
        ReadOnlySpan<ushort> w = weight.AsReadOnlySpan<ushort>();
        double maxStep = 0, maxRule = 0, difference = 0, reference = 0, maxExcess = double.NegativeInfinity;
        long minus128 = 0, zeroRows = 0, releaseViolations = 0, engineViolations = 0;
        for (long r = 0; r < rows; r++)
        {
            float scale = s[(int)r];
            if (!float.IsFinite(scale) || scale < 0f)
                throw new InvalidDataException($"'{name}': row {r} has scale {scale}.");
            double rowMax = 0;
            for (long c = 0; c < cols; c++)
            {
                int i = checked((int)(r * cols + c));
                sbyte level = unchecked((sbyte)q[i]);
                double dequantized = (double)level * scale;
                double value = BitConverter.UInt32BitsToSingle((uint)w[i] << 16);
                double error = Math.Abs(dequantized - value);
                double excess = error - ((0.5 + 1e-4) * scale + Math.Abs(value) / 256.0);
                // A NaN weight fails rather than slipping past the comparison.
                if (!(excess <= 0))
                    releaseViolations++;
                maxExcess = Math.Max(maxExcess, excess);
                if (BitConverter.SingleToUInt32Bits(engine[i]) != (uint)TensorCasts.F32ToBf16Bits(level * scale) << 16)
                    engineViolations++;
                if (scale > 0f)
                    maxStep = Math.Max(maxStep, error / scale);
                if (level == sbyte.MinValue)
                    minus128++;
                rowMax = Math.Max(rowMax, Math.Abs(value));
                difference += error * error;
                reference += value * value;
            }
            if (scale == 0f)
                zeroRows++;
            if (rowMax > 0)
                maxRule = Math.Max(maxRule, Math.Abs(127.0 * scale - rowMax) / rowMax);
        }
        return new LinearReport(type, name, quantized.Shape, maxStep, maxRule, Math.Sqrt(difference / reference), minus128, zeroRows, maxExcess,
            releaseViolations, engineViolations);
    }

    private static string Shard(Dictionary<string, string> map, string key) =>
        map.TryGetValue(key, out string? file) ? file : throw new InvalidDataException($"The index lists no '{key}'.");

    private static ShardedSafeTensorSet Open(string index, IEnumerable<string> shards)
    {
        string directory = Path.GetDirectoryName(Path.GetFullPath(index))!;   // a full file path always has a directory
        return ShardedSafeTensorSet.OpenFiles([.. shards.Distinct(StringComparer.Ordinal).Select(file => Path.Combine(directory, file))]);
    }

    private static Dictionary<string, string> WeightMap(string index)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(index));
        return document.RootElement.GetProperty("weight_map").EnumerateObject()
            .ToDictionary(static p => p.Name, static p => p.Value.GetString()!, StringComparer.Ordinal);
    }
}

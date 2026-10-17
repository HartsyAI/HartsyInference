using System.Globalization;
using System.Text.RegularExpressions;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.ModelAssets.Gguf;

/// <summary>Resolves the parts of a llama.cpp split GGUF set and checks that they form one complete, consistent set.</summary>
internal static partial class GgufSplitDetector
{
    private const string SplitCountKey = "split.count";
    private const string SplitNoKey = "split.no";
    private const string SplitTensorCountKey = "split.tensors.count";

    /// <summary>Every part's path in order, or just <paramref name="path"/> when the file is not split.</summary>
    internal static IReadOnlyList<string> PartPaths(string path, GgufMetadata metadata)
    {
        if (!TryReadCount(metadata, SplitCountKey, out long count) || count <= 1)
            return [path];

        Match name = PartNameRegex().Match(Path.GetFileName(path));
        if (!name.Success)
            throw Invalid(
                $"'{path}' is part of a {count}-part split, but its name is not <stem>-NNNNN-of-NNNNN.gguf, so the other parts cannot be found");

        int namedCount = int.Parse(name.Groups["count"].Value, CultureInfo.InvariantCulture);
        if (namedCount != count)
            throw Invalid($"'{path}' is named as part of {namedCount}, but declares split.count {count}");

        string stem = name.Groups["stem"].Value;
        string directory = Path.GetDirectoryName(path) ?? string.Empty;
        List<string> parts = new((int)count);
        List<string> missing = [];
        for (int index = 1; index <= count; index++)
        {
            string part = Path.Combine(directory, $"{stem}-{index:D5}-of-{count:D5}.gguf");
            parts.Add(part);
            if (!File.Exists(part))
                missing.Add(Path.GetFileName(part));
        }

        if (missing.Count > 0)
            throw Invalid($"split GGUF '{stem}' is incomplete: {missing.Count} of {count} parts missing ({string.Join(", ", missing)})");
        return parts;
    }

    /// <summary>Checks that the part at <paramref name="index"/> declares its own position and the set's part count.</summary>
    internal static void ValidatePart(GgufMetadata metadata, string path, int index, int count)
    {
        bool positioned = TryReadCount(metadata, SplitNoKey, out long no) && no == index;
        bool counted = TryReadCount(metadata, SplitCountKey, out long declared) && declared == count;
        if (!positioned || !counted)
            throw Invalid($"'{path}' should be part {index + 1} of {count}, but its metadata declares otherwise");
    }

    /// <summary>The tensor total the set declares, or null when the metadata does not say.</summary>
    internal static long? DeclaredTensorCount(GgufMetadata metadata) =>
        TryReadCount(metadata, SplitTensorCountKey, out long total) ? total : null;

    [GeneratedRegex(@"^(?<stem>.+)-\d{5}-of-(?<count>\d{5})\.gguf$", RegexOptions.CultureInvariant)]
    private static partial Regex PartNameRegex();

    private static UnsupportedModelException Invalid(string message) => new(message, architecture: null, format: "GGUF");

    private static bool TryReadCount(GgufMetadata metadata, string key, out long value)
    {
        if (metadata.TryGetValue(key, out object? raw))
            return TryToLong(raw, out value);
        value = 0;
        return false;
    }

    private static bool TryToLong(object? value, out long result)
    {
        switch (value)
        {
            case byte or sbyte or short or ushort or int or uint or long:
                result = Convert.ToInt64(value);
                return true;
            case ulong unsigned when unsigned <= long.MaxValue:
                result = (long)unsigned;
                return true;
            default:
                result = 0;
                return false;
        }
    }
}

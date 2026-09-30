using HartsyInference.Core.Exceptions;

namespace HartsyInference.ModelAssets.Gguf;

/// <summary>Detects llama.cpp split GGUF files, which this loader cannot read piecewise.</summary>
internal static class GgufSplitDetector
{
    private const string SplitCountKey = "split.count";
    private const string SplitNoKey = "split.no";

    /// <summary>Throws when the metadata declares more than one split, since the other parts' tensors would silently be missing.</summary>
    internal static void ThrowIfSplit(GgufMetadata metadata, string path)
    {
        if (!metadata.TryGetValue(SplitCountKey, out object? raw) || !TryToLong(raw, out long count) || count <= 1)
            return;

        string part = metadata.TryGetValue(SplitNoKey, out object? no) && TryToLong(no, out long index)
            ? $"part {index + 1} of {count}"
            : $"one of {count} parts";
        throw new UnsupportedModelException(
            $"'{path}' is a split GGUF ({part}); loading a single part would leave tensors missing. "
            + "Merge the parts first with `llama-gguf-split --merge <first-part.gguf> <merged.gguf>`, then load the merged file.",
            architecture: null, format: "GGUF");
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

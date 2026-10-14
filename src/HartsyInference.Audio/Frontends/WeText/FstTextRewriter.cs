using System.Text;

namespace HartsyInference.Audio.Frontends.WeText;

/// <summary>kaldifst's <c>TextNormalizer</c>: rewrites a string through a byte-labelled FST by composing the linear
/// acceptor over its UTF-8 bytes with the FST and reading the one-best path's non-epsilon output bytes. Returns an
/// empty string when the FST accepts no path, as kaldifst does. Stateless and safe to share across threads.</summary>
public sealed class FstTextRewriter
{
    private static readonly UTF8Encoding StrictUtf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    /// <summary>Wraps an FST whose labels are UTF-8 bytes.</summary>
    public FstTextRewriter(StdVectorFst fst)
    {
        ArgumentNullException.ThrowIfNull(fst);
        Fst = fst;
    }

    /// <summary>The rule FST.</summary>
    public StdVectorFst Fst { get; }

    /// <summary>Reads the FST at <paramref name="path"/>.</summary>
    public static FstTextRewriter Load(string path) => new(StdVectorFst.Read(path));

    /// <summary>The one-best rewrite of <paramref name="text"/>; empty when no path exists. Throws on a NUL character
    /// (epsilon to the FST) and, as kaldifst's Python binding does, when the output is not valid UTF-8.</summary>
    public string Rewrite(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        byte[] bytes = StrictUtf8.GetBytes(text);
        int[] labels = new int[bytes.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            if (bytes[i] == 0) throw new ArgumentException("text contains a NUL character", nameof(text));
            labels[i] = bytes[i];
        }
        List<int> output = new(bytes.Length * 2);
        if (!FstOneBest.TryRun(Fst, labels, output)) return string.Empty;
        byte[] result = new byte[output.Count];
        for (int i = 0; i < result.Length; i++) result[i] = unchecked((byte)output[i]);
        return StrictUtf8.GetString(result);
    }
}

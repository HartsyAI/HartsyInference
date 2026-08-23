using System.Text;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>GPT-2 byte-level BPE codec: the reversible byte↔unicode mapping HuggingFace/GPT-2 use so every byte is a printable codepoint. Byte-level BPE tokenizers (Qwen, Llama-3, Mistral, GPT-OSS, DeepSeek) store and emit token text in this remapped space, where a leading space is <c>Ġ</c> (U+0120) and newline is <c>Ċ</c>. <see cref="Decode"/> reverses it back to real UTF-8 text — without it, decoded output leaks <c>Ġ</c> for spaces. Lifted from the working implementation in <c>WhisperTokenizer</c> into a shared util.</summary>
public static class ByteLevelCodec
{
    private static readonly char[] ByteToChar = BuildByteToUnicode();
    private static readonly Dictionary<char, byte> CharToByte = BuildUnicodeToByte();

    /// <summary>Reverses the GPT-2 byte_encoder mapping on a raw BPE-decoded string: each char → byte → UTF-8. Multi-byte UTF-8 characters span multiple tokens; concatenating the raw pieces yields a complete byte sequence that decodes correctly. Unknown codepoints are skipped.</summary>
    public static string Decode(string raw)
    {
        if (raw.Length == 0) return string.Empty;
        byte[] bytes = new byte[Encoding.UTF8.GetByteCount(raw)];
        int n = 0;
        foreach (char c in raw)
        {
            if (CharToByte.TryGetValue(c, out byte b)) bytes[n++] = b;
        }
        return Encoding.UTF8.GetString(bytes, 0, n);
    }

    /// <summary>Encodes a UTF-8 string into the GPT-2 byte-level space (each byte → its remapped char). The inverse of <see cref="Decode"/>; used to look up byte-level token strings for special-token matching.</summary>
    public static string Encode(string text)
    {
        if (text.Length == 0) return string.Empty;
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        StringBuilder sb = new(bytes.Length);
        foreach (byte b in bytes) sb.Append(ByteToChar[b]);
        return sb.ToString();
    }

    private static char[] BuildByteToUnicode()
    {
        // Identical to GPT-2's bytes_to_unicode(): [33..126] ∪ [161..172] ∪ [174..255] map to themselves;
        // remaining bytes get codepoints starting at 256.
        char[] table = new char[256];
        bool[] used = new bool[256];
        void Mark(int lo, int hi) { for (int b = lo; b <= hi; b++) { table[b] = (char)b; used[b] = true; } }
        Mark('!', '~'); Mark(161, 172); Mark(174, 255);
        int next = 256;
        for (int b = 0; b < 256; b++)
            if (!used[b]) { table[b] = (char)next; next++; }
        return table;
    }

    private static Dictionary<char, byte> BuildUnicodeToByte()
    {
        Dictionary<char, byte> map = new(256);
        char[] forward = BuildByteToUnicode();
        for (int b = 0; b < 256; b++) map[forward[b]] = (byte)b;
        return map;
    }
}

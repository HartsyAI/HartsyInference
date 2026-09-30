using System.Globalization;
using System.Text;
using System.Text.Json;

namespace HartsyInference.Tools.Parsing;

/// <summary>Converts Gemma 4's argument block (<c>{city:&lt;|"|&gt;Paris&lt;|"|&gt;,days:3,opts:{a:true}}</c>: unquoted keys, <c>&lt;|"|&gt;</c> string delimiters, JSON literals otherwise) into a JSON object. When the delimiter token was dropped by the detokenizer a bare word up to the next <c>,</c> / <c>}</c> / <c>]</c> reads as a string, so strings containing those characters cannot be recovered in that case.</summary>
internal static class GemmaCallDsl
{
    /// <summary>The string delimiter Gemma renders around string values (a user-defined token in the vocabulary).</summary>
    public const string Quote = "<|\"|>";

    private const int MaxDepth = 32;

    /// <summary>Converts <paramref name="block"/> (starting at its opening brace) to a JSON object; false when it does not parse.</summary>
    public static bool TryConvert(string block, out string json)
    {
        using MemoryStream buffer = new();
        using (Utf8JsonWriter writer = new(buffer))
        {
            int pos = 0;
            SkipWhitespace(block, ref pos);
            if (pos >= block.Length || block[pos] != '{' || !WriteValue(block, ref pos, writer, 0))
            {
                json = "";
                return false;
            }
            SkipWhitespace(block, ref pos);
            if (pos != block.Length)
            {
                json = "";
                return false;
            }
        }
        json = Encoding.UTF8.GetString(buffer.GetBuffer(), 0, (int)buffer.Length);
        return true;
    }

    private static bool WriteValue(string s, ref int pos, Utf8JsonWriter writer, int depth)
    {
        if (depth > MaxDepth) return false;
        SkipWhitespace(s, ref pos);
        if (pos >= s.Length) return false;
        if (StartsWith(s, pos, Quote))
        {
            pos += Quote.Length;
            int end = s.IndexOf(Quote, pos, StringComparison.Ordinal);
            if (end < 0) return false;
            writer.WriteStringValue(s.AsSpan(pos, end - pos));
            pos = end + Quote.Length;
            return true;
        }
        char c = s[pos];
        if (c == '"')
        {
            int end = s.IndexOf('"', pos + 1);
            if (end < 0) return false;
            writer.WriteStringValue(s.AsSpan(pos + 1, end - pos - 1));
            pos = end + 1;
            return true;
        }
        if (c == '{') return WriteObject(s, ref pos, writer, depth);
        if (c == '[') return WriteArray(s, ref pos, writer, depth);
        int start = pos;
        while (pos < s.Length && s[pos] is not (',' or '}' or ']')) pos++;
        ReadOnlySpan<char> word = s.AsSpan(start, pos - start).Trim();
        if (word.Length == 0) return false;
        if (word.SequenceEqual("true")) writer.WriteBooleanValue(true);
        else if (word.SequenceEqual("false")) writer.WriteBooleanValue(false);
        else if (word.SequenceEqual("null")) writer.WriteNullValue();
        else if (long.TryParse(word, NumberStyles.Integer, CultureInfo.InvariantCulture, out long integer)) writer.WriteNumberValue(integer);
        else if (double.TryParse(word, NumberStyles.Float, CultureInfo.InvariantCulture, out double number)) writer.WriteNumberValue(number);
        else writer.WriteStringValue(word);
        return true;
    }

    private static bool WriteObject(string s, ref int pos, Utf8JsonWriter writer, int depth)
    {
        pos++;
        writer.WriteStartObject();
        SkipWhitespace(s, ref pos);
        if (pos < s.Length && s[pos] == '}')
        {
            pos++;
            writer.WriteEndObject();
            return true;
        }
        while (true)
        {
            SkipWhitespace(s, ref pos);
            if (!ReadKey(s, ref pos, out ReadOnlySpan<char> key)) return false;
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length || s[pos] != ':') return false;
            pos++;
            writer.WritePropertyName(key);
            if (!WriteValue(s, ref pos, writer, depth + 1)) return false;
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length) return false;
            if (s[pos] == ',')
            {
                pos++;
                continue;
            }
            if (s[pos] != '}') return false;
            pos++;
            writer.WriteEndObject();
            return true;
        }
    }

    private static bool WriteArray(string s, ref int pos, Utf8JsonWriter writer, int depth)
    {
        pos++;
        writer.WriteStartArray();
        SkipWhitespace(s, ref pos);
        if (pos < s.Length && s[pos] == ']')
        {
            pos++;
            writer.WriteEndArray();
            return true;
        }
        while (true)
        {
            if (!WriteValue(s, ref pos, writer, depth + 1)) return false;
            SkipWhitespace(s, ref pos);
            if (pos >= s.Length) return false;
            if (s[pos] == ',')
            {
                pos++;
                continue;
            }
            if (s[pos] != ']') return false;
            pos++;
            writer.WriteEndArray();
            return true;
        }
    }

    private static bool ReadKey(string s, ref int pos, out ReadOnlySpan<char> key)
    {
        key = default;
        if (pos >= s.Length) return false;
        if (StartsWith(s, pos, Quote))
        {
            int start = pos + Quote.Length;
            int end = s.IndexOf(Quote, start, StringComparison.Ordinal);
            if (end < 0) return false;
            key = s.AsSpan(start, end - start);
            pos = end + Quote.Length;
            return key.Length > 0;
        }
        if (s[pos] == '"')
        {
            int end = s.IndexOf('"', pos + 1);
            if (end < 0) return false;
            key = s.AsSpan(pos + 1, end - pos - 1);
            pos = end + 1;
            return key.Length > 0;
        }
        int keyStart = pos;
        while (pos < s.Length && (char.IsLetterOrDigit(s[pos]) || s[pos] is '_' or '-' or '.')) pos++;
        key = s.AsSpan(keyStart, pos - keyStart);
        return key.Length > 0;
    }

    private static void SkipWhitespace(string s, ref int pos)
    {
        while (pos < s.Length && char.IsWhiteSpace(s[pos])) pos++;
    }

    private static bool StartsWith(string s, int pos, string literal)
        => pos + literal.Length <= s.Length && string.CompareOrdinal(s, pos, literal, 0, literal.Length) == 0;
}

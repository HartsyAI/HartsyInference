using System.Text;

namespace HartsyInference.Audio.Frontends.WeText;

/// <summary>Port of wetext's <c>TokenParser.reorder</c> for TN: parses the tagger output
/// (<c>name { key: "value" ... }</c> tokens) and rewrites each token with its fields in the order the verbalizer
/// expects. A token named in the order table keeps only the listed fields unless it carries
/// <c>preserve_order: "true"</c>. Input the Python parser would reject (assert) or loop on forever throws.</summary>
internal sealed class WeTextTokenParser
{
    private static readonly Dictionary<string, string[]> EnglishOrders = new(StringComparer.Ordinal)
    {
        ["date"] = ["preserve_order", "text", "day", "month", "year"],
        ["money"] = ["integer_part", "fractional_part", "quantity", "currency_maj"],
    };

    private static readonly Dictionary<string, string[]> ChineseOrders = new(StringComparer.Ordinal)
    {
        ["date"] = ["year", "month", "day"],
        ["fraction"] = ["denominator", "numerator"],
        ["measure"] = ["denominator", "numerator", "value"],
        ["money"] = ["value", "currency"],
        ["time"] = ["noon", "hour", "minute", "second"],
    };

    private readonly string _text;
    private int _index;
    private bool _eos;

    private WeTextTokenParser(string text) => _text = text;

    // The current character, or -1 past the end (Python's "<EOS>" sentinel).
    private int Current => _eos ? -1 : _text[_index];

    /// <summary>wetext <c>TokenParser(lang, "tn").reorder(text)</c> of <paramref name="version"/>: 0.1.x re-escapes
    /// values (<c>\</c> and <c>"</c>) and keeps fields outside the order table after the ordered ones; 0.0.4 writes
    /// values raw and drops those fields.</summary>
    public static string Reorder(string tagged, WeTextLanguage language, WeTextVersion version = WeTextVersion.V0_0_4)
    {
        ArgumentNullException.ThrowIfNull(tagged);
        Dictionary<string, string[]> orders = language switch
        {
            WeTextLanguage.English => EnglishOrders,
            WeTextLanguage.Chinese => ChineseOrders,
            _ => throw new ArgumentOutOfRangeException(nameof(language), language, "reorder needs a concrete language"),
        };
        if (tagged.Length == 0) throw new FormatException("the tagger produced no tokens");
        WeTextTokenParser parser = new(tagged);
        StringBuilder output = new(tagged.Length + 16);
        foreach (Token token in parser.Parse()) token.Write(output, orders, version).Append(' ');
        return PythonStrip(output.ToString());
    }

    /// <summary>Python <c>str.strip()</c>: <see cref="char.IsWhiteSpace(char)"/> plus the U+001C..U+001F
    /// separators.</summary>
    internal static string PythonStrip(string text)
    {
        int start = 0, end = text.Length;
        while (start < end && IsPythonSpace(text[start])) start++;
        while (end > start && IsPythonSpace(text[end - 1])) end--;
        return text[start..end];
    }

    private static bool IsPythonSpace(char c) => char.IsWhiteSpace(c) || (c >= '\x1c' && c <= '\x1f');

    private List<Token> Parse()
    {
        List<Token> tokens = [];
        while (ParseWhitespace())
        {
            Token token = new(ParseKey());
            ParseChars(" { ");
            while (ParseWhitespace())
            {
                if (Current == '}')
                {
                    ParseChar('}');
                    break;
                }
                string key = ParseKey();
                ParseChars(": \"");
                string value = ParseValue();
                ParseChar('"');
                token.Append(key, value);
            }
            tokens.Add(token);
        }
        return tokens;
    }

    private bool Read()
    {
        if (_index < _text.Length - 1)
        {
            _index++;
            return true;
        }
        _eos = true;
        return false;
    }

    private bool ParseWhitespace()
    {
        bool notEos = !_eos;
        while (notEos && Current == ' ') notEos = Read();
        return notEos;
    }

    private bool ParseChar(char expected)
    {
        if (Current != expected) return false;
        Read();
        return true;
    }

    private void ParseChars(string expected)
    {
        foreach (char c in expected) ParseChar(c);
    }

    private string ParseKey()
    {
        if (_eos) throw Malformed("a key at the end of input");
        if (Current is ' ' or '\t' or '\n' or '\r' or '\v' or '\f') throw Malformed("a key starting with whitespace");
        int start = _index;
        while (!_eos && (char.IsAsciiLetter(_text[_index]) || _text[_index] == '_')) Read();
        return _text[start..(_eos ? _text.Length : _index)];
    }

    private string ParseValue()
    {
        if (_eos) throw Malformed("a value at the end of input");
        StringBuilder value = new();
        while (Current != '"')
        {
            if (_eos) throw Malformed("an unterminated value");
            char c = _text[_index];
            value.Append(c);
            Read();
            if (c != '\\') continue;
            if (_eos) throw Malformed("an escape at the end of input");
            value.Append(_text[_index]);
            Read();
        }
        return value.ToString();
    }

    private FormatException Malformed(string what) =>
        new($"malformed tagger output: {what} at offset {_index} in \"{_text}\"");

    private sealed class Token(string name)
    {
        private readonly List<string> _order = [];
        private readonly Dictionary<string, string> _members = new(StringComparer.Ordinal);

        public void Append(string key, string value)
        {
            _order.Add(key);
            _members[key] = value;
        }

        public StringBuilder Write(StringBuilder output, Dictionary<string, string[]> orders, WeTextVersion version)
        {
            output.Append(name).Append(" {");
            IReadOnlyList<string> order = _order;
            if (orders.TryGetValue(name, out string[]? fixedOrder)
                && !(_members.TryGetValue("preserve_order", out string? preserve) && preserve == "true"))
                order = version == WeTextVersion.V0_1_8 ? [.. fixedOrder, .. _order.Where(k => Array.IndexOf(fixedOrder, k) < 0)] : fixedOrder;
            foreach (string key in order)
            {
                if (!_members.TryGetValue(key, out string? value)) continue;
                if (version == WeTextVersion.V0_1_8) value = value.Replace(@"\", @"\\", StringComparison.Ordinal).Replace("\"", "\\\"", StringComparison.Ordinal);
                output.Append(' ').Append(key).Append(": \"").Append(value).Append('"');
            }
            return output.Append(" }");
        }
    }
}

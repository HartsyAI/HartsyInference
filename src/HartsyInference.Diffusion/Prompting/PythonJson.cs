using System.Globalization;
using System.Text;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>An integer literal kept as its digits, so a value past 64 bits survives like Python's <c>int</c>.</summary>
internal readonly record struct PyInt(string Digits);

/// <summary>An insertion-ordered JSON object; a repeated key keeps its first position and takes the last value, as a Python dict does.</summary>
internal sealed class PyDict
{
    private readonly List<string> _keys = [];
    private readonly Dictionary<string, object?> _values = new(StringComparer.Ordinal);

    public int Count => _keys.Count;

    public IReadOnlyList<string> Keys => _keys;

    public bool ContainsKey(string key) => _values.ContainsKey(key);

    public bool TryGetValue(string key, out object? value) => _values.TryGetValue(key, out value);

    public void Set(string key, object? value)
    {
        if (!_values.ContainsKey(key))
            _keys.Add(key);
        _values[key] = value;
    }

    public object? Pop(string key)
    {
        if (!_values.Remove(key, out object? value))
            return null;
        _keys.Remove(key);
        return value;
    }

    public object? Get(string key) => _values.TryGetValue(key, out object? value) ? value : null;
}

/// <summary>Python's <c>json.loads</c> and <c>json.dumps(separators=(",", ":"), ensure_ascii=False)</c>, because ComfyUI's Fibo 2.0
/// caption handling goes through them: NaN and Infinity literals parse, floats print as <c>repr</c>, only control characters, quotes
/// and backslashes are escaped. Values are null, bool, string, <see cref="PyInt"/>, double, <see cref="List{T}"/> and <see cref="PyDict"/>.</summary>
internal static class PythonJson
{
    /// <summary>Parses <paramref name="text"/>; false when Python's decoder would raise <c>JSONDecodeError</c>.</summary>
    public static bool TryParse(string text, out object? value)
    {
        int position = 0;
        value = null;
        SkipWhitespace(text, ref position);
        if (!TryParseValue(text, ref position, out object? parsed))
            return false;
        SkipWhitespace(text, ref position);
        if (position != text.Length)
            return false;
        value = parsed;
        return true;
    }

    /// <summary>Compact serialization.</summary>
    public static string Dumps(object? value)
    {
        StringBuilder builder = new();
        Write(builder, value);
        return builder.ToString();
    }

    /// <summary>Python's <c>repr(float)</c>, with <c>json.dumps</c>'s spellings for the non-finite values.</summary>
    public static string FloatRepr(double value)
    {
        if (double.IsNaN(value))
            return "NaN";
        if (double.IsPositiveInfinity(value))
            return "Infinity";
        if (double.IsNegativeInfinity(value))
            return "-Infinity";
        if (value == 0)
            return double.IsNegative(value) ? "-0.0" : "0.0";

        string shortest = Math.Abs(value).ToString("R", CultureInfo.InvariantCulture);
        int exponentAt = shortest.IndexOfAny(['E', 'e']);
        int exponent = exponentAt < 0 ? 0 : int.Parse(shortest.AsSpan(exponentAt + 1), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
        string mantissa = exponentAt < 0 ? shortest : shortest[..exponentAt];
        int pointAt = mantissa.IndexOf('.');
        string digits = pointAt < 0 ? mantissa : mantissa.Remove(pointAt, 1);
        int decimalPoint = (pointAt < 0 ? mantissa.Length : pointAt) + exponent;
        int leading = 0;
        while (leading < digits.Length - 1 && digits[leading] == '0')
            leading++;
        digits = digits[leading..].TrimEnd('0');
        decimalPoint -= leading;
        if (digits.Length == 0)
            digits = "0";

        StringBuilder text = new();
        if (value < 0)
            text.Append('-');
        if (decimalPoint > 16 || decimalPoint < -3)
        {
            text.Append(digits[0]);
            if (digits.Length > 1)
                text.Append('.').Append(digits, 1, digits.Length - 1);
            int scientific = decimalPoint - 1;
            text.Append('e').Append(scientific < 0 ? '-' : '+');
            text.Append(Math.Abs(scientific).ToString("00", CultureInfo.InvariantCulture));
        }
        else if (decimalPoint <= 0)
        {
            text.Append("0.").Append('0', -decimalPoint).Append(digits);
        }
        else if (decimalPoint >= digits.Length)
        {
            text.Append(digits).Append('0', decimalPoint - digits.Length).Append(".0");
        }
        else
        {
            text.Append(digits, 0, decimalPoint).Append('.').Append(digits, decimalPoint, digits.Length - decimalPoint);
        }
        return text.ToString();
    }

    private static void Write(StringBuilder builder, object? value)
    {
        switch (value)
        {
            case null:
                builder.Append("null");
                break;
            case bool flag:
                builder.Append(flag ? "true" : "false");
                break;
            case string text:
                WriteString(builder, text);
                break;
            case PyInt integer:
                builder.Append(integer.Digits);
                break;
            case double number:
                builder.Append(FloatRepr(number));
                break;
            case List<object?> list:
                builder.Append('[');
                for (int i = 0; i < list.Count; i++)
                {
                    if (i > 0)
                        builder.Append(',');
                    Write(builder, list[i]);
                }
                builder.Append(']');
                break;
            case PyDict dict:
                builder.Append('{');
                for (int i = 0; i < dict.Keys.Count; i++)
                {
                    if (i > 0)
                        builder.Append(',');
                    WriteString(builder, dict.Keys[i]);
                    builder.Append(':');
                    Write(builder, dict.Get(dict.Keys[i]));
                }
                builder.Append('}');
                break;
            default:
                throw new ArgumentException($"Cannot serialize a {value.GetType().Name}.", nameof(value));
        }
    }

    private static void WriteString(StringBuilder builder, string text)
    {
        builder.Append('"');
        foreach (char c in text)
        {
            switch (c)
            {
                case '"': builder.Append("\\\""); break;
                case '\\': builder.Append("\\\\"); break;
                case '\n': builder.Append("\\n"); break;
                case '\r': builder.Append("\\r"); break;
                case '\t': builder.Append("\\t"); break;
                case '\b': builder.Append("\\b"); break;
                case '\f': builder.Append("\\f"); break;
                default:
                    if (c < 0x20)
                        builder.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    else
                        builder.Append(c);
                    break;
            }
        }
        builder.Append('"');
    }

    private static void SkipWhitespace(string text, ref int position)
    {
        while (position < text.Length && text[position] is ' ' or '\t' or '\n' or '\r')
            position++;
    }

    private static bool TryParseValue(string text, ref int position, out object? value)
    {
        value = null;
        if (position >= text.Length)
            return false;
        char c = text[position];
        switch (c)
        {
            case '"':
                if (!TryParseString(text, ref position, out string? parsedString))
                    return false;
                value = parsedString;
                return true;
            case '{':
                return TryParseObject(text, ref position, out value);
            case '[':
                return TryParseArray(text, ref position, out value);
            case 'n':
                return TryLiteral(text, ref position, "null", null, out value);
            case 't':
                return TryLiteral(text, ref position, "true", true, out value);
            case 'f':
                return TryLiteral(text, ref position, "false", false, out value);
            case 'N':
                return TryLiteral(text, ref position, "NaN", double.NaN, out value);
            case 'I':
                return TryLiteral(text, ref position, "Infinity", double.PositiveInfinity, out value);
            default:
                if (c == '-' && string.CompareOrdinal(text, position, "-Infinity", 0, 9) == 0)
                    return TryLiteral(text, ref position, "-Infinity", double.NegativeInfinity, out value);
                return TryParseNumber(text, ref position, out value);
        }
    }

    private static bool TryLiteral(string text, ref int position, string literal, object? result, out object? value)
    {
        value = null;
        if (string.CompareOrdinal(text, position, literal, 0, literal.Length) != 0)
            return false;
        position += literal.Length;
        value = result;
        return true;
    }

    private static bool TryParseNumber(string text, ref int position, out object? value)
    {
        value = null;
        int start = position;
        int i = position;
        if (i < text.Length && text[i] == '-')
            i++;
        if (i >= text.Length || !char.IsAsciiDigit(text[i]))
            return false;
        if (text[i] == '0')
            i++;
        else
            while (i < text.Length && char.IsAsciiDigit(text[i]))
                i++;
        int integerEnd = i;
        bool isFloat = false;
        if (i + 1 < text.Length && text[i] == '.' && char.IsAsciiDigit(text[i + 1]))
        {
            isFloat = true;
            i++;
            while (i < text.Length && char.IsAsciiDigit(text[i]))
                i++;
        }
        if (i < text.Length && text[i] is 'e' or 'E')
        {
            int j = i + 1;
            if (j < text.Length && text[j] is '+' or '-')
                j++;
            if (j < text.Length && char.IsAsciiDigit(text[j]))
            {
                isFloat = true;
                while (j < text.Length && char.IsAsciiDigit(text[j]))
                    j++;
                i = j;
            }
        }
        position = i;
        if (!isFloat)
        {
            string digits = text[start..integerEnd];
            value = new PyInt(digits == "-0" ? "0" : digits);
            return true;
        }
        value = double.Parse(text.AsSpan(start, i - start), NumberStyles.Float, CultureInfo.InvariantCulture);
        return true;
    }

    private static bool TryParseString(string text, ref int position, out string? value)
    {
        value = null;
        StringBuilder builder = new();
        int i = position + 1;
        while (i < text.Length)
        {
            char c = text[i++];
            if (c == '"')
            {
                position = i;
                value = builder.ToString();
                return true;
            }
            if (c < 0x20)
                return false;
            if (c != '\\')
            {
                builder.Append(c);
                continue;
            }
            if (i >= text.Length)
                return false;
            char escape = text[i++];
            switch (escape)
            {
                case '"': builder.Append('"'); break;
                case '\\': builder.Append('\\'); break;
                case '/': builder.Append('/'); break;
                case 'b': builder.Append('\b'); break;
                case 'f': builder.Append('\f'); break;
                case 'n': builder.Append('\n'); break;
                case 'r': builder.Append('\r'); break;
                case 't': builder.Append('\t'); break;
                case 'u':
                    if (i + 4 > text.Length
                        || !int.TryParse(text.AsSpan(i, 4), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out int code)
                        || text[i] is '+' or '-' or ' ')
                        return false;
                    builder.Append((char)code);
                    i += 4;
                    break;
                default:
                    return false;
            }
        }
        return false;
    }

    private static bool TryParseArray(string text, ref int position, out object? value)
    {
        value = null;
        List<object?> list = [];
        int i = position + 1;
        SkipWhitespace(text, ref i);
        if (i < text.Length && text[i] == ']')
        {
            position = i + 1;
            value = list;
            return true;
        }
        while (true)
        {
            SkipWhitespace(text, ref i);
            if (!TryParseValue(text, ref i, out object? item))
                return false;
            list.Add(item);
            SkipWhitespace(text, ref i);
            if (i >= text.Length)
                return false;
            if (text[i] == ',')
            {
                i++;
                continue;
            }
            if (text[i] != ']')
                return false;
            position = i + 1;
            value = list;
            return true;
        }
    }

    private static bool TryParseObject(string text, ref int position, out object? value)
    {
        value = null;
        PyDict dict = new();
        int i = position + 1;
        SkipWhitespace(text, ref i);
        if (i < text.Length && text[i] == '}')
        {
            position = i + 1;
            value = dict;
            return true;
        }
        while (true)
        {
            SkipWhitespace(text, ref i);
            if (i >= text.Length || text[i] != '"' || !TryParseString(text, ref i, out string? key))
                return false;
            SkipWhitespace(text, ref i);
            if (i >= text.Length || text[i] != ':')
                return false;
            i++;
            SkipWhitespace(text, ref i);
            if (!TryParseValue(text, ref i, out object? item))
                return false;
            dict.Set(key!, item);
            SkipWhitespace(text, ref i);
            if (i >= text.Length)
                return false;
            if (text[i] == ',')
            {
                i++;
                continue;
            }
            if (text[i] != '}')
                return false;
            position = i + 1;
            value = dict;
            return true;
        }
    }
}

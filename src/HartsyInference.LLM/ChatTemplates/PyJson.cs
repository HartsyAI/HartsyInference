using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Parse and serialize JSON exactly like Python's <c>json.loads</c> / <c>json.dumps(ensure_ascii=False)</c> so rendered prompts match the reference byte for byte.</summary>
internal static class PyJson
{
    /// <summary>Parses JSON text into <see cref="PyJsonObject"/>, <see cref="List{Object}"/>, string, bool, null, BigInteger or double.</summary>
    public static object? Parse(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        return Convert(doc.RootElement);
    }

    /// <summary>Serializes a parsed value with Python's default separators (", " and ": ").</summary>
    public static string Dump(object? value)
    {
        StringBuilder sb = new();
        Write(sb, value);
        return sb.ToString();
    }

    private static object? Convert(JsonElement el)
    {
        switch (el.ValueKind)
        {
            case JsonValueKind.Object:
                PyJsonObject obj = new();
                foreach (JsonProperty p in el.EnumerateObject()) obj[p.Name] = Convert(p.Value);
                return obj;
            case JsonValueKind.Array:
                List<object?> list = new(el.GetArrayLength());
                foreach (JsonElement item in el.EnumerateArray()) list.Add(Convert(item));
                return list;
            case JsonValueKind.String:
                return el.GetString();
            case JsonValueKind.True:
                return true;
            case JsonValueKind.False:
                return false;
            case JsonValueKind.Number:
                return ConvertNumber(el.GetRawText());
            default:
                return null;
        }
    }

    // Python treats a token with no '.', 'e' or 'E' as an arbitrary-precision int, everything else as a float.
    private static object ConvertNumber(string raw)
    {
        if (raw.AsSpan().IndexOfAny('.', 'e', 'E') < 0) return BigInteger.Parse(raw, CultureInfo.InvariantCulture);
        return double.Parse(raw, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    private static void Write(StringBuilder sb, object? value)
    {
        switch (value)
        {
            case null: sb.Append("null"); break;
            case bool b: sb.Append(b ? "true" : "false"); break;
            case string s: WriteString(sb, s); break;
            case BigInteger n: sb.Append(n.ToString(CultureInfo.InvariantCulture)); break;
            case double d: sb.Append(FloatRepr(d)); break;
            case PyJsonObject obj: WriteObject(sb, obj); break;
            case List<object?> list: WriteArray(sb, list); break;
            default: throw new ArgumentException($"Cannot serialize {value.GetType().Name} to JSON.");
        }
    }

    private static void WriteObject(StringBuilder sb, PyJsonObject obj)
    {
        sb.Append('{');
        for (int i = 0; i < obj.Keys.Count; i++)
        {
            if (i > 0) sb.Append(", ");
            WriteString(sb, obj.Keys[i]);
            sb.Append(": ");
            Write(sb, obj[obj.Keys[i]]);
        }
        sb.Append('}');
    }

    private static void WriteArray(StringBuilder sb, List<object?> list)
    {
        sb.Append('[');
        for (int i = 0; i < list.Count; i++)
        {
            if (i > 0) sb.Append(", ");
            Write(sb, list[i]);
        }
        sb.Append(']');
    }

    private static void WriteString(StringBuilder sb, string s)
    {
        sb.Append('"');
        AppendEscaped(sb, s);
        sb.Append('"');
    }

    /// <summary>Appends <paramref name="s"/> escaped as the body of a JSON string exactly like <c>json.dumps(ensure_ascii=False)</c>; per-character, so it may be applied to a string in pieces.</summary>
    public static void AppendEscaped(StringBuilder sb, string s)
    {
        foreach (char c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                default:
                    if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    else sb.Append(c);
                    break;
            }
        }
    }

    /// <summary>Python's shortest-repr float formatting (exponent form when the decimal point sits at or below -4 or above 16).</summary>
    internal static string FloatRepr(double d)
    {
        if (double.IsNaN(d)) return "NaN";
        if (double.IsInfinity(d)) return d > 0 ? "Infinity" : "-Infinity";
        string sign = d < 0 || (d == 0 && double.IsNegative(d)) ? "-" : string.Empty;
        string r = Math.Abs(d).ToString("R", CultureInfo.InvariantCulture);
        int exp = 0;
        int e = r.IndexOf('E');
        if (e >= 0)
        {
            exp = int.Parse(r.AsSpan(e + 1), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
            r = r[..e];
        }
        int dot = r.IndexOf('.');
        string digits = dot < 0 ? r : r[..dot] + r[(dot + 1)..];
        int decpt = (dot < 0 ? r.Length : dot) + exp;
        int lead = 0;
        while (lead < digits.Length - 1 && digits[lead] == '0') { lead++; decpt--; }
        digits = digits[lead..].TrimEnd('0');
        if (digits.Length == 0) { digits = "0"; decpt = 1; }

        if (decpt <= -4 || decpt > 16)
        {
            int e10 = decpt - 1;
            string mantissa = digits.Length > 1 ? digits[..1] + "." + digits[1..] : digits;
            string expPart = (e10 < 0 ? "-" : "+") + Math.Abs(e10).ToString("00", CultureInfo.InvariantCulture);
            return sign + mantissa + "e" + expPart;
        }
        if (decpt <= 0) return sign + "0." + new string('0', -decpt) + digits;
        if (decpt >= digits.Length) return sign + digits + new string('0', decpt - digits.Length) + ".0";
        return sign + digits[..decpt] + "." + digits[decpt..];
    }
}

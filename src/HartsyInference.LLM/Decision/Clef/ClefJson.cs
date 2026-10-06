using System.Globalization;
using System.Text;
using System.Text.Json;

namespace HartsyInference.LLM.Decision.Clef;

/// <summary>Renders JSON the way Python's <c>json.dumps(ensure_ascii=False, separators=(",", ":"), sort_keys=True)</c> does,
/// because the exact characters are what get tokenized.</summary>
internal static class ClefJson
{
    /// <summary>A string is used as is; anything else is serialized canonically.</summary>
    internal static string Render(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.String)
        {
            return value.GetString()!;
        }
        StringBuilder sb = new();
        Write(sb, value);
        return sb.ToString();
    }

    private static void Write(StringBuilder sb, JsonElement value)
    {
        switch (value.ValueKind)
        {
            case JsonValueKind.Object:
                List<JsonProperty> props = [.. value.EnumerateObject()];
                props.Sort((a, b) => string.CompareOrdinal(a.Name, b.Name));
                sb.Append('{');
                for (int i = 0; i < props.Count; i++)
                {
                    if (i > 0)
                    {
                        sb.Append(',');
                    }
                    WriteString(sb, props[i].Name);
                    sb.Append(':');
                    Write(sb, props[i].Value);
                }
                sb.Append('}');
                break;
            case JsonValueKind.Array:
                sb.Append('[');
                bool first = true;
                foreach (JsonElement item in value.EnumerateArray())
                {
                    if (!first)
                    {
                        sb.Append(',');
                    }
                    first = false;
                    Write(sb, item);
                }
                sb.Append(']');
                break;
            case JsonValueKind.String:
                WriteString(sb, value.GetString()!);
                break;
            case JsonValueKind.Number:
                sb.Append(Number(value.GetRawText()));
                break;
            case JsonValueKind.True:
                sb.Append("true");
                break;
            case JsonValueKind.False:
                sb.Append("false");
                break;
            default:
                sb.Append("null");
                break;
        }
    }

    /// <summary>Writes <paramref name="value"/> as a JSON string literal, escaping what Python escapes.</summary>
    internal static void WriteString(StringBuilder sb, string value)
    {
        sb.Append('"');
        foreach (char c in value)
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
                    if (c < 0x20)
                    {
                        sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        sb.Append(c);
                    }
                    break;
            }
        }
        sb.Append('"');
    }

    private static string Number(string raw)
    {
        if (raw.AsSpan().IndexOfAny('.', 'e', 'E') < 0)
        {
            return raw;
        }
        return PythonFloat(double.Parse(raw, CultureInfo.InvariantCulture));
    }

    /// <summary>Python's <c>repr(float)</c>: shortest round-trip digits, scientific form outside 1e-4 .. 1e16.</summary>
    internal static string PythonFloat(double d)
    {
        if (double.IsNaN(d))
        {
            return "NaN";
        }
        if (double.IsInfinity(d))
        {
            return d > 0 ? "Infinity" : "-Infinity";
        }
        if (d == 0)
        {
            return double.IsNegative(d) ? "-0.0" : "0.0";
        }
        string r = Math.Abs(d).ToString("E16", CultureInfo.InvariantCulture);
        for (int digits = 1; digits <= 17; digits++)
        {
            string candidate = Math.Abs(d).ToString("E" + (digits - 1), CultureInfo.InvariantCulture);
            if (double.Parse(candidate, CultureInfo.InvariantCulture) == Math.Abs(d))
            {
                r = candidate;
                break;
            }
        }
        int e = r.IndexOf('E');
        string mantissa = r[..e].Replace(".", string.Empty);
        int exp = int.Parse(r[(e + 1)..], CultureInfo.InvariantCulture);
        mantissa = mantissa.TrimEnd('0');
        if (mantissa.Length == 0)
        {
            mantissa = "0";
        }
        string sign = d < 0 ? "-" : string.Empty;
        if (exp < -4 || exp >= 16)
        {
            string body = mantissa.Length == 1 ? mantissa : mantissa[0] + "." + mantissa[1..];
            return $"{sign}{body}e{(exp < 0 ? '-' : '+')}{Math.Abs(exp):00}";
        }
        if (exp >= 0)
        {
            string intPart = mantissa.Length > exp + 1 ? mantissa[..(exp + 1)] : mantissa.PadRight(exp + 1, '0');
            string frac = mantissa.Length > exp + 1 ? mantissa[(exp + 1)..] : "0";
            return $"{sign}{intPart}.{frac}";
        }
        return $"{sign}0.{new string('0', -exp - 1)}{mantissa}";
    }
}

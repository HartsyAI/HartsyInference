using System.Diagnostics.CodeAnalysis;
using System.Text;
using System.Text.RegularExpressions;

namespace HartsyInference.Audio.Frontends;

/// <summary>Port of cn2an 0.5.24's <c>Transform.transform(text, "an2cn")</c> (Ailln/cn2an, MIT): Arabic numbers in
/// running text become Chinese numerals — ranges before a measure word, dates (the year digit by digit), fractions,
/// percentages, degrees Celsius, then every remaining integer or decimal. A number cn2an cannot read (more than 16
/// integer digits, a non-ASCII digit that is not full-width) is left as written, as cn2an leaves it after its warning.</summary>
internal static partial class ChineseNumberNormalizer
{
    private const string MeasureWords = "斤|克|千克|公斤|吨|米|厘米|毫米|公里|升|毫升|元|角|分|个|只|条|张|块|瓶|杯|份|本|"
        + "辆|台|匹|头|位|亩|小时|分钟|秒|天|半";
    private const string Digits = "零一二三四五六七八九";
    private static readonly string[] Units = ["", "十", "百", "千", "万", "十", "百", "千", "亿", "十", "百", "千", "万", "十",
        "百", "千"];

    /// <summary>cn2an <c>transform(text, "an2cn")</c>.</summary>
    public static string Transform(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        text = RangeRegex().Replace(text, m => Range(m.Value));
        text = DateRegex().Replace(text, m => Date(m.Value));
        text = FractionRegex().Replace(text, m => Fraction(m.Value));
        text = PercentRegex().Replace(text, m => TryAn2Cn(m.Value[..^1], direct: false, out string? cn)
            ? "百分之" + cn : m.Value);
        text = CelsiusRegex().Replace(text, m => TryAn2Cn(m.Value[..^1], direct: false, out string? cn)
            ? cn + "摄氏度" : m.Value);
        return NumberRegex().Replace(text, m => TryAn2Cn(m.Value, direct: false, out string? cn) ? cn : m.Value);
    }

    /// <summary>cn2an <c>An2Cn.an2cn</c> in <c>low</c> (<paramref name="direct"/> false) or <c>direct</c> (digit by
    /// digit) mode; false where cn2an raises.</summary>
    internal static bool TryAn2Cn(string input, bool direct, [NotNullWhen(true)] out string? output)
    {
        output = null;
        if (input.Length == 0) return false;
        StringBuilder half = new(input.Length);
        foreach (char c in input)
        {
            // proces full_angle_to_half_angle; traditional_to_simplified leaves digits alone.
            char h = c == '　' ? ' ' : c >= '！' && c <= '～' ? (char)(c - 0xfee0) : c;
            if (!char.IsAsciiDigit(h) && h != '.' && h != '-') return false;
            half.Append(h);
        }
        string s = half.ToString();
        string sign = "";
        if (s[0] == '-')
        {
            sign = "负";
            s = s[1..];
        }
        if (direct)
        {
            StringBuilder sb = new(s.Length);
            foreach (char c in s)
            {
                if (c == '.') sb.Append('点');
                else if (char.IsAsciiDigit(c)) sb.Append(Digits[c - '0']);
                else return false;
            }
            output = sign + sb;
            return true;
        }
        string[] parts = s.Split('.');
        if (parts.Length > 2 || !TryInteger(parts[0], out string? integer)) return false;
        if (parts.Length == 1)
        {
            output = sign + integer;
            return true;
        }
        string fraction = parts[1].Length > 16 ? parts[1][..16] : parts[1];
        StringBuilder dec = new(parts[1].Length == 0 ? "" : "点");
        foreach (char c in fraction)
        {
            if (!char.IsAsciiDigit(c)) return false;
            dec.Append(Digits[c - '0']);
        }
        output = sign + integer + dec;
        return true;
    }

    // An2Cn.__integer_convert in low mode.
    private static bool TryInteger(string data, [NotNullWhen(true)] out string? output)
    {
        output = null;
        if (data.Length == 0) return false;
        foreach (char c in data)
        {
            if (!char.IsAsciiDigit(c)) return false;
        }
        string digits = data.TrimStart('0');
        if (digits.Length == 0) digits = "0";
        int n = digits.Length;
        if (n > Units.Length) return false;
        StringBuilder sb = new(n * 2);
        for (int i = 0; i < n; i++)
        {
            int d = digits[i] - '0';
            int place = n - i - 1;
            if (d != 0)
            {
                sb.Append(Digits[d]).Append(Units[place]);
                continue;
            }
            if (place % 4 == 0) sb.Append('零').Append(Units[place]);
            if (i > 0 && sb[^1] != '零') sb.Append('零');
        }
        string cn = sb.ToString().Replace("零零", "零").Replace("零万", "万").Replace("零亿", "亿").Replace("亿万", "亿")
            .Trim('零');
        cn = ZeroBeforeThousandRegex().Replace(cn, "$1$2");
        if (cn.StartsWith("一十", StringComparison.Ordinal)) cn = cn[1..];
        output = cn.Length == 0 ? "零" : cn;
        return true;
    }

    private static string Range(string value)
    {
        string[] ends = value.Split('-');
        return ends.Length == 2 && TryAn2Cn(ends[0], false, out string? from) && TryAn2Cn(ends[1], false, out string? to)
            ? from + "到" + to : value;
    }

    // The year reads digit by digit, the month and day as numbers.
    private static string Date(string value)
    {
        if (value.Length == 0) return value;
        bool failed = false;
        string result = YearDigitsRegex().Replace(value, m => Convert(m.Value, direct: true, ref failed));
        result = DigitsRegex().Replace(result, m => Convert(m.Value, direct: false, ref failed));
        return failed ? value : result;
    }

    private static string Fraction(string value)
    {
        bool failed = false;
        string result = DigitsRegex().Replace(value, m => Convert(m.Value, direct: false, ref failed));
        string[] parts = result.Split('/');
        return failed || parts.Length != 2 ? value : parts[1] + "分之" + parts[0];
    }

    private static string Convert(string digits, bool direct, ref bool failed)
    {
        if (TryAn2Cn(digits, direct, out string? cn)) return cn;
        failed = true;
        return digits;
    }

    [GeneratedRegex(@"\d+(\.\d+)?-\d+(\.\d+)?(?=(" + MeasureWords + "))")]
    private static partial Regex RangeRegex();

    [GeneratedRegex(@"(\d{2,4}年)?(\d{1,2}月)?(\d{1,2}日)?")]
    private static partial Regex DateRegex();

    [GeneratedRegex(@"\d+/\d+")]
    private static partial Regex FractionRegex();

    [GeneratedRegex(@"-?(\d+\.)?\d+%")]
    private static partial Regex PercentRegex();

    [GeneratedRegex(@"\d+℃")]
    private static partial Regex CelsiusRegex();

    [GeneratedRegex(@"-?(\d+\.)?\d+")]
    private static partial Regex NumberRegex();

    [GeneratedRegex(@"\d+(?=年)")]
    private static partial Regex YearDigitsRegex();

    [GeneratedRegex(@"\d+")]
    private static partial Regex DigitsRegex();

    [GeneratedRegex("([万亿])零([一二三四五六七八九壹贰叁肆伍陆柒捌玖][千仟])")]
    private static partial Regex ZeroBeforeThousandRegex();
}

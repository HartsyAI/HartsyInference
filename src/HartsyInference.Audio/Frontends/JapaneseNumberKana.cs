using System.Text;

namespace HartsyInference.Audio.Frontends;

/// <summary>misaki's <c>num2kana.Convert</c> (from Greatdane's Convert-Numbers-to-Japanese, MIT) for its hiragana
/// table: reads a run of ASCII digits as Japanese kana, with the 300/600/800/3000/8000 sound changes and いっせん inside
/// larger numbers. Covers up to nine digits as the original does.</summary>
internal static class JapaneseNumberKana
{
    private static readonly Dictionary<string, string> Kana = new()
    {
        ["0"] = "ゼロ", ["1"] = "いち", ["2"] = "に", ["3"] = "さん", ["4"] = "よん", ["5"] = "ご", ["6"] = "ろく",
        ["7"] = "なな", ["8"] = "はち", ["9"] = "きゅう", ["10"] = "じゅう", ["100"] = "ひゃく", ["1000"] = "せん",
        ["10000"] = "まん", ["100000000"] = "おく", ["300"] = "さんびゃく", ["600"] = "ろっぴゃく",
        ["800"] = "はっぴゃく", ["3000"] = "さんぜん", ["8000"] = "はっせん", ["01000"] = "いっせん",
    };

    /// <summary>The hiragana reading of the ASCII digit string <paramref name="digits"/>. misaki fails on runs over
    /// nine digits; those are read digit by digit here.</summary>
    public static string Convert(string digits)
    {
        ArgumentException.ThrowIfNullOrEmpty(digits);
        foreach (char c in digits)
        {
            if (c is < '0' or > '9') throw new ArgumentException($"'{digits}' is not an ASCII digit string.", nameof(digits));
        }
        if (digits.Length > 9)
        {
            StringBuilder sb = new();
            foreach (char c in digits) sb.Append(Kana[c.ToString()]);
            return sb.ToString();
        }
        int start = 0;
        while (start < digits.Length - 1 && digits[start] == '0') start++;
        return DoConvert(digits[start..]);
    }

    private static string DoConvert(string n) => n.Length switch
    {
        1 => Kana[n],
        2 => LenTwo(n),
        3 => LenThree(n),
        4 => LenFour(n, standAlone: true),
        _ => LenX(n),
    };

    private static string LenTwo(string n)
    {
        if (n[0] == '0') return Kana[n[1..]];
        if (n == "10") return Kana["10"];
        if (n[0] == '1') return Kana["10"] + Kana[n[1..]];
        if (n[1] == '0') return Kana[n[..1]] + Kana["10"];
        return Kana[n[..1]] + Kana["10"] + Kana[n[1..]];
    }

    private static string LenThree(string n)
    {
        StringBuilder sb = new();
        switch (n[0])
        {
            case '1': sb.Append(Kana["100"]); break;
            case '3': sb.Append(Kana["300"]); break;
            case '6': sb.Append(Kana["600"]); break;
            case '8': sb.Append(Kana["800"]); break;
            default: sb.Append(Kana[n[..1]]).Append(Kana["100"]); break;
        }
        if (n[1..] != "00") sb.Append(n[1] == '0' ? Kana[n[2..]] : LenTwo(n[1..]));
        return sb.ToString();
    }

    private static string LenFour(string n, bool standAlone)
    {
        if (n == "0000") return "";
        while (n[0] == '0') n = n[1..];
        if (n.Length == 1) return Kana[n];
        if (n.Length == 2) return LenTwo(n);
        if (n.Length == 3) return LenThree(n);
        StringBuilder sb = new();
        if (n[0] == '1') sb.Append(standAlone ? Kana["1000"] : Kana["01000"]);
        else if (n[0] == '3') sb.Append(Kana["3000"]);
        else if (n[0] == '8') sb.Append(Kana["8000"]);
        else sb.Append(Kana[n[..1]]).Append(Kana["1000"]);
        if (n[1..] != "000") sb.Append(n[1] == '0' ? LenTwo(n[2..]) : LenThree(n[1..]));
        return sb.ToString();
    }

    private static string LenX(string n)
    {
        StringBuilder sb = new();
        string head = n[..^4];
        switch (head.Length)
        {
            case 1: sb.Append(Kana[head]).Append(Kana["10000"]); break;
            case 2: sb.Append(LenTwo(head)).Append(Kana["10000"]); break;
            case 3: sb.Append(LenThree(head)).Append(Kana["10000"]); break;
            case 4: sb.Append(LenFour(head, standAlone: false)).Append(Kana["10000"]); break;
            default:
                sb.Append(Kana[n[..1]]).Append(Kana["100000000"]).Append(LenFour(n[1..5], standAlone: false));
                if (n[1..5] != "0000") sb.Append(Kana["10000"]);
                break;
        }
        sb.Append(LenFour(n[^4..], standAlone: false));
        return sb.ToString();
    }
}

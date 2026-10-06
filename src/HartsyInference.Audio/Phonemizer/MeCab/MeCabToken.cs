using System.Text;

namespace HartsyInference.Audio.Phonemizer.MeCab;

/// <summary>One morpheme on MeCab's best path, as fugashi exposes it: surface, raw feature CSV, the character
/// category id of its first character (<c>char_type</c>) and whether it came from the unknown-word dictionary.</summary>
internal readonly record struct MeCabToken(string Surface, string Feature, int CharType, bool IsUnknown)
{
    /// <summary>Feature field <paramref name="index"/> (UniDic: 9 = <c>pron</c>, 20 = <c>kana</c>), CSV-unquoted;
    /// null when the feature has fewer fields, as fugashi pads unknown words.</summary>
    public string? Field(int index)
    {
        string f = Feature;
        int field = 0;
        int i = 0;
        StringBuilder? sb = null;
        while (true)
        {
            string value;
            if (i < f.Length && f[i] == '"')
            {
                sb ??= new StringBuilder();
                sb.Clear();
                i++;
                while (i < f.Length)
                {
                    if (f[i] == '"')
                    {
                        if (i + 1 < f.Length && f[i + 1] == '"')
                        {
                            sb.Append('"');
                            i += 2;
                            continue;
                        }
                        i++;
                        break;
                    }
                    sb.Append(f[i++]);
                }
                int comma = f.IndexOf(',', i);
                if (comma < 0) comma = f.Length;
                sb.Append(f, i, comma - i);
                value = sb.ToString();
                i = comma;
            }
            else
            {
                int comma = f.IndexOf(',', i);
                if (comma < 0) comma = f.Length;
                value = f[i..comma];
                i = comma;
            }
            if (field == index) return value;
            if (i >= f.Length) return null;
            i++;
            field++;
        }
    }
}

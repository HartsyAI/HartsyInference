using System.Text;
using System.Text.Json;

namespace HartsyInference.Audio.Phonemizer;

/// <summary>Maps IPA phoneme codepoints to model token ids, matching how piper-phonemize tokenizes for a Piper voice; encoding follows piper's interleaving: begin-of-sentence, a pad, then each phoneme's id(s) each followed by a pad, then end-of-sentence (<c>^ _ p0 _ p1 _ … pN _ $</c>).</summary>
public sealed class PhonemeIdMap
{
    private const string Bos = "^";
    private const string Eos = "$";
    private const string Pad = "_";

    private readonly Dictionary<int, int[]> _byCodepoint;
    private readonly int[] _bos;
    private readonly int[] _eos;
    private readonly int[] _pad;

    private PhonemeIdMap(Dictionary<int, int[]> byCodepoint, int[] bos, int[] eos, int[] pad)
    {
        _byCodepoint = byCodepoint;
        _bos = bos;
        _eos = eos;
        _pad = pad;
    }

    /// <summary>Builds a map from a Piper <c>phoneme_id_map</c> object (string codepoint → id array).</summary>
    public static PhonemeIdMap FromPhonemeMap(IReadOnlyDictionary<string, int[]> phonemeIdMap)
    {
        Dictionary<int, int[]> byCp = new();
        foreach (KeyValuePair<string, int[]> kv in phonemeIdMap)
        {
            if (kv.Key.Length == 0) continue;
            byCp[char.ConvertToUtf32(kv.Key, 0)] = kv.Value;
        }
        return new PhonemeIdMap(byCp, phonemeIdMap.TryGetValue(Bos, out int[]? b) ? b : [1],
            phonemeIdMap.TryGetValue(Eos, out int[]? e) ? e : [2], phonemeIdMap.TryGetValue(Pad, out int[]? p) ? p : [0]);
    }

    /// <summary>Parses the full Piper voice config JSON (<c>*.onnx.json</c>) and extracts its <c>phoneme_id_map</c>.</summary>
    public static PhonemeIdMap FromPiperConfig(Stream onnxJson)
    {
        using JsonDocument doc = JsonDocument.Parse(onnxJson);
        JsonElement map = doc.RootElement.GetProperty("phoneme_id_map");
        Dictionary<string, int[]> parsed = new();
        foreach (JsonProperty prop in map.EnumerateObject())
        {
            int[] ids = new int[prop.Value.GetArrayLength()];
            int i = 0;
            foreach (JsonElement v in prop.Value.EnumerateArray()) ids[i++] = v.GetInt32();
            parsed[prop.Name] = ids;
        }
        return FromPhonemeMap(parsed);
    }

    /// <summary>Encodes an IPA phoneme string to model token ids (piper interleaving); codepoints absent from the map are skipped, exactly as piper-phonemize does.</summary>
    public int[] Encode(string ipa)
    {
        List<int> ids = new(ipa.Length * 2 + 4);
        ids.AddRange(_bos);
        ids.AddRange(_pad);
        foreach (Rune rune in ipa.EnumerateRunes())
        {
            if (!_byCodepoint.TryGetValue(rune.Value, out int[]? mapped)) continue;
            ids.AddRange(mapped);
            ids.AddRange(_pad);
        }
        ids.AddRange(_eos);
        return ids.ToArray();
    }
}

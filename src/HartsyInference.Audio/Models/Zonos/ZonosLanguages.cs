namespace HartsyInference.Audio.Models.Zonos;

/// <summary>The 109 espeak language codes Zonos supports, in checkpoint order — the IntegerConditioner
/// <c>language_id</c> is the index into this list (see <c>zonos/conditioning.py::supported_language_codes</c>).</summary>
public static class ZonosLanguages
{
    public static readonly IReadOnlyList<string> Codes =
    [
        "af", "am", "an", "ar", "as", "az", "ba", "bg", "bn", "bpy", "bs", "ca", "cmn", "cs", "cy", "da", "de", "el",
        "en-029", "en-gb", "en-gb-scotland", "en-gb-x-gbclan", "en-gb-x-gbcwmd", "en-gb-x-rp", "en-us", "eo", "es",
        "es-419", "et", "eu", "fa", "fa-latn", "fi", "fr-be", "fr-ch", "fr-fr", "ga", "gd", "gn", "grc", "gu", "hak",
        "hi", "hr", "ht", "hu", "hy", "hyw", "ia", "id", "is", "it", "ja", "jbo", "ka", "kk", "kl", "kn", "ko", "kok",
        "ku", "ky", "la", "lfn", "lt", "lv", "mi", "mk", "ml", "mr", "ms", "mt", "my", "nb", "nci", "ne", "nl", "om",
        "or", "pa", "pap", "pl", "pt", "pt-br", "py", "quc", "ro", "ru", "ru-lv", "sd", "shn", "si", "sk", "sl", "sq",
        "sr", "sv", "sw", "ta", "te", "tn", "tr", "tt", "ur", "uz", "vi", "vi-vn-x-central", "vi-vn-x-south", "yue",
    ];

    /// <summary>Language id for <c>en-us</c> (24).</summary>
    public static readonly int DefaultId = 24;

    /// <summary>Resolves an espeak code to its <c>language_id</c>, falling back to en-us for unknown codes.</summary>
    public static int Resolve(string code)
    {
        for (int i = 0; i < Codes.Count; i++) if (Codes[i] == code) return i;
        return DefaultId;
    }
}

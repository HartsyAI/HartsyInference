using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Models.Kokoro;
using HartsyInference.Audio.Phonemizer.Espeak;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Kokoro's text front-ends, one per language, picked as the reference <c>KPipeline</c> picks them: by the
/// voice name's first letter. <c>a</c> American and <c>b</c> British English read through misaki's lexicon (us_ or
/// gb_ dictionaries) with espeak and, for American, the CMU dictionary behind it; <c>e</c> Spanish, <c>f</c> French,
/// <c>h</c> Hindi, <c>i</c> Italian and <c>p</c> Brazilian Portuguese through espeak-ng as misaki's
/// <c>EspeakG2P</c> uses it; <c>j</c> Japanese through misaki's <c>JAG2P</c> (MeCab over UniDic 3.1.0) and <c>z</c>
/// Mandarin through misaki's <c>ZHG2P</c> (jieba, pypinyin and cn2an). Each is built on first use and kept.</summary>
internal sealed class KokoroFrontend
{
    /// <summary>Public-domain CMU Pronouncing Dictionary — the American fallback after misaki and espeak.</summary>
    private const string CmudictUrl = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict";

    /// <summary>misaki's English dictionaries (Apache-2.0), pinned to one commit. Name and SHA-256 of each.</summary>
    private const string MisakiDataUrl = "https://raw.githubusercontent.com/hexgrad/misaki/fba1236595f2d2bf21d414ba6e57d25256afada3/misaki/data/";
    private static readonly Dictionary<string, string> MisakiSha256 = new()
    {
        ["us_gold.json"] = "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f",
        ["us_silver.json"] = "de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475",
        ["gb_gold.json"] = "29e62f4b60261c88f7f3c2c7811ca3825978948090b72d2b27d565b729282f71",
        ["gb_silver.json"] = "48131e2d92ccc41655f4543e87e0f938e71463eb5a54be7f0693bb712ebb6bce",
    };

    /// <summary>The espeak-ng language each espeak-read Kokoro language code uses (KPipeline's <c>LANG_CODES</c>).</summary>
    private static readonly Dictionary<char, string> EspeakLanguages = new()
    {
        ['e'] = "es", ['f'] = "fr-fr", ['h'] = "hi", ['i'] = "it", ['p'] = "pt-br",
    };

    private readonly Dictionary<char, Func<string, string>> _frontends = [];
    private readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>The language code of <paramref name="voice"/>: its first letter, lower-cased. For a blend
    /// (<see cref="KokoroVoiceMix"/>) the first voice's.</summary>
    public static char LanguageOf(string voice)
    {
        string first = voice.TrimStart();
        return first.Length == 0 ? 'a' : char.ToLowerInvariant(first[0]);
    }

    /// <summary>The phonemizer for the language <paramref name="voice"/> speaks, built on first use.</summary>
    /// <exception cref="HartsyInferenceException">The voice's language has no front-end here.</exception>
    public async Task<Func<string, string>> ForVoiceAsync(string voice, CancellationToken cancel)
    {
        char lang = LanguageOf(voice);
        lock (_frontends)
        {
            if (_frontends.TryGetValue(lang, out Func<string, string>? cached)) return cached;
        }
        await _gate.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            lock (_frontends)
            {
                if (_frontends.TryGetValue(lang, out Func<string, string>? cached)) return cached;
            }
            Func<string, string> frontend = await LoadAsync(lang, voice, cancel).ConfigureAwait(false);
            lock (_frontends) _frontends[lang] = frontend;
            return frontend;
        }
        finally
        {
            _gate.Release();
        }
    }

    private static async Task<Func<string, string>> LoadAsync(char lang, string voice, CancellationToken cancel)
    {
        if (lang is 'a' or 'b')
        {
            EnglishG2P g2p = await LoadEnglishAsync(british: lang == 'b', cancel).ConfigureAwait(false);
            return g2p.ToIpa;
        }
        if (EspeakLanguages.TryGetValue(lang, out string? language))
        {
            string data = await EspeakDataInstaller.EnsureAsync(cancel).ConfigureAwait(false);
            KokoroEspeakG2P g2p = new(EspeakPhonemizer.FromDataDirectory(data, language));
            return g2p.ToIpa;
        }
        if (lang == 'j')
        {
            KokoroJapaneseG2P g2p = await KokoroJapaneseAssets.LoadAsync(cancel).ConfigureAwait(false);
            return g2p.ToIpa;
        }
        if (lang == 'z')
        {
            KokoroMandarinG2P g2p = await KokoroMandarinAssets.LoadAsync(cancel).ConfigureAwait(false);
            return g2p.ToIpa;
        }
        throw new HartsyInferenceException($"Kokoro voice '{voice}' speaks a language ('{lang}') this engine has no "
            + "front-end for. Use a voice starting with a, b, e, f, h, i, j, p or z.");
    }

    /// <summary>misaki's lexicon for American (us_) or British (gb_) English, with espeak (en-us or en-gb) behind it
    /// and, for American, the CMU dictionary. Each file is fetched once into the shared audio folder.</summary>
    private static async Task<EnglishG2P> LoadEnglishAsync(bool british, CancellationToken cancel)
    {
        string prefix = british ? "gb_" : "us_";
        string[] lexicon = new string[2];
        string[] names = [prefix + "gold.json", prefix + "silver.json"];
        for (int i = 0; i < names.Length; i++)
        {
            lexicon[i] = AudioModelRoot.SharedFile("misaki_" + names[i]);
            if (!File.Exists(lexicon[i]))
            {
                Logs.Info($"[Audio][Kokoro] Downloading the misaki pronunciation dictionary ({names[i]})...");
                await AudioFileFetcher.EnsureAsync(MisakiDataUrl + names[i], lexicon[i], MisakiSha256[names[i]], cancel)
                    .ConfigureAwait(false);
            }
        }
        string? cmudict = null;
        if (!british)
        {
            cmudict = AudioModelRoot.SharedFile("cmudict.dict");
            if (!File.Exists(cmudict))
            {
                Logs.Info("[Audio][Kokoro] Downloading the public-domain CMU Pronouncing Dictionary (cmudict.dict)...");
                await AudioFileFetcher.EnsureAsync(CmudictUrl, cmudict, cancel).ConfigureAwait(false);
            }
        }
        EspeakPhonemizer? espeak = null;
        try
        {
            string data = await EspeakDataInstaller.EnsureAsync(cancel).ConfigureAwait(false);
            espeak = EspeakPhonemizer.FromDataDirectory(data, british ? "en-gb" : "en-us");
        }
        catch (Exception ex) when (ex is HartsyInferenceException or HttpRequestException or IOException or InvalidDataException
            or UnauthorizedAccessException)
        {
            Logs.Warning($"[Audio][Kokoro] espeak-ng data unavailable ({ex.Message}); words outside misaki's dictionary "
                + (british ? "fall back to letter rules." : "fall back to the CMU dictionary."));
        }
        return new EnglishG2P(MisakiLexicon.FromFiles(lexicon[0], lexicon[1], british), cmudict, espeak);
    }
}

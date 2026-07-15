using System.Linq;
using System.Text;
using HartsyInference.Tokenizers;

namespace HartsyInference.Audio.Frontends;

/// <summary>Text front-ends for the token-based audio pipelines (which take token ids, not raw text):
/// composes the engine tokenizers into each pipeline's id stream. Returns the RAW text ids only — the
/// pipeline adds its own control/special tokens.</summary>
public static class AudioTextFrontend
{
    /// <summary>Llama-3 byte-level BPE, shared by Orpheus / CSM / FishSpeech. Built from the embedded canonical
    /// <c>tokenizer.json</c> via <see cref="HfTokenizerJson"/> (reusing the engine's own BPE core), so the
    /// family-specific split regex + <c>ignore_merges</c> reproduce HF token ids exactly.</summary>
    private static readonly Lazy<GgufTokenizer> _llama = new(() =>
    {
        using Stream json = EmbeddedTokenizerResources.OpenLlama3TokenizerJson();
        return HfTokenizerJson.LoadByteLevelBpe(json);
    });

    private static GgufTokenizer RequireLlama()
    {
        if (!EmbeddedTokenizerResources.HasLlama3TokenizerJson)
        {
            throw new InvalidOperationException(
                "The Llama-3 tokenizer.json is not embedded in HartsyInference.Tokenizers, so the Llama-family "
                + "audio front-ends (Orpheus, CSM, FishSpeech) can't tokenize text. Drop a Llama-3.x tokenizer.json "
                + "in HartsyInference.Tokenizers/Resources/ as llama3_tokenizer.json, then rebuild.");
        }
        return _llama.Value;
    }

    /// <summary>Dia takes raw UTF-8 bytes (0–255), one int per byte. Inline speaker tags <c>[S1]</c>/<c>[S2]</c>
    /// become bytes 0x01/0x02 (upstream <c>_encode_text</c>); there is no trained tokenizer.</summary>
    public static int[] DiaBytes(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        byte[] utf8 = Encoding.UTF8.GetBytes(text.Replace("[S1]", "\u0001").Replace("[S2]", "\u0002"));
        int[] ids = new int[utf8.Length];
        for (int i = 0; i < utf8.Length; i++)
        {
            ids[i] = utf8[i];
        }
        return ids;
    }

    /// <summary>Orpheus: Llama-3 BPE of <c>"{voice}: {text}"</c> (the upstream Orpheus prompt format).
    /// The pipeline adds the human/text control tokens. Default voice <c>tara</c> (others: leah, jess,
    /// leo, dan, mia, zac, zoe). Pass an empty voice to tokenize the bare text.</summary>
    public static int[] OrpheusText(string text, string voice = "tara")
    {
        ArgumentNullException.ThrowIfNull(text);
        string prompt = string.IsNullOrWhiteSpace(voice) ? text : $"{voice}: {text}";
        int[] ids = RequireLlama().EncodeOrdinary(prompt);
        // The reference tokenizes with add_special_tokens=True → a leading BOS. The pipeline then wraps with
        // [StartOfHuman] … [EndOfText, EndOfHuman, StartOfAi, StartOfSpeech]. Omitting the BOS leaves the model
        // unconditioned and it produces gibberish speech (verified vs the transformers reference).
        int[] outp = new int[ids.Length + 1];
        outp[0] = Llama3Bos;
        Array.Copy(ids, 0, outp, 1, ids.Length);
        return outp;
    }

    /// <summary>CSM (Sesame): plain Llama-3 BPE of the text. Multi-turn conversation history, when used,
    /// is prepended by the caller as separate segments — this handles the single-utterance text.</summary>
    public static int[] CsmText(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return RequireLlama().EncodeOrdinary(text);
    }

    /// <summary>Qwen2.5/3 byte-level BPE with the exact HF split regex (from the embedded canonical
    /// <c>tokenizer.json</c>) — used by ACE-Step 1.5 conditioning. The two-file vocab+merges Qwen3Tokenizer
    /// misses family merges (e.g. ":\n\n", "/A") and is NOT id-exact for template text.</summary>
    private static readonly Lazy<GgufTokenizer> _qwen3 = new(() =>
    {
        using Stream json = EmbeddedTokenizerResources.OpenQwen3TokenizerJson();
        return HfTokenizerJson.LoadByteLevelBpe(json);
    });

    /// <summary>Raw Qwen2.5/3 BPE ids of <paramref name="text"/> (no specials appended), HF-exact.</summary>
    public static int[] Qwen3Ids(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (!EmbeddedTokenizerResources.HasQwen3TokenizerJson)
        {
            throw new InvalidOperationException(
                "The Qwen tokenizer.json is not embedded in HartsyInference.Tokenizers. Drop a Qwen2.5/3 "
                + "tokenizer.json in HartsyInference.Tokenizers/Resources/ as qwen3_tokenizer.json, then rebuild.");
        }
        return _qwen3.Value.EncodeOrdinary(text);
    }

    /// <summary>Decodes base-vocab Qwen BPE ids back to text (specials are the caller's problem).</summary>
    public static string Qwen3Decode(IReadOnlyList<int> ids) => _qwen3.Value.Decode(ids);

    private const int Llama3Bos = 128000;
    private const int Llama3Eos = 128001;

    /// <summary>HeartMuLa tags section (upstream <c>preprocess</c>): lowercase, wrap in
    /// <c>&lt;tag&gt;…&lt;/tag&gt;</c> (plain text, not a special token), Llama-3 BPE, then BOS/EOS-wrap.</summary>
    public static int[] HeartMulaTags(string tags)
    {
        ArgumentNullException.ThrowIfNull(tags);
        string s = tags.ToLowerInvariant();
        if (!s.StartsWith("<tag>", StringComparison.Ordinal)) s = "<tag>" + s;
        if (!s.EndsWith("</tag>", StringComparison.Ordinal)) s += "</tag>";
        return BosEosWrap(RequireLlama().EncodeOrdinary(s));
    }

    /// <summary>HeartMuLa lyrics section (upstream <c>preprocess</c>): lowercase, Llama-3 BPE, BOS/EOS-wrap.
    /// Structure markers like <c>[verse]</c>/<c>[chorus]</c> stay inline in <paramref name="lyrics"/>.</summary>
    public static int[] HeartMulaLyrics(string lyrics)
    {
        ArgumentNullException.ThrowIfNull(lyrics);
        return BosEosWrap(RequireLlama().EncodeOrdinary(lyrics.ToLowerInvariant()));
    }

    private static int[] BosEosWrap(int[] ids)
    {
        bool needBos = ids.Length == 0 || ids[0] != Llama3Bos;
        bool needEos = ids.Length == 0 || ids[^1] != Llama3Eos;
        int[] outp = new int[ids.Length + (needBos ? 1 : 0) + (needEos ? 1 : 0)];
        int o = 0;
        if (needBos) outp[o++] = Llama3Bos;
        Array.Copy(ids, 0, outp, o, ids.Length);
        if (needEos) outp[^1] = Llama3Eos;
        return outp;
    }

    /// <summary>Bark: BERT WordPiece ids shifted by <paramref name="textEncodingOffset"/>
    /// (<c>BarkConfig.TextEncodingOffset</c>). The pipeline appends its own semantic-infer token. The caller
    /// supplies the BERT tokenizer (loaded from the model's bert-base-multilingual-cased vocab.txt).</summary>
    public static int[] BarkText(BertWordPieceTokenizer tokenizer, string text, int textEncodingOffset)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        return [.. tokenizer.EncodeRaw(text ?? "").Select(id => id + textEncodingOffset)];
    }
}

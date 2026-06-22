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

    /// <summary>Dia takes raw UTF-8 bytes (0–255), one int per byte. Speaker tags like <c>[S1]</c>/<c>[S2]</c>
    /// are written inline in <paramref name="text"/> by the caller; there is no trained tokenizer.</summary>
    public static int[] DiaBytes(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        byte[] utf8 = Encoding.UTF8.GetBytes(text);
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
        return RequireLlama().EncodeOrdinary(prompt);
    }

    /// <summary>CSM (Sesame): plain Llama-3 BPE of the text. Multi-turn conversation history, when used,
    /// is prepended by the caller as separate segments — this handles the single-utterance text.</summary>
    public static int[] CsmText(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return RequireLlama().EncodeOrdinary(text);
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

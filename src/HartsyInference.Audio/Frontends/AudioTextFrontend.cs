using System.Text;
using HartsyInference.Tokenizers;

namespace HartsyInference.Audio.Frontends;

/// <summary>Text front-ends for the token-based audio pipelines (which take token ids, not raw text):
/// composes the engine tokenizers into each pipeline's id stream. Returns the RAW text ids only — the
/// pipeline adds its own control/special tokens.</summary>
public static class AudioTextFrontend
{
    /// <summary>Llama-3 BPE, shared by Orpheus / CSM / FishSpeech. Asset may be absent — see <see cref="RequireLlama"/>.</summary>
    private static readonly Lazy<LlamaTokenizer> _llama = new(() => new LlamaTokenizer(maxLength: 8192));

    private static LlamaTokenizer RequireLlama()
    {
        if (!EmbeddedTokenizerResources.HasLlama3Assets)
        {
            throw new InvalidOperationException(
                "Llama-3 tokenizer assets are not embedded in HartsyInference.Tokenizers, so the Llama-family "
                + "audio front-ends (Orpheus, CSM, FishSpeech) can't tokenize text. Extract vocab.json + merges.txt "
                + "from a Llama-3 tokenizer.json and drop them in HartsyInference.Tokenizers/Resources/ as "
                + "llama3_vocab.json + llama3_merges.txt, then rebuild.");
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
        return [.. RequireLlama().EncodeRaw(prompt)];
    }

    /// <summary>CSM (Sesame): plain Llama-3 BPE of the text. Multi-turn conversation history, when used,
    /// is prepended by the caller as separate segments — this handles the single-utterance text.</summary>
    public static int[] CsmText(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return [.. RequireLlama().EncodeRaw(text)];
    }
}

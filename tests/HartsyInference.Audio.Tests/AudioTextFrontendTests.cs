using System.Text;
using HartsyInference.Audio.Frontends;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free tests for the token-based TTS text front-ends — they verify the text→token-id
/// mapping each pipeline expects, without needing model weights or a backend.</summary>
public sealed class AudioTextFrontendTests
{
    [Fact]
    public void DiaBytes_AreRawUtf8_WithSpeakerTagsFoldedTo1And2()
    {
        const string text = "[S1] Hello there. [S2] Hi!";
        int[] ids = AudioTextFrontend.DiaBytes(text);
        byte[] expected = Encoding.UTF8.GetBytes("\u0001 Hello there. \u0002 Hi!");

        Assert.Equal(expected.Length, ids.Length);
        Assert.Equal(1, ids[0]);                // [S1] → 0x01 (upstream _encode_text)
        Assert.Contains(2, ids);                // [S2] → 0x02
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], ids[i]);
            Assert.InRange(ids[i], 0, 255); // Dia's text vocab is the 256 byte values
        }
    }

    [Fact]
    public void DiaBytes_Null_Throws()
        => Assert.Throws<ArgumentNullException>(() => AudioTextFrontend.DiaBytes(null!));

    // The Orpheus / CSM front-ends tokenize via the embedded Llama-3 tokenizer.json (HfTokenizerJson →
    // GgufTokenizer). The asset is a conditional ~17 MB embedded resource; if a checkout omits it these
    // skip cleanly (return) rather than fail.
    private static GgufTokenizer? TryLlama()
    {
        if (!EmbeddedTokenizerResources.HasLlama3TokenizerJson) return null;
        using Stream json = EmbeddedTokenizerResources.OpenLlama3TokenizerJson();
        return HfTokenizerJson.LoadByteLevelBpe(json);
    }

    // <|begin_of_text|> / <|end_of_text|> (meta-llama/Llama-3.1 tokenizer_config.json) — mirrors the private
    // constants in AudioTextFrontend so these expectations can state the wrap explicitly.
    private const int Llama3Bos = 128000;
    private const int Llama3Eos = 128001;

    private static int[] Prepend(int token, int[] ids)
    {
        int[] outp = new int[ids.Length + 1];
        outp[0] = token;
        Array.Copy(ids, 0, outp, 1, ids.Length);
        return outp;
    }

    private static int[] BosEosWrap(int[] ids)
    {
        int[] outp = new int[ids.Length + 2];
        outp[0] = Llama3Bos;
        Array.Copy(ids, 0, outp, 1, ids.Length);
        outp[^1] = Llama3Eos;
        return outp;
    }

    [Fact]
    public void OrpheusText_PrependsVoicePrefix_MatchesLlamaBpe()
    {
        GgufTokenizer? llama = TryLlama();
        if (llama is null) return;
        // Reference orpheus_tts._format_prompt tokenizes "{voice}: {text}" with add_special_tokens=True — a
        // leading BOS (AudioTextFrontend.OrpheusText's own citation: omitting it left the model unconditioned).
        int[] expected = Prepend(Llama3Bos, llama.EncodeOrdinary("tara: hello world"));

        int[] ids = AudioTextFrontend.OrpheusText("hello world", "tara");

        Assert.NotEmpty(ids);
        Assert.Equal(expected, ids);
    }

    [Fact]
    public void OrpheusText_EmptyVoice_TokenizesBareText()
    {
        GgufTokenizer? llama = TryLlama();
        if (llama is null) return;
        // An empty voice drops the "{voice}: " prefix, but BOS is unconditional — "bare" means no voice prefix,
        // not no BOS.
        int[] bare = Prepend(Llama3Bos, llama.EncodeOrdinary("hello world"));

        int[] withVoice = AudioTextFrontend.OrpheusText("hello world", "tara");
        int[] noVoice = AudioTextFrontend.OrpheusText("hello world", "");

        Assert.Equal(bare, noVoice);
        Assert.NotEqual(bare, withVoice); // the voice prefix really changes the id stream
    }

    [Fact]
    public void CsmText_WrapsSpeakerTaggedTextWithBosEos()
    {
        GgufTokenizer? llama = TryLlama();
        if (llama is null) return;
        // SesameAILabs/csm generator.py _tokenize_text_segment: f"[{speaker}]{text}", Llama-3 BPE, then the
        // reference tokenizer's own BOS/EOS TemplateProcessing wrap (AudioTextFrontend.CsmText's own citation).
        int[] expected = BosEosWrap(llama.EncodeOrdinary("[0]the quick brown fox"));

        int[] ids = AudioTextFrontend.CsmText("the quick brown fox");

        Assert.NotEmpty(ids);
        Assert.Equal(expected, ids);
    }

    [Fact]
    public void CsmText_SpeakerIdChangesTheIdStream()
    {
        GgufTokenizer? llama = TryLlama();
        if (llama is null) return;
        int[] speaker1Expected = BosEosWrap(llama.EncodeOrdinary("[1]hello"));

        int[] speaker0 = AudioTextFrontend.CsmText("hello", speaker: 0);
        int[] speaker1 = AudioTextFrontend.CsmText("hello", speaker: 1);

        Assert.NotEqual(speaker0, speaker1); // "[0]" vs "[1]" changes the BPE'd prefix, not just a label
        Assert.Equal(speaker1Expected, speaker1); // and speaker 1 is exactly the [1]-tagged, BOS/EOS-wrapped encode
    }
}

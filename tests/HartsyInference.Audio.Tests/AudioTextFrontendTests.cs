using System.Text;
using HartsyInference.Audio.Frontends;
using HartsyInference.Tokenizers;
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

    [Fact]
    public void OrpheusText_PrependsVoicePrefix_MatchesLlamaBpe()
    {
        GgufTokenizer? llama = TryLlama();
        if (llama is null) return;
        int[] expected = llama.EncodeOrdinary("tara: hello world");

        int[] ids = AudioTextFrontend.OrpheusText("hello world", "tara");

        Assert.NotEmpty(ids);
        Assert.Equal(expected, ids);
    }

    [Fact]
    public void OrpheusText_EmptyVoice_TokenizesBareText()
    {
        GgufTokenizer? llama = TryLlama();
        if (llama is null) return;
        int[] bare = llama.EncodeOrdinary("hello world");

        int[] withVoice = AudioTextFrontend.OrpheusText("hello world", "tara");
        int[] noVoice = AudioTextFrontend.OrpheusText("hello world", "");

        Assert.Equal(bare, noVoice);
        Assert.NotEqual(bare, withVoice); // the voice prefix really changes the id stream
    }

    [Fact]
    public void CsmText_IsPlainLlamaBpe()
    {
        GgufTokenizer? llama = TryLlama();
        if (llama is null) return;
        int[] expected = llama.EncodeOrdinary("the quick brown fox");

        int[] ids = AudioTextFrontend.CsmText("the quick brown fox");

        Assert.NotEmpty(ids);
        Assert.Equal(expected, ids);
    }
}

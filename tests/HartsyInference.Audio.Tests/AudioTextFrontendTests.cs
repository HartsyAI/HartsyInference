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
    public void DiaBytes_AreRawUtf8_OneIntPerByte()
    {
        const string text = "[S1] Hello there. [S2] Hi!";
        int[] ids = AudioTextFrontend.DiaBytes(text);
        byte[] expected = Encoding.UTF8.GetBytes(text);

        Assert.Equal(expected.Length, ids.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], ids[i]);
            Assert.InRange(ids[i], 0, 255); // Dia's text vocab is the 256 byte values
        }
    }

    [Fact]
    public void DiaBytes_Null_Throws()
        => Assert.Throws<ArgumentNullException>(() => AudioTextFrontend.DiaBytes(null!));

    // The Llama-3 vocab/merges are a conditional ~5 MB embedded asset absent from a default checkout
    // (EmbeddedTokenizerResources.HasLlama3Assets). These run once llama3_vocab.json + llama3_merges.txt
    // are dropped into HartsyInference.Tokenizers/Resources/.
    private const string LlamaAssetSkip =
        "Pending Llama-3 tokenizer asset (llama3_vocab.json + llama3_merges.txt) in HartsyInference.Tokenizers/Resources.";

    [Fact(Skip = LlamaAssetSkip)]
    public void OrpheusText_PrependsVoicePrefix_MatchesLlamaBpe()
    {
        using LlamaTokenizer llama = new(maxLength: 8192);
        int[] expected = [.. llama.EncodeRaw("tara: hello world")];

        int[] ids = AudioTextFrontend.OrpheusText("hello world", "tara");

        Assert.NotEmpty(ids);
        Assert.Equal(expected, ids);
    }

    [Fact(Skip = LlamaAssetSkip)]
    public void OrpheusText_EmptyVoice_TokenizesBareText()
    {
        using LlamaTokenizer llama = new(maxLength: 8192);
        int[] bare = [.. llama.EncodeRaw("hello world")];

        int[] withVoice = AudioTextFrontend.OrpheusText("hello world", "tara");
        int[] noVoice = AudioTextFrontend.OrpheusText("hello world", "");

        Assert.Equal(bare, noVoice);
        Assert.NotEqual(bare, withVoice); // the voice prefix really changes the id stream
    }

    [Fact(Skip = LlamaAssetSkip)]
    public void CsmText_IsPlainLlamaBpe()
    {
        using LlamaTokenizer llama = new(maxLength: 8192);
        int[] expected = [.. llama.EncodeRaw("the quick brown fox")];

        int[] ids = AudioTextFrontend.CsmText("the quick brown fox");

        Assert.NotEmpty(ids);
        Assert.Equal(expected, ids);
    }
}

using HartsyInference.Audio.Frontends;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Locks the MiniMax Music 3 prompt contract against the reference <c>_clean_caption</c>/<c>_normalize_lyrics</c>.
/// Expected values are the diffusers PR #14456 output for the same inputs — the assembled prompt is a checkpoint
/// contract, so a silent drift here changes the generated audio without failing anything else.</summary>
public class MiniMaxMusic3PromptTests
{
    [Fact]
    public void CleanCaption_PassesPlainProseThrough()
    {
        const string caption = "A warm acoustic pop song with intimate female vocals, fingerpicked guitar, soft piano, "
            + "and a gradual emotional build into a wide final chorus.";
        Assert.Equal(caption, MiniMaxMusic3Prompt.CleanCaption(caption));
    }

    [Fact]
    public void CleanCaption_RewritesMetadataTagsAndStripsMarkdown()
    {
        const string caption = "### Global Metadata\r\n"
            + "- <|bpm 92|> and <|key Eminor|> plus <|instrumental|>\r\n"
            + "* **Genre:** *electro swing*, **very** punchy\r\n"
            + "+ item two\r\n"
            + "---\r\n"
            + "• bullet char\r\n"
            + "    four space indent\r\n"
            + "\r\n"
            + "\r\n"
            + "trailing spaces here   \r\n";
        const string expected = "Global Metadata\n"
            + "bpm is 92 and key is Eminor plus instrumental\n"
            + "Genre: electro swing, very punchy\n"
            + "item two\n"
            + "bullet char\n"
            + "four space indent\n"
            + "trailing spaces here";
        Assert.Equal(expected, MiniMaxMusic3Prompt.CleanCaption(caption));
    }

    [Fact]
    public void NormalizeLyrics_PrependsStartAndLowercasesTags()
    {
        const string lyrics = "[Verse]\nMorning light filtering through the pine\n[Chorus]\nSoftly the world begins to breathe";
        const string expected = "[start]\n[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe";
        Assert.Equal(expected, MiniMaxMusic3Prompt.NormalizeLyrics(lyrics));
    }

    [Fact]
    public void NormalizeLyrics_DropsTextSharingALineWithALeadingTag()
    {
        const string lyrics = "[verse]\nI’m learning how to fill up\nevery space I used to leave,\n"
            + "[pre-chorus] Breathe a little deeper,\nlove is here to heal.\n"
            + "[bass-quartet-rumbles-in]\nEvery heartbeat sounded like,\n"
            + "[interlude]\nYou gotta let love— (Oh love… lift us up…)\n"
            + "[OUTRO]\nWe’re gonna let love stay,";
        const string expected = "[start]\n[verse]\nI’m learning how to fill up\nevery space I used to leave,\n"
            + "[pre-chorus]\nlove is here to heal.\n"
            + "[bass-quartet-rumbles-in]\nEvery heartbeat sounded like,\n"
            + "[interlude]\nYou gotta let love— (Oh love… lift us up…)\n"
            + "[outro]\nWe’re gonna let love stay,";
        Assert.Equal(expected, MiniMaxMusic3Prompt.NormalizeLyrics(lyrics));
    }

    [Fact]
    public void NormalizeLyrics_SplitsConsecutiveTagsAndCaretSeparators()
    {
        const string lyrics = "[intro] [verse] some text on the tag line\nplain line ^ split here\n[Chorus] [Post-Chorus]\nlast line";
        const string expected = "[start]\n[intro]\n[verse]\nplain line\nsplit here\n[chorus]\n[post-chorus]\nlast line";
        Assert.Equal(expected, MiniMaxMusic3Prompt.NormalizeLyrics(lyrics));
    }

    /// <summary>Reference token ids for the README example, from diffusers PR #14456 driving the checkpoint's own
    /// <c>Qwen2Tokenizer</c>. Skips without the checkpoint; the added tokens are not in the engine's embedded Qwen
    /// tokenizer, so this can only run against the model's own <c>tokenizer.json</c>.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void Tokenize_MatchesTheReferenceIds()
    {
        string? root = Environment.GetEnvironmentVariable("HARTSY_MINIMAX_MUSIC3_PATH");
        string tokenizerPath = Path.Combine(root ?? "", "tokenizer", "tokenizer.json");
        if (root is null || !File.Exists(tokenizerPath))
        {
            return;
        }
        using FileStream json = File.OpenRead(tokenizerPath);
        GgufTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(json);

        (int[] conditional, int[] unconditional) = MiniMaxMusic3Prompt.Tokenize(
            tokenizer,
            "A warm acoustic pop song with intimate female vocals, fingerpicked guitar, soft piano, "
            + "and a gradual emotional build into a wide final chorus.",
            "[Verse]\nMorning light filtering through the pine\n[Chorus]\nSoftly the world begins to breathe");

        int[] expected =
        [
            151644, 151671, 32, 8205, 44066, 2420, 5492, 448, 31387, 8778, 46096, 11, 14317, 93499, 16986, 11,
            8413, 26278, 11, 323, 264, 52622, 14269, 1936, 1119, 264, 6884, 1590, 55810, 13, 151672, 151673,
            28463, 921, 58, 4450, 921, 84344, 3100, 29670, 1526, 279, 33597, 198, 58, 6150, 355, 921, 30531,
            398, 279, 1879, 12033, 311, 36297, 151674, 151645, 151669,
        ];
        Assert.Equal(expected, conditional);

        int[] expectedUnconditional = new int[expected.Length];
        expected.CopyTo(expectedUnconditional, 0);
        for (int i = 1; i < expectedUnconditional.Length - 2; i++)
        {
            expectedUnconditional[i] = 151654;
        }
        Assert.Equal(expectedUnconditional, unconditional);
    }

    [Fact]
    public void Build_WrapsBothSectionsInTheCheckpointsSpecialTokens()
    {
        string assembled = MiniMaxMusic3Prompt.Build("acoustic pop", "[Verse]\nla la la");
        Assert.Equal(
            "<|im_start|><|caption_start|>acoustic pop<|caption_end|>"
            + "<|lyrics_start|>[start]\n[verse]\nla la la<|lyrics_end|><|im_end|><|audio_start|>",
            assembled);
    }
}

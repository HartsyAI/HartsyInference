using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.Codecs.NeuCodec;
using HartsyInference.Audio.Models.NeuTts;
using HartsyInference.Tokenizers;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Checkpoint-free tests for NeuTTS Air + NeuCodec: the Qwen2.5-0.5B preset with extended vocab, the
/// speech-token offset, and the single FSQ codebook geometry.</summary>
public sealed class NeuTtsTests
{
    [Fact]
    public void Air_PresetIsQwen25_0_5B_WithExtendedVocab()
    {
        NeuTtsConfig c = NeuTtsConfig.Air;
        Assert.Equal(896, c.Llm.HiddenSize);
        Assert.Equal(24, c.Llm.NumHiddenLayers);
        Assert.Equal(14, c.Llm.NumAttentionHeads);
        Assert.Equal(2, c.Llm.NumKeyValueHeads);          // GQA 7:1
        Assert.Equal(1_000_000f, c.Llm.RopeTheta);
        Assert.True(c.Llm.TieWordEmbeddings);
        Assert.Equal(217_652, c.Llm.VocabSize);           // extended for 65536 speech tokens
        Assert.Equal(151_671, c.SpeechTokenBase);
        Assert.Equal(65_536, c.CodebookSize);
    }

    [Fact]
    public void SpeechTokenRange_FitsInsideVocab()
    {
        NeuTtsConfig c = NeuTtsConfig.Air;
        // The last speech token id must be addressable in the LM vocab.
        Assert.True(c.SpeechTokenBase + c.CodebookSize - 1 < c.Llm.VocabSize);
        Assert.Equal(217_206, c.SpeechTokenBase + c.CodebookSize - 1);   // <|speech_65535|>
    }

    [Fact]
    public void NeuCodec_SingleFsqCodebookIs65536()
    {
        NeuCodecConfig cfg = NeuCodecConfig.Default;
        Assert.Equal(8, cfg.FsqLevels.Count);
        Assert.Equal(65_536, Fsq.VocabSize([.. cfg.FsqLevels]));   // 4^8
        Assert.Equal(65_536, cfg.CodebookSize);
        Assert.Equal(64, cfg.HeadDim);                              // 1024 / 16
        Assert.Equal(1_920, cfg.NFft);                             // hop 480 × 4
    }

    [Fact]
    public void NeuCodecDecoder_ConstructsCleanly()
    {
        NeuCodecDecoder d = new(NeuCodecConfig.Default);
        Assert.Equal(24_000, d.SampleRate);
        Assert.Equal(50, d.FrameRate);
        Assert.Equal(1, d.NCodebooks);
    }

    [Fact]
    public void PhonemeTokenTables_MatchCheckpointLayout()
    {
        // Air: 445 IPA tokens at 217207..217651 — the very top of the 217,652 vocab.
        NeuTtsConfig air = NeuTtsConfig.Air;
        Assert.Equal(445, air.PhonemeTokens.Count);
        Assert.Equal("ɐ", air.PhonemeTokens[0]);                                       // ɐ = 217207
        Assert.Equal("ꜟ", air.PhonemeTokens[^1]);                                      // ꜟ = 217651
        Assert.Equal(air.Llm.VocabSize - 1, air.PhonemeTokenBase + air.PhonemeTokens.Count - 1);
        Assert.True(air.PhonemeTokenBase > air.SpeechTokenBase + air.CodebookSize - 1);     // above speech block

        // Nano: 448 IPA tokens at 193798..194245, inside the 194,256 vocab.
        NeuTtsConfig nano = NeuTtsConfig.Nano;
        Assert.Equal(448, nano.PhonemeTokens.Count);
        Assert.Equal(193_798, nano.PhonemeTokenBase);
        Assert.True(nano.PhonemeTokenBase > nano.SpeechTokenBase + nano.CodebookSize - 1);
        Assert.True(nano.PhonemeTokenBase + nano.PhonemeTokens.Count <= nano.Llm.VocabSize);
    }

    [Fact]
    public void PromptPrefix_MatchesUpstreamTemplateLayout()
    {
        // Upstream _apply_chat_template: enc("user: Convert the text to speech:") + [TEXT_PROMPT_START] +
        // enc(phonemes) + [TEXT_PROMPT_END] + enc("\nassistant:"); Synthesize appends SPEECH_GENERATION_START.
        NeuTtsConfig c = NeuTtsConfig.Air;
        List<string> calls = [];
        int[] ids = NeuTtsPromptBuilder.BuildPromptPrefix(c, s => { calls.Add(s); return [1_000 + calls.Count - 1]; }, "zz");
        Assert.Equal([1_000, c.TextPromptStart, 1_001, c.TextPromptEnd, 1_002], ids);
        Assert.Equal(["user: Convert the text to speech:", "zz", "\nassistant:"], calls);
    }

    [Fact]
    public void PhonemeEncode_MapsIpaAddedTokens_AndBpesTheRest()
    {
        NeuTtsConfig c = NeuTtsConfig.Air;
        List<int> dst = [];
        List<string> segs = [];
        NeuTtsPromptBuilder.EncodePhonemes(c, s => { segs.Add(s); return [7]; }, "həl", dst);
        Assert.Equal([7, 217_216, 7], dst);       // 'h' → BPE, ə → added-token 217216, 'l' → BPE
        Assert.Equal(["h", "l"], segs);           // segments BPE'd independently (leading spaces stay attached)
    }

    [Fact]
    public void PromptPrefix_MatchesHfTokenizer_GroundTruth()
    {
        // Ground truth from the HF fast tokenizer (neuphonic/neutts-air tokenizer.json) for the exact upstream
        // template around espeak IPA for "Hello world, this is a test.".
        using Qwen2Tokenizer tok = new();
        string ipa = "həlˈoʊ wˈɜːld, ðɪs ɪz ɐ tˈɛst.";
        int[] ids = NeuTtsPromptBuilder.BuildPromptPrefix(NeuTtsConfig.Air, tok.EncodeRawByteLevel, ipa);
        int[] expected =
        [
            872, 25, 7169, 279, 1467, 311, 8806, 25,                       // "user: Convert the text to speech:"
            151666,                                                        // <|TEXT_PROMPT_START|>
            71, 217216, 75, 217455, 78, 217265, 289, 217455, 217219, 217463, 507, 11,
            1683, 108, 217233, 82, 220, 217233, 89, 220, 217207, 259, 217455, 217218, 267, 13,
            151667,                                                        // <|TEXT_PROMPT_END|>
            198, 77091, 25,                                                // "\nassistant:"
        ];
        Assert.Equal(expected, ids);
    }
}

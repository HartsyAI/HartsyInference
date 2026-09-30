using System.Text;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Special-token resolution per checkpoint layout, over synthetic vocab files that mirror the real
/// <c>added_tokens.json</c> ids: the English-only <c>*.en</c> releases sit one below the multilingual layout from
/// EOT onward and prompt with SOT alone. Every id below was read from the shipped <c>openai/whisper-small.en</c>
/// and <c>openai/whisper-tiny</c> files.</summary>
public sealed class WhisperTokenizerLayoutTests
{
    private const int EnglishOnlyEot = 50_256;
    private const int MultilingualEot = 50_257;

    [Fact]
    public void EnglishOnly_ResolvesEveryIdFromTheCheckpoint()
    {
        using TempLayout layout = new(EnglishOnlyEot);
        using WhisperTokenizer tok = new(layout.Directory);
        Assert.False(tok.IsMultilingual);
        Assert.Equal(51_864, tok.VocabSize);
        Assert.Equal(50_256, tok.EotId);
        Assert.Equal(50_257, tok.SotId);
        Assert.Equal(50_258, tok.FirstLanguageId);
        Assert.Equal(50_357, tok.TranslateId);
        Assert.Equal(50_358, tok.TranscribeId);
        Assert.Equal(50_361, tok.NoSpeechId);
        Assert.Equal(50_362, tok.NoTimestampsId);
        Assert.Equal(50_363, tok.FirstTimestampId);
        Assert.Equal(51_863, tok.LastTimestampId);
        Assert.Equal(50_258, tok.LanguageId("en"));
        Assert.Equal(50_259, tok.LanguageId("zh-CN"));
    }

    [Fact]
    public void EnglishOnly_PromptIsSotAndNoTimestamps_WhateverTheCallerAsks()
    {
        using TempLayout layout = new(EnglishOnlyEot);
        using WhisperTokenizer tok = new(layout.Directory);
        Assert.Equal([50_257, 50_362], tok.BuildPromptIds());
        Assert.Equal([50_257, 50_362], tok.BuildPromptIds(language: "en", translate: true));
        Assert.Equal([50_257, 50_362], tok.BuildPromptIds(language: null));
        Assert.Equal([50_257], tok.BuildPromptIds(withTimestamps: true));
    }

    [Fact]
    public void EnglishOnly_DecodeDropsItsOwnEot()
    {
        using TempLayout layout = new(EnglishOnlyEot);
        using WhisperTokenizer tok = new(layout.Directory);
        // 50256 is below the multilingual EOT; a fixed threshold would have rendered it as literal text.
        Assert.Equal("ab", tok.Decode([2, 50_256]));
        Assert.Equal(string.Empty, tok.DecodeOne(50_256));
        Assert.Equal("ab<|startoftranscript|>", tok.Decode([2, 50_257], includeSpecial: true));
    }

    [Fact]
    public void Multilingual_ResolvesTheV2Layout()
    {
        using TempLayout layout = new(MultilingualEot);
        using WhisperTokenizer tok = new(layout.Directory);
        Assert.True(tok.IsMultilingual);
        Assert.Equal(51_865, tok.VocabSize);
        Assert.Equal(50_257, tok.EotId);
        Assert.Equal(50_258, tok.SotId);
        Assert.Equal(50_259, tok.FirstLanguageId);
        Assert.Equal(50_358, tok.TranslateId);
        Assert.Equal(50_359, tok.TranscribeId);
        Assert.Equal(50_362, tok.NoSpeechId);
        Assert.Equal(50_363, tok.NoTimestampsId);
        Assert.Equal(50_364, tok.FirstTimestampId);
        Assert.Equal(51_864, tok.LastTimestampId);
        Assert.Equal("ab", tok.Decode([2, 50_257]));
    }

    [Fact]
    public void Multilingual_PromptCarriesLanguageAndTask()
    {
        using TempLayout layout = new(MultilingualEot);
        using WhisperTokenizer tok = new(layout.Directory);
        Assert.Equal([50_258, 50_259, 50_359, 50_363], tok.BuildPromptIds());
        Assert.Equal([50_258, 50_259, 50_358, 50_363], tok.BuildPromptIds(translate: true));
        Assert.Equal([50_258, 50_359, 50_363], tok.BuildPromptIds(language: null));
        Assert.Equal([50_258, 50_260, 50_359], tok.BuildPromptIds(language: "zh", withTimestamps: true));
    }

    [Fact]
    public void NoAddedTokens_ReadsAsTheMultilingualV2Layout()
    {
        using TempLayout layout = new(MultilingualEot, writeAddedTokens: false);
        using WhisperTokenizer tok = new(layout.Directory);
        Assert.True(tok.IsMultilingual);
        Assert.Equal(51_865, tok.VocabSize);
        Assert.Equal(50_257, tok.EotId);
        Assert.Equal([50_258, 50_259, 50_359, 50_363], tok.BuildPromptIds());
        Assert.Equal(50_364, tok.FirstTimestampId);
        Assert.Equal("<|0.02|>", tok.DecodeOne(50_365));
    }

    /// <summary>A temp directory holding a three-token BPE vocab, one merge and the special-token table of one
    /// layout: EOT in <c>vocab.json</c>, everything from SOT onward in <c>added_tokens.json</c> at
    /// <c>eot + 1 + offset</c>, exactly as the HuggingFace files lay them out.</summary>
    private sealed class TempLayout : IDisposable
    {
        public string Directory { get; }

        public TempLayout(int eot, bool writeAddedTokens = true)
        {
            Directory = System.IO.Directory.CreateTempSubdirectory("whisper-layout-").FullName;
            File.WriteAllText(Path.Combine(Directory, "vocab.json"),
                "{\"a\":0,\"b\":1,\"ab\":2,\"<|endoftext|>\":" + eot + "}");
            File.WriteAllText(Path.Combine(Directory, "merges.txt"), "#version: 0.2\na b\n");
            if (!writeAddedTokens) return;

            int sot = eot + 1;
            StringBuilder json = new("{");
            Append(json, "<|startoftranscript|>", sot);
            Append(json, "<|en|>", sot + 1);
            Append(json, "<|zh|>", sot + 2);
            Append(json, "<|translate|>", sot + 100);
            Append(json, "<|transcribe|>", sot + 101);
            Append(json, "<|startoflm|>", sot + 102);
            Append(json, "<|startofprev|>", sot + 103);
            Append(json, "<|nocaptions|>", sot + 104);
            Append(json, "<|notimestamps|>", sot + 105);
            Append(json, "<|0.00|>", sot + 106);
            Append(json, "<|30.00|>", sot + 106 + 1500);
            json.Length--;
            json.Append('}');
            File.WriteAllText(Path.Combine(Directory, "added_tokens.json"), json.ToString());
        }

        private static void Append(StringBuilder json, string token, int id)
            => json.Append('"').Append(token).Append("\":").Append(id).Append(',');

        public void Dispose() => System.IO.Directory.Delete(Directory, recursive: true);
    }
}

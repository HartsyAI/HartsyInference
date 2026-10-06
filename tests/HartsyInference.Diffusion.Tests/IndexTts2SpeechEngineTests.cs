using HartsyInference.Audio.Cache;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Engine wiring of IndexTTS-2: catalog registration, 2.5/2.0 variant resolution to repos and file lists,
/// request-to-job-to-options mapping of the emotion controls, and the catalog rows.</summary>
public sealed class IndexTts2SpeechEngineTests
{
    private static TtsJob Job(Action<TtsJobBuilder>? configure = null)
    {
        TtsJobBuilder b = new();
        configure?.Invoke(b);
        return SpeechService.BuildJob("hello", b.Request, null, null, CancellationToken.None);
    }

    private sealed class TtsJobBuilder
    {
        public SpeechRequest Request { get; set; } = new() { Text = "hello" };
    }

    [Theory]
    [InlineData("indextts2")]
    [InlineData("INDEXTTS2")]
    public void TtsCatalog_ResolvesIndexTts2(string id) => Assert.Same(IndexTts2Model.Descriptor, TtsCatalog.Resolve(id));

    [Theory]
    [InlineData(null, false, "IndexTeam/IndexTTS-2.5")]
    [InlineData("", false, "IndexTeam/IndexTTS-2.5")]
    [InlineData("indextts2", false, "IndexTeam/IndexTTS-2.5")]
    [InlineData("2.5", false, "IndexTeam/IndexTTS-2.5")]
    [InlineData("v2_5", false, "IndexTeam/IndexTTS-2.5")]
    [InlineData("IndexTeam/IndexTTS-2.5", false, "IndexTeam/IndexTTS-2.5")]
    [InlineData("2.0", true, "IndexTeam/IndexTTS-2")]
    [InlineData("v2_0", true, "IndexTeam/IndexTTS-2")]
    [InlineData("IndexTeam/IndexTTS-2", true, "IndexTeam/IndexTTS-2")]
    [InlineData("someone/indextts-2-finetune", false, "someone/indextts-2-finetune")]
    public void Variant_ResolvesVersionAndRepo(string? variant, bool v20, string repo)
    {
        Assert.Equal(v20, IndexTts2Model.IsV2_0(variant));
        Assert.Equal(repo, TtsCatalog.Resolve("indextts2").ResolveRepo(variant ?? ""));
    }

    [Fact]
    public void Files_V25_UseTiktokenAndTheBundledCodec()
    {
        IReadOnlyList<AudioModelFile> files = IndexTts2Model.Files("2.5");
        Assert.Contains(files, f => f.Name == "multilingual_zh_ja_yue_char_del.tiktoken" && f.Repo is null);
        Assert.Contains(files, f => f.Name == "codec.pth" && f.Repo is null);
        Assert.DoesNotContain(files, f => f.Name == "bpe.model");
        Assert.DoesNotContain(files, f => f.Repo == "amphion/MaskGCT");
    }

    [Fact]
    public void Files_V20_UseSentencePieceAndTheMaskGctCodec()
    {
        IReadOnlyList<AudioModelFile> files = IndexTts2Model.Files("2.0");
        Assert.Contains(files, f => f.Name == "bpe.model" && f.Repo is null);
        Assert.Contains(files, f => f.Name == "semantic_codec/model.safetensors" && f.Repo == "amphion/MaskGCT");
        Assert.DoesNotContain(files, f => f.Name == "codec.pth");
        Assert.DoesNotContain(files, f => f.Name.EndsWith(".tiktoken", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("2.5")]
    [InlineData("2.0")]
    public void Files_BothVersions_ShareTheStackAndHaveUniqueNames(string variant)
    {
        IReadOnlyList<AudioModelFile> files = IndexTts2Model.Files(variant);
        Assert.Equal(files.Count, files.Select(f => f.Name).Distinct(StringComparer.Ordinal).Count());
        foreach (string name in new[] { "gpt.pth", "s2mel.pth", "wav2vec2bert_stats.pt", "feat1.pt", "feat2.pt" })
            Assert.Contains(files, f => f.Name == name && f.Repo is null);
        foreach (string qwen in IndexTts2Model.QwenFiles)
            Assert.Contains(files, f => f.Name == $"qwen0.6bemo4-merge/{qwen}" && f.Repo is null);
        Assert.Contains(files, f => f.Repo == "facebook/w2v-bert-2.0");
        Assert.Contains(files, f => f.Repo == "funasr/campplus");
        Assert.Contains(files, f => f.Repo == "nvidia/bigvgan_v2_22khz_80band_256x");
    }

    [Fact]
    public void BuildOptions_Defaults_MatchTheReferenceSampler()
    {
        var o = IndexTts2Model.BuildOptions(Job());
        Assert.Equal(0.8f, o.Temperature);
        Assert.Equal(30, o.TopK);
        Assert.Equal(0.8f, o.TopP);
        Assert.Null(o.EmoVector);
        Assert.Null(o.EmoAudioReference);
        Assert.False(o.UseEmoText);
        Assert.Equal(1.0f, o.EmoAlpha);
    }

    [Fact]
    public void BuildOptions_MapsTheEmotionVectorInIndexTts2Order()
    {
        var o = IndexTts2Model.BuildOptions(Job(b => b.Request = b.Request with { Emotion = [0.9, 0, 0.2, 0, 0, 0, 0, 0.5], EmotionAlpha = 0.6 }));
        Assert.Equal(new[] { 0.9f, 0f, 0.2f, 0f, 0f, 0f, 0f, 0.5f }, o.EmoVector);
        Assert.Equal(0.6f, o.EmoAlpha);
    }

    [Fact]
    public void BuildOptions_RejectsAnEmotionVectorOfTheWrongLength()
        => Assert.Throws<ArgumentException>(() => IndexTts2Model.BuildOptions(Job(b => b.Request = b.Request with { Emotion = [1, 2, 3] })));

    [Theory]
    [InlineData("furious and shaking", false, true, "furious and shaking")]
    [InlineData(null, true, true, null)]
    [InlineData("  ", false, false, null)]
    [InlineData(null, false, false, null)]
    public void BuildOptions_MapsTextEmotion(string? emotionText, bool fromText, bool useText, string? expectedText)
    {
        var o = IndexTts2Model.BuildOptions(Job(b => b.Request = b.Request with { EmotionText = emotionText, EmotionFromText = fromText }));
        Assert.Equal(useText, o.UseEmoText);
        Assert.Equal(expectedText, o.EmoText);
    }

    [Fact]
    public void BuildJob_CarriesTheEmotionFields()
    {
        AudioClip clip = new() { Data = [1, 2, 3], Format = "wav" };
        TtsJob job = Job(b => b.Request = b.Request with { EmotionReference = clip, EmotionAlpha = 0.4, EmotionText = "calm", EmotionFromText = true });
        Assert.Same(clip, job.EmotionReference);
        Assert.Equal(0.4, job.EmotionAlpha);
        Assert.Equal("calm", job.EmotionText);
        Assert.True(job.EmotionFromText);
    }

    [Fact]
    public void CatalogEntry_ListsBothVersionsInSeparateSubdirs()
    {
        CatalogEntry entry = ModelCatalog.Find("indextts2")!;
        Assert.Equal(Modality.Speech, entry.Modality);
        Assert.True(entry.CliDrivable);
        Assert.Contains(entry.Assets, a => a.Repo == "IndexTeam/IndexTTS-2.5" && a.RepoPath == "codec.pth");
        Assert.Contains(entry.Assets, a => a.Repo == "IndexTeam/IndexTTS-2" && a.RepoPath == "bpe.model");
        Assert.Contains(entry.Assets, a => a.Repo == "amphion/MaskGCT" && a.RepoPath == "semantic_codec/model.safetensors");
        // The two repos ship same-named gpt.pth/s2mel.pth, so they must not share a target directory.
        string gpt25 = entry.Assets.Single(a => a.Repo == "IndexTeam/IndexTTS-2.5" && a.RepoPath == "gpt.pth").TargetSubdir;
        string gpt20 = entry.Assets.Single(a => a.Repo == "IndexTeam/IndexTTS-2" && a.RepoPath == "gpt.pth").TargetSubdir;
        Assert.NotEqual(gpt25, gpt20);
    }
}

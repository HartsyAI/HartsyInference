using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Metadata;
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
    [InlineData(null, false, "IndexTeam/IndexTTS-2.5")]
    [InlineData("v2_5", false, "IndexTeam/IndexTTS-2.5")]
    [InlineData("2.0", true, "IndexTeam/IndexTTS-2")]
    [InlineData("someone/indextts-2-finetune", false, "someone/indextts-2-finetune")]
    [InlineData("me/foo-v2.05", false, "me/foo-v2.05")]
    [InlineData("x/IndexTTS-2.0-fp16", false, "x/IndexTTS-2.0-fp16")]
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

    [Fact]
    public void BuildOptions_MapsTheEmotionVectorInIndexTts2Order()
    {
        IndexTts2Options o = IndexTts2Model.BuildOptions(Job(b => b.Request = b.Request with { Emotion = [0.9, 0, 0.2, 0, 0, 0, 0, 0.5], EmotionAlpha = 0.6 }));
        Assert.Equal(new[] { 0.9f, 0f, 0.2f, 0f, 0f, 0f, 0f, 0.5f }, o.EmoVector);
        Assert.Equal(0.6f, o.EmoAlpha);
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    public void BuildOptions_RejectsOutOfRangeEmotionWeights(double bad)
        => Assert.Throws<ArgumentException>(() => IndexTts2Model.BuildOptions(Job(b => b.Request = b.Request with { Emotion = [bad, 0, 0, 0, 0, 0, 0, 0] })));

    [Fact]
    public void BuildOptions_RejectsAnEmotionVectorOfTheWrongLength()
        => Assert.Throws<ArgumentException>(() => IndexTts2Model.BuildOptions(Job(b => b.Request = b.Request with { Emotion = [1, 2, 3] })));

    [Theory]
    [InlineData("furious and shaking", false, true, "furious and shaking")]
    [InlineData(null, true, true, null)]
    [InlineData("  ", false, false, null)]
    public void BuildOptions_MapsTextEmotion(string? emotionText, bool fromText, bool useText, string? expectedText)
    {
        IndexTts2Options o = IndexTts2Model.BuildOptions(Job(b => b.Request = b.Request with { EmotionText = emotionText, EmotionFromText = fromText }));
        Assert.Equal(useText, o.UseEmoText);
        Assert.Equal(expectedText, o.EmoText);
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

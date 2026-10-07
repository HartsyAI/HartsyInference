using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Metadata;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Engine wiring of AuK: catalog registration, variant to repo/file resolution, request-to-job mapping, instruction framing, the shard-prefix filter and the catalog pins.</summary>
public sealed class AukSpeechEngineTests
{
    [Theory]
    [InlineData("auk")]
    [InlineData("AUK")]
    public void TtsCatalog_ResolvesAuk(string id) => Assert.Same(AukModel.Descriptor, TtsCatalog.Resolve(id));

    [Theory]
    [InlineData("flash", "tencent/AuK-Flash", "auk_flash.safetensors")]
    [InlineData("Flash", "tencent/AuK-Flash", "auk_flash.safetensors")]
    [InlineData("base", "tencent/AuK", "auk_base.safetensors")]
    [InlineData("", "tencent/AuK", "auk_base.safetensors")]
    [InlineData("auk", "tencent/AuK", "auk_base.safetensors")]
    [InlineData("tencent/AuK-Flash", "tencent/AuK-Flash", "auk_flash.safetensors")]
    [InlineData("someone/auk-finetune", "someone/auk-finetune", "auk_base.safetensors")]
    public void Variant_ResolvesRepoAndCheckpoint(string variant, string repo, string file)
    {
        Assert.Equal(repo, TtsCatalog.Resolve("auk").ResolveRepo(variant));
        Assert.Equal(file, AukModel.CheckpointFile(variant));
        Assert.Equal(variant.Contains("flash", StringComparison.OrdinalIgnoreCase), AukModel.IsFlash(variant));
    }

    [Fact]
    public void BuildJob_MapsInstructionAndDuration_AndLeavesThemUnsetByDefault()
    {
        SpeechRequest request = new() { Text = "hello", Instruction = "a calm voice", DurationSeconds = 2.5, RefText = "ref", Seed = 9 };
        TtsJob job = SpeechService.BuildJob(request.Text, request, null, null, CancellationToken.None);
        Assert.Equal("a calm voice", job.Instruction);
        Assert.Equal(2.5, job.DurationSeconds);
        Assert.Equal("ref", job.RefText);
        Assert.Equal(9, job.Seed);

        TtsJob plain = SpeechService.BuildJob("x", new SpeechRequest { Text = "x" }, null, null, CancellationToken.None);
        Assert.Null(plain.Instruction);
        Assert.Null(plain.DurationSeconds);
    }

    [Fact]
    public void BuildInstruction_FramesTheTextPerMode()
    {
        Assert.Equal("Say the following with the same voice: \"hi\"", AukModel.BuildInstruction("hi", null, hasReference: true));
        Assert.Equal("Remove the noise", AukModel.BuildInstruction("ignored", "Remove the noise", hasReference: true));
        Assert.Equal("Generate speech based on the following description: \"deep voice\". The content to speak is: \"hi\".",
            AukModel.BuildInstruction("hi", "deep voice", hasReference: false));
        Assert.Equal("deep voice", AukModel.BuildInstruction(" ", "deep voice", hasReference: false));
        Assert.Throws<InvalidOperationException>(() => AukModel.BuildInstruction("hi", null, hasReference: false));
    }

    [Fact]
    public void ShardIndex_FilteredByKeyPrefix_NamesOnlyTheShardsHoldingThem()
    {
        string path = Path.Combine(Path.GetTempPath(), $"auk_index_{Guid.NewGuid():N}.json");
        try
        {
            File.WriteAllText(path, """
                {"weight_map": {
                  "thinker.visual.a": "s1", "thinker.audio_tower.x": "s1", "thinker.model.embed_tokens.weight": "s1",
                  "thinker.model.layers.20.mlp.up_proj.weight": "s2", "talker.model.x": "s2", "thinker.lm_head.weight": "s2",
                  "token2wav.y": "s3"}}
                """);
            Assert.Equal(["s1", "s2", "s3"], AudioCheckpoints.ReadShardNames(path).Order());
            Assert.Equal(["s1", "s2"], AudioCheckpoints.ReadShardNames(path, AukModel.OmniKeyPrefixes).Order());
            Assert.Equal(["s1"], AudioCheckpoints.ReadShardNames(path, ["thinker.audio_tower."]).Order());
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CatalogEntry_PinsEveryDownloadedFile()
    {
        CatalogEntry entry = ModelCatalog.Find("auk")!;
        Assert.Equal(Modality.Speech, entry.Modality);
        Assert.True(entry.CliDrivable);
        Dictionary<string, ModelAsset> byPath = entry.Assets.ToDictionary(a => a.RepoPath);
        Assert.Equal("29c65c0c6045e8d8fb454019f99c9680508f553fc98fa143feaca3711d0b8614", byPath["auk_base.safetensors"].Sha256);
        Assert.Equal("9b3ec1400a9ebcc87ee540ebfad0aeeb08e4f210149e190b1469fd8eeb977fe4", byPath["auk_flash.safetensors"].Sha256);
        Assert.Equal("0f5857fd0d6b916d161c73fef433e0ccb42484a2dad3aa08e8a51975e29a0645", byPath["vae.safetensors"].Sha256);
        Assert.Equal("349972cebff443030e9e96e1e930d7230979c6961cf046fa29fc580e4bcf49a6", byPath["model-00001-of-00003.safetensors"].Sha256);
        Assert.Equal("b29a76dfefb3aa33a5bd0faa19c681d039a44de57223653ea0e933df728c6d8c", byPath["model-00002-of-00003.safetensors"].Sha256);
        Assert.Equal(AukModel.TokenizerSha256, byPath["tokenizer.json"].Sha256);
        Assert.DoesNotContain(entry.Assets, a => a.RepoPath == "model-00003-of-00003.safetensors");
        Assert.Equal("tencent/AuK-Flash", byPath["auk_flash.safetensors"].Repo);
        Assert.All(entry.Assets.Where(a => a.Repo == "Qwen/Qwen2.5-Omni-3B"), a => Assert.NotNull(a.RepoPath));
        Assert.Contains(entry.Assets, a => a.Role.Contains("Qwen Research License", StringComparison.Ordinal));
    }

    [Fact]
    public void CatalogEntry_PinsTheShardIndexToo()
        => Assert.Equal("5b7629198e2ef80e37612a491d9bfd71639d2f212632d36d8ab086922e74e129",
            ModelCatalog.Find("auk")!.Assets.Single(a => a.RepoPath == "model.safetensors.index.json").Sha256);

    [Theory]
    [InlineData(7, 7)]
    [InlineData(-3, -3)]
    public void ResolveSeed_KeepsAnExplicitSeed(int seed, int expected) => Assert.Equal(expected, AukModel.ResolveSeed(seed, () => 99));

    [Fact]
    public void ResolveSeed_DrawsAFreshNonZeroSeedWhenUnset()
    {
        Assert.Equal(99, AukModel.ResolveSeed(0, () => 99));
        Assert.Equal(1, AukModel.ResolveSeed(0, () => 0));
        HashSet<int> seen = [];
        for (int i = 0; i < 8; i++)
        {
            int seed = AukModel.ResolveSeed(0);
            Assert.NotEqual(0, seed);
            seen.Add(seed);
        }
        Assert.True(seen.Count > 1, "an unset seed must not collapse to one fixed value");
    }

    [Fact]
    public void FlashIgnoredKnobsMessage_NamesWhatWasDiscarded()
    {
        Assert.Null(AukModel.FlashIgnoredKnobsMessage(null, null));
        Assert.Contains("steps=16", AukModel.FlashIgnoredKnobsMessage(16, null), StringComparison.Ordinal);
        string both = AukModel.FlashIgnoredKnobsMessage(8, 2.5)!;
        Assert.Contains("steps=8", both, StringComparison.Ordinal);
        Assert.Contains("cfg=2.5", both, StringComparison.Ordinal);
        Assert.Contains("ignored", both, StringComparison.Ordinal);
    }

    [Fact]
    public void ModelIdentity_IsRegistered() => Assert.NotNull(ModelIdentityCatalog.Find("auk"));
}

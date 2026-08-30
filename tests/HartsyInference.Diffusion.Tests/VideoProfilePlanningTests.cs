using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed class VideoProfilePlanningTests : IDisposable
{
    private readonly string _tempDir;

    public VideoProfilePlanningTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"hartsy-video-plan-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public void Manifest_RecognizesKnownAliasAndRejectsInputMajor()
    {
        Assert.True(VideoProfileManifest.TryGetByHash(
            "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779", out VideoKnownArtifact? refBase));
        Assert.NotNull(refBase);
        Assert.Equal("minimax-h3-ref2va-base-int8-convrot", refBase!.Id);
        Assert.Equal(VideoTaskFamily.Ref2Va, refBase.Task);

        Assert.True(VideoProfileManifest.TryGetByHash(
            "1dfe28c517a937fb9876f0975f224fd6e7ecb8744219f89bb8ba954403e10dc3", out VideoKnownArtifact? rejected));
        Assert.NotNull(rejected);
        Assert.Equal(VideoProfileArtifactRole.Rejected, rejected!.Role);
        Assert.Contains("input-major", rejected.RejectionReason, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a", "Main")]
    [InlineData("9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779", "Main")]
    [InlineData("12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993", "Main")]
    [InlineData("f86f2f79ebd2d76eb8eeb46091e83982e6ff51d255747e7b16e92834b392b8e9", "Main")]
    [InlineData("9ad5c98b533894c122050d32804a14f49fca8edc16c52564a281cdc5825ac934", "Main")]
    [InlineData("e64cef63bc2785bcd72e6103c52aa78c6cd2c4f9870a7ce79675083fd65cf2e7", "Main")]
    [InlineData("e0441d26414f6e0c28f43d580e6cc56fad424da0fa4d261b698ca73188aa6332", "Main")]
    [InlineData("497c0ff6377eb239d8b446c991a52a69e0817447d92c4525bd85ff8b449fcbaa", "Main")]
    [InlineData("5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c", "Adapter")]
    [InlineData("9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99", "Adapter")]
    [InlineData("08cfe946033af7d27719b964b6e0a0e50c32138daabbd6ce4137e23df6bf9980", "Adapter")]
    [InlineData("9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4", "Adapter")]
    [InlineData("2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e", "Adapter")]
    [InlineData("e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f", "Adapter")]
    [InlineData("c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc", "Adapter")]
    [InlineData("1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c", "Adapter")]
    [InlineData("449d80f301ac571622c72e28b8fd72a4b3681b7a8df8a92f17c8f6ec43f56558", "Adapter")]
    [InlineData("b5e25a59292d51bca3fc02b9a0b2284e11b4eb20921a9c5adc2db785956b8966", "Adapter")]
    [InlineData("0b29be7042d883970eb0c20774a9ba03d95669ed80a721bb4d21be8ea0d0a196", "Adapter")]
    [InlineData("111c82e669f6e20e628228172edf39395f1a9fc3ad049793895e542c0f55b18c", "Adapter")]
    [InlineData("7221ae65d78780354d51e5048d29728d9f1f8fb9baf50b1dd3df85f5101413d3", "Main")]
    [InlineData("919a48acb525dc8fc70287fcd94ec1f5e5e289a77f1df14d01099c6ce204eb02", "ControlNet")]
    [InlineData("9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410", "VideoVae")]
    public void Manifest_ContainsEveryPublishedContractHash(string hash, string expectedRole)
    {
        Assert.True(VideoProfileManifest.TryGetByHash(hash, out VideoKnownArtifact? artifact));
        Assert.NotNull(artifact);
        Assert.Equal(expectedRole, artifact!.Role.ToString());
    }

    [Fact]
    public void Manifest_LightXComfyAndDiffusersAliasesShareTheEffectiveProfile()
    {
        Assert.True(VideoProfileManifest.TryGetByHash(
            "08cfe946033af7d27719b964b6e0a0e50c32138daabbd6ce4137e23df6bf9980",
            out VideoKnownArtifact? comfy));
        Assert.True(VideoProfileManifest.TryGetByHash(
            "9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4",
            out VideoKnownArtifact? diffusers));

        Assert.NotNull(comfy);
        Assert.NotNull(diffusers);
        Assert.Equal(comfy!.Id, diffusers!.Id);
        Assert.Equal(comfy.Task, diffusers.Task);
        Assert.Equal(comfy.Steps, diffusers.Steps);
        Assert.Equal(comfy.FlowShift, diffusers.FlowShift);
        Assert.Equal(comfy.AudioFlowShift, diffusers.AudioFlowShift);
    }

    [Fact]
    public void ComponentFormat_DetectsNvfp4AwqBeforeItsInt8EmbeddingFallback()
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = new(StringComparer.Ordinal)
        {
            ["model.layers.0.self_attn.q_proj.weight"] = Descriptor(
                "model.layers.0.self_attn.q_proj.weight", DType.U8, 16, 16),
            ["model.layers.0.self_attn.q_proj.weight_scale_2"] = Descriptor(
                "model.layers.0.self_attn.q_proj.weight_scale_2", DType.F32, 1),
            ["model.layers.0.self_attn.o_proj.pre_quant_scale"] = Descriptor(
                "model.layers.0.self_attn.o_proj.pre_quant_scale", DType.F32, 16),
            ["model.embed_tokens.weight"] = Descriptor("model.embed_tokens.weight", DType.I8, 16, 16),
        };

        Assert.Equal("nvfp4-awq", VideoProfileResolver.DetectFormat(descriptors));

        static SafeTensorDescriptor Descriptor(string name, DType dtype, params long[] dimensions) => new()
        {
            Name = name,
            DType = dtype,
            Shape = new TensorShape(dimensions),
            DataOffset = 0,
            ByteLength = dtype.ComputeByteCount(dimensions.Aggregate(1L, (product, value) => product * value)),
        };
    }

    [Fact]
    public async Task HashCache_KeysByCanonicalPathSizeAndModificationTime()
    {
        string path = Path.Combine(_tempDir, "hash.bin");
        byte[] first = Encoding.UTF8.GetBytes("first-payload");
        byte[] second = Encoding.UTF8.GetBytes("other-payload");
        Assert.Equal(first.Length, second.Length);
        await File.WriteAllBytesAsync(path, first);
        DateTime stamp = File.GetLastWriteTimeUtc(path);

        string expected = Convert.ToHexString(SHA256.HashData(first)).ToLowerInvariant();
        string firstHash = await VideoCheckpointHashCache.GetSha256Async(path, CancellationToken.None);
        Assert.Equal(expected, firstHash);

        await File.WriteAllBytesAsync(path, second);
        File.SetLastWriteTimeUtc(path, stamp);
        string cached = await VideoCheckpointHashCache.GetSha256Async(path, CancellationToken.None);
        Assert.Equal(firstHash, cached);

        File.SetLastWriteTimeUtc(path, stamp.AddSeconds(2));
        string changed = await VideoCheckpointHashCache.GetSha256Async(path, CancellationToken.None);
        Assert.NotEqual(firstHash, changed);
        Assert.Equal(1, VideoCheckpointHashCache.StateCountFor(path));
    }

    [Fact]
    public async Task HashCache_PersistsExactHashAcrossProcessStateReset()
    {
        string path = Path.Combine(_tempDir, "persistent-hash.bin");
        byte[] data = Encoding.UTF8.GetBytes("one exact immutable checkpoint state");
        await File.WriteAllBytesAsync(path, data);
        VideoCheckpointHashCache.RemovePersistent(path);
        VideoCheckpointHashCache.Clear();
        int before = VideoCheckpointHashCache.ComputeCountFor(path);

        string first = await VideoCheckpointHashCache.GetSha256Async(path, CancellationToken.None);
        int afterFirst = VideoCheckpointHashCache.ComputeCountFor(path);
        Assert.Equal(before + 1, afterFirst);

        // Dropping the static dictionary models a fresh process while leaving its durable metadata intact.
        VideoCheckpointHashCache.Clear();
        string second = await VideoCheckpointHashCache.GetSha256Async(path, CancellationToken.None);

        Assert.Equal(first, second);
        Assert.Equal(afterFirst, VideoCheckpointHashCache.ComputeCountFor(path));
    }

    [Fact]
    public async Task HashCache_DownloaderSeedIsTrustedOnlyForTheMatchingFileStamp()
    {
        string path = Path.Combine(_tempDir, "download-seed.bin");
        byte[] firstData = Encoding.UTF8.GetBytes("first-payload");
        byte[] secondData = Encoding.UTF8.GetBytes("other-payload");
        Assert.Equal(firstData.Length, secondData.Length);
        await File.WriteAllBytesAsync(path, firstData);
        DateTime firstStamp = File.GetLastWriteTimeUtc(path);
        string verified = Convert.ToHexString(SHA256.HashData(firstData)).ToLowerInvariant();
        await VideoCheckpointHashCache.RecordVerifiedSha256Async(path, verified);
        VideoCheckpointHashCache.Clear();
        int before = VideoCheckpointHashCache.ComputeCountFor(path);

        Assert.Equal(verified,
            await VideoCheckpointHashCache.GetSha256Async(path, CancellationToken.None));
        Assert.Equal(before, VideoCheckpointHashCache.ComputeCountFor(path));

        await File.WriteAllBytesAsync(path, secondData);
        File.SetLastWriteTimeUtc(path, firstStamp.AddSeconds(2));
        VideoCheckpointHashCache.Clear();
        string changed = await VideoCheckpointHashCache.GetSha256Async(path, CancellationToken.None);

        Assert.Equal(Convert.ToHexString(SHA256.HashData(secondData)).ToLowerInvariant(), changed);
        Assert.NotEqual(verified, changed);
        Assert.Equal(before + 1, VideoCheckpointHashCache.ComputeCountFor(path));
    }

    [Fact]
    public async Task HashCache_DeduplicatesAliasesThroughSymlinkedModelDirectories()
    {
        if (OperatingSystem.IsWindows())
        {
            // Windows runners do not consistently grant the symbolic-link privilege required by this test.
            return;
        }

        string models = Path.Combine(_tempDir, "models");
        string firstAlias = Path.Combine(_tempDir, "models-alias-one");
        string secondAlias = Path.Combine(_tempDir, "models-alias-two");
        Directory.CreateDirectory(models);
        string checkpoint = Path.Combine(models, "control.safetensors");
        await File.WriteAllTextAsync(checkpoint, "shared-control-checkpoint");
        Directory.CreateSymbolicLink(firstAlias, models);
        Directory.CreateSymbolicLink(secondAlias, models);
        string firstPath = Path.Combine(firstAlias, Path.GetFileName(checkpoint));
        string secondPath = Path.Combine(secondAlias, Path.GetFileName(checkpoint));

        try
        {
            VideoCheckpointHashCache.Clear();
            VideoCheckpointHashCache.RemovePersistent(firstPath);

            string firstHash = await VideoCheckpointHashCache.GetSha256Async(firstPath, CancellationToken.None);
            string secondHash = await VideoCheckpointHashCache.GetSha256Async(secondPath, CancellationToken.None);

            Assert.Equal(firstHash, secondHash);
            Assert.Equal(1, VideoCheckpointHashCache.Count);
            Assert.True(VideoArtifactPath.Comparer.Equals(
                VideoArtifactPath.Canonicalize(firstPath), VideoArtifactPath.Canonicalize(secondPath)));
            Assert.Equal(ControlCacheKey(firstPath), ControlCacheKey(secondPath));
            Assert.Equal(ControlCacheKey(firstPath), ControlCacheKey(firstPath, secondPath));
        }
        finally
        {
            VideoCheckpointHashCache.Clear();
            VideoCheckpointHashCache.RemovePersistent(firstPath);
            Directory.Delete(firstAlias);
            Directory.Delete(secondAlias);
        }

        static string ControlCacheKey(params string[] paths) => RecipeCacheKey.Describe(new VideoRequest
        {
            Prompt = "test",
            Controls = paths.Select(path => new VideoControl
            {
                Model = path,
                Video = new VideoClip { Data = [1] },
            }).ToArray(),
        });
    }

    [Fact]
    public async Task GenericPlan_ResolvesAudioShiftAndTypedInputIssues()
    {
        ModelSpec spec = new ModelSpec { Requested = "test", Modality = Modality.Video };
        VideoRequest request = new VideoRequest
        {
            Prompt = "test",
            AudioFlowShift = 2.5f,
            Guides =
            [
                new VideoGuide { FrameIndex = 0, Image = Image(8, 8), Video = new VideoClip { Data = [1] } },
            ],
            AudioDenoiseMask = new AudioDenoiseMask { Values = [1f, 0.5f], Rate = 0f },
            Controls =
            [
                new VideoControl
                {
                    Model = "",
                    Video = new VideoClip { Data = [] },
                    Strength = -1,
                    Start = 0.8,
                    End = 0.2,
                    Kind = VideoControlKind.Inpaint,
                },
            ],
        };
        VideoDefaults defaults = new VideoDefaults
        {
            Steps = 10,
            CfgScale = 4f,
            Width = 640,
            Height = 384,
            Frames = 25,
            Fps = 24,
            FlowShift = 1.5f,
            AudioFlowShift = 3f,
            Sampler = "euler",
            Scheduler = "normal",
        };

        VideoPlan plan = await VideoProfileResolver.ResolveAsync(spec, request, "test-video", defaults,
            VideoFeatures.Guides | VideoFeatures.AudioDenoiseMask | VideoFeatures.VideoControlNet
                | VideoFeatures.VideoInpaint, CancellationToken.None);

        Assert.Equal(2.5f, plan.EffectiveSettings.AudioFlowShift);
        Assert.Contains(plan.Issues, issue => issue.Code == "video.guide.visual_xor");
        Assert.Contains(plan.Issues, issue => issue.Code == "audio.mask.rate_invalid");
        Assert.Contains(plan.Issues, issue => issue.Code == "audio.mask.source_required");
        Assert.Contains(plan.Issues, issue => issue.Code == "video.control.model_missing");
        Assert.Contains(plan.Issues, issue => issue.Code == "video.control.window_invalid");
        Assert.Contains(plan.Issues, issue => issue.Code == "video.control.inpaint_payload_missing");
        Assert.False(plan.IsValid);
    }

    [Fact]
    public async Task UnknownTurbo_RequiresHashBoundSidecarAndThenUsesIt()
    {
        string checkpoint = WriteHeaderOnlyH3("community-turbo.safetensors");
        string hash = await VideoCheckpointHashCache.GetSha256Async(checkpoint, CancellationToken.None);
        ModelSpec spec = new ModelSpec
        {
            Requested = checkpoint,
            LocalPath = checkpoint,
            Modality = Modality.Video,
            ProfileId = "community-fl-turbo8",
        };
        VideoRequest request = new VideoRequest { Prompt = "test" };
        VideoDefaults defaults = new VideoDefaults
        {
            Steps = 30,
            CfgScale = 1f,
            Width = 1344,
            Height = 768,
            Frames = 124,
            Fps = 24,
            FlowShift = 12f,
            AudioFlowShift = 3f,
            Sampler = "euler",
            Scheduler = "normal",
        };

        VideoPlan withoutSidecar = await VideoProfileResolver.ResolveAsync(spec, request, "minimax-h3", defaults,
            VideoFeatures.InitImage | VideoFeatures.EndFrame | VideoFeatures.Lora, CancellationToken.None);
        Assert.Contains(withoutSidecar.Issues, issue => issue.Code == "video.profile.sidecar_required");
        Assert.Equal(VideoAccelerationKind.None, withoutSidecar.Profile.Acceleration);

        string sidecar = checkpoint + ".hartsy-video-profile.json";
        string json = JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["sha256"] = hash,
            ["profileId"] = "community-fl-turbo8",
            ["displayName"] = "Community FL Turbo 8",
            ["task"] = "Fl2Va",
            ["acceleration"] = "Turbo",
            ["attention"] = "Dense",
            ["steps"] = 8,
            ["cfgScale"] = 1.0,
            ["flowShift"] = 12.0,
            ["audioFlowShift"] = 3.0,
            ["sampler"] = "euler",
            ["scheduler"] = "normal",
        });
        await File.WriteAllTextAsync(sidecar, json);

        VideoPlan withSidecar = await VideoProfileResolver.ResolveAsync(spec, request, "minimax-h3", defaults,
            VideoFeatures.InitImage | VideoFeatures.EndFrame | VideoFeatures.Lora, CancellationToken.None);
        Assert.Equal("community-fl-turbo8", withSidecar.Profile.Id);
        Assert.Equal(VideoAccelerationKind.Turbo, withSidecar.Profile.Acceleration);
        Assert.Equal(8, withSidecar.EffectiveSettings.Steps);
        Assert.DoesNotContain(withSidecar.Issues, issue => issue.Code == "video.profile.sidecar_required");
    }

    [Fact]
    public async Task VsaGates_RequireAnExactHashBoundSemanticProfile()
    {
        string checkpoint = WriteHeaderOnlyH3("community-gated.safetensors", gateCount: 50);
        string hash = await VideoCheckpointHashCache.GetSha256Async(checkpoint, CancellationToken.None);
        ModelSpec spec = new ModelSpec { Requested = checkpoint, LocalPath = checkpoint, Modality = Modality.Video };
        VideoRequest request = new VideoRequest { Prompt = "test" };

        VideoPlan unbound = await VideoProfileResolver.ResolveAsync(spec, request, "minimax-h3", H3Defaults(),
            VideoFeatures.Lora, CancellationToken.None);
        Assert.Contains(unbound.Issues, issue => issue.Code == "video.vsa.profile_required");

        string json = JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["sha256"] = hash,
            ["profileId"] = "operator-fastvideo-vsa64-v1",
            ["task"] = "T2Va",
            ["acceleration"] = "Vsa",
            ["attention"] = "FastVideoVsa64V1",
            ["steps"] = 4,
            ["cfgScale"] = 1.0,
            ["flowShift"] = 12.0,
            ["audioFlowShift"] = 3.0,
            ["sampler"] = "euler",
            ["scheduler"] = "normal",
        });
        await File.WriteAllTextAsync(checkpoint + ".hartsy-video-profile.json", json);
        VideoPlan bound = await VideoProfileResolver.ResolveAsync(spec, request, "minimax-h3", H3Defaults(),
            VideoFeatures.Lora, CancellationToken.None);
        Assert.Equal(VideoAttentionKind.FastVideoVsa64V1, bound.Profile.Attention);
        Assert.DoesNotContain(bound.Issues, issue => issue.Code is "video.vsa.profile_required" or "video.vsa.gates_required");

        VideoPlan releaseGated = VideoService.ApplyH3VsaReleaseGate(bound);
        VideoPlanIssue issue = Assert.Single(releaseGated.Issues,
            candidate => candidate.Code == "video.vsa.release_blocked");
        Assert.Equal(VideoPlanIssueSeverity.Error, issue.Severity);
        Assert.Contains("published build cannot execute", issue.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ConvertedPdd_IsAcceptedOnlyWhenMetadataBindsSelectedBase()
    {
        string checkpoint = WriteHeaderOnlyH3("pruned-base.safetensors", pruned: true);
        string baseHash = await VideoCheckpointHashCache.GetSha256Async(checkpoint, CancellationToken.None);
        Dictionary<string, string> metadata = new(StringComparer.Ordinal)
        {
            ["hartsy.pdd.format"] = "minimax_h3_pdd_hartsy_pruned_v1",
            ["hartsy.pdd.task"] = "fl2va",
            ["hartsy.pdd.adapter_sha256"] = new string('a', 64),
            ["hartsy.pdd.full_base_sha256"] = new string('b', 64),
            ["hartsy.pdd.target_base_sha256"] = baseHash,
            ["hartsy.pdd.affine_residual"] = "1.0E-05",
            ["hartsy.pdd.converter"] = "HartsyInference.MiniMaxH3PddPrunedConverter/v1",
            ["pdd_num_steps"] = "32",
            ["pdd_block_size"] = "4",
            ["lora_rank"] = "64",
            ["lora_alpha"] = "64",
        };
        string adapter = WriteHeaderOnlyPdd("converted-pdd.safetensors", metadata);
        ModelSpec spec = new ModelSpec { Requested = checkpoint, LocalPath = checkpoint, Modality = Modality.Video };
        VideoRequest request = new VideoRequest
        {
            Prompt = "test",
            Loras = new LoraStack { Entries = [new LoraEntry { Model = adapter }] },
        };

        VideoPlan accepted = await VideoProfileResolver.ResolveAsync(spec, request, "minimax-h3", H3Defaults(),
            VideoFeatures.Lora, CancellationToken.None);
        Assert.Equal(VideoAccelerationKind.Pdd, accepted.Profile.Acceleration);
        Assert.Equal(8, accepted.EffectiveSettings.Steps);
        Assert.True(accepted.ArtifactMetadata.ContainsKey("lora:0"));
        Assert.DoesNotContain(accepted.Issues, issue => issue.Code == "video.pdd.profile_required");

        metadata["hartsy.pdd.target_base_sha256"] = new string('c', 64);
        string mismatch = WriteHeaderOnlyPdd("mismatched-pdd.safetensors", metadata);
        VideoPlan rejected = await VideoProfileResolver.ResolveAsync(spec, request with
        {
            Loras = new LoraStack { Entries = [new LoraEntry { Model = mismatch }] },
        }, "minimax-h3", H3Defaults(), VideoFeatures.Lora, CancellationToken.None);
        Assert.Contains(rejected.Issues, issue => issue.Code == "video.pdd.target_hash_mismatch");
        Assert.False(rejected.IsValid);
    }

    [Fact]
    public async Task OrdinaryLora_MustMatchEverySelectedBaseTargetBeforeGenerationStarts()
    {
        string checkpoint = WriteHeaderOnlyH3("ordinary-lora-pruned-base.safetensors", pruned: true);
        string baseHash = await VideoCheckpointHashCache.GetSha256Async(checkpoint, CancellationToken.None);
        await WriteFlSidecar(checkpoint, baseHash);
        ModelSpec spec = new ModelSpec { Requested = checkpoint, LocalPath = checkpoint, Modality = Modality.Video };

        string incompatible = WriteHeader("full-width-lora.safetensors", new Dictionary<string, object>
        {
            ["blocks.0.adaln_proj.linear.lora_A.weight"] = Descriptor("F32", [16, 2688]),
            ["blocks.0.adaln_proj.linear.lora_B.weight"] = Descriptor("F32", [96768, 16]),
        });
        VideoPlan rejected = await Plan(incompatible);
        Assert.Contains(rejected.Issues, issue => issue.Code == "video.lora.target_shape_mismatch"
            && issue.Message.Contains("[96768,8]", StringComparison.Ordinal));
        Assert.False(rejected.IsValid);

        string compatible = WriteHeader("curve-form-lora.safetensors", new Dictionary<string, object>
        {
            ["blocks.0.adaln_proj.linear.lora_A.default.weight"] = Descriptor("F32", [16, 8]),
            ["blocks.0.adaln_proj.linear.lora_B.default.weight"] = Descriptor("F32", [96768, 16]),
        });
        VideoPlan accepted = await Plan(compatible);
        Assert.DoesNotContain(accepted.Issues, issue => issue.Code.StartsWith("video.lora.", StringComparison.Ordinal));

        Task<VideoPlan> Plan(string lora) => VideoProfileResolver.ResolveAsync(spec, new VideoRequest
        {
            Prompt = "test",
            Loras = new LoraStack { Entries = [new LoraEntry { Model = lora }] },
        }, "minimax-h3", H3Defaults(), VideoFeatures.Lora, CancellationToken.None);
    }

    [Fact]
    public async Task RebasedFunControl_IsHeaderValidatedDeduplicatedAndBoundToTheExactPrunedBase()
    {
        string checkpoint = WriteHeaderOnlyH3("fun-pruned-base.safetensors", pruned: true);
        string baseHash = await VideoCheckpointHashCache.GetSha256Async(checkpoint, CancellationToken.None);
        await WriteFlSidecar(checkpoint, baseHash);
        Dictionary<string, string> metadata = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["hartsy.controlnet.format"] = "minimax_h3_fun_pruned_v1",
            ["hartsy.controlnet.control_sha256"] =
                "919a48acb525dc8fc70287fcd94ec1f5e5e289a77f1df14d01099c6ce204eb02",
            ["hartsy.controlnet.full_base_sha256"] = new string('a', 64),
            ["hartsy.controlnet.target_base_sha256"] = baseHash,
            ["hartsy.controlnet.affine_residual"] = "1.0E-05",
        };
        string controlDirectory = Path.Combine(_tempDir, "control-models");
        Directory.CreateDirectory(controlDirectory);
        string control = WriteHeaderOnlyFunControl("control-models/rebased-fun.safetensors", metadata, timeDim: 8);
        string firstControlPath = control;
        string secondControlPath = control;
        string? firstAlias = null;
        string? secondAlias = null;
        if (!OperatingSystem.IsWindows())
        {
            firstAlias = Path.Combine(_tempDir, "control-models-one");
            secondAlias = Path.Combine(_tempDir, "control-models-two");
            Directory.CreateSymbolicLink(firstAlias, controlDirectory);
            Directory.CreateSymbolicLink(secondAlias, controlDirectory);
            firstControlPath = Path.Combine(firstAlias, Path.GetFileName(control));
            secondControlPath = Path.Combine(secondAlias, Path.GetFileName(control));
        }
        VideoControl first = new VideoControl
        {
            Model = firstControlPath,
            Video = new VideoClip { Data = [1] },
            Strength = 0.75,
            Start = 0.1,
            End = 0.8,
        };
        VideoControl second = first with
        {
            Model = secondControlPath,
            Strength = 0.25,
            Start = 0.5,
            End = 1.0,
        };
        ModelSpec spec = new ModelSpec
        {
            Requested = checkpoint,
            LocalPath = checkpoint,
            Modality = Modality.Video,
        };
        VideoRequest request = new VideoRequest { Prompt = "test", Controls = [first, second] };

        VideoPlan plan;
        try
        {
            plan = await VideoProfileResolver.ResolveAsync(spec, request, "minimax-h3", H3Defaults(),
                VideoFeatures.VideoControlNet | VideoFeatures.VideoInpaint | VideoFeatures.Lora,
                CancellationToken.None);
        }
        finally
        {
            if (firstAlias is not null && secondAlias is not null)
            {
                Directory.Delete(firstAlias);
                Directory.Delete(secondAlias);
            }
        }

        Assert.DoesNotContain(plan.Issues, issue => issue.Code.StartsWith("video.control.", StringComparison.Ordinal));
        Assert.Single(plan.ComponentPaths,
            pair => pair.Key.StartsWith("controlModel:", StringComparison.Ordinal));
        Assert.True(VideoArtifactPath.Comparer.Equals(
            VideoArtifactPath.Canonicalize(control), plan.ComponentPaths["controlModel:0"]));
        Assert.Equal("h3-fun-pruned-rebased-fp32", plan.ComponentFormats["controlModel:0"]);
        Assert.Equal(baseHash,
            plan.ArtifactMetadata["controlModel:0"]["hartsy.controlnet.target_base_sha256"]);
    }

    [Fact]
    public async Task FunControl_RejectsUnknownArtifactsAndRebaseMetadataForAnotherBase()
    {
        string checkpoint = WriteHeaderOnlyH3("fun-binding-base.safetensors", pruned: true);
        string baseHash = await VideoCheckpointHashCache.GetSha256Async(checkpoint, CancellationToken.None);
        await WriteFlSidecar(checkpoint, baseHash);
        ModelSpec spec = new ModelSpec
        {
            Requested = checkpoint,
            LocalPath = checkpoint,
            Modality = Modality.Video,
        };

        string unknown = WriteHeaderOnlyFunControl(
            "unknown-fun.safetensors", new Dictionary<string, string>(), timeDim: 8);
        VideoPlan unknownPlan = await Plan(unknown);
        Assert.Contains(unknownPlan.Issues, issue => issue.Code == "video.control.unrecognized_artifact");

        Dictionary<string, string> mismatchedMetadata = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["hartsy.controlnet.format"] = "minimax_h3_fun_pruned_v1",
            ["hartsy.controlnet.control_sha256"] =
                "919a48acb525dc8fc70287fcd94ec1f5e5e289a77f1df14d01099c6ce204eb02",
            ["hartsy.controlnet.full_base_sha256"] = new string('a', 64),
            ["hartsy.controlnet.target_base_sha256"] = new string('b', 64),
            ["hartsy.controlnet.affine_residual"] = "1.0E-05",
        };
        string mismatch = WriteHeaderOnlyFunControl("mismatched-fun.safetensors", mismatchedMetadata, timeDim: 8);
        VideoPlan mismatchPlan = await Plan(mismatch);
        Assert.Contains(mismatchPlan.Issues, issue => issue.Code == "video.control.target_hash_mismatch");
        Assert.False(mismatchPlan.IsValid);

        async Task<VideoPlan> Plan(string controlPath)
        {
            VideoRequest request = new VideoRequest
            {
                Prompt = "test",
                Controls =
                [
                    new VideoControl { Model = controlPath, Video = new VideoClip { Data = [1] } },
                ],
            };
            return await VideoProfileResolver.ResolveAsync(spec, request, "minimax-h3", H3Defaults(),
                VideoFeatures.VideoControlNet | VideoFeatures.VideoInpaint | VideoFeatures.Lora,
                CancellationToken.None);
        }
    }

    [Fact]
    public async Task H3HeaderValidation_RejectsTruncatedBlocksAndMalformedQuantCompanions()
    {
        string checkpoint = WriteHeader("truncated-int8-h3.safetensors", new Dictionary<string, object>
        {
            ["video_patch_proj.weight"] = Descriptor("I8", [5376, 96]),
            ["audio_patch_proj.weight"] = Descriptor("F32", [5376, 32]),
            ["blocks.0.attn.qkv_proj.weight"] = Descriptor("F32", [21504, 5376]),
            ["final_layer.video_out.weight"] = Descriptor("F32", [96, 5376]),
            ["final_layer.audio_out.weight"] = Descriptor("F32", [32, 5376]),
        });
        VideoPlan plan = await VideoProfileResolver.ResolveAsync(new ModelSpec
        {
            Requested = checkpoint,
            LocalPath = checkpoint,
            Modality = Modality.Video,
        }, new VideoRequest { Prompt = "test" }, "minimax-h3", H3Defaults(), VideoFeatures.Lora,
            CancellationToken.None);

        Assert.Contains(plan.Issues, issue => issue.Code == "video.checkpoint.tensor_missing"
            && issue.Message.Contains("blocks.49", StringComparison.Ordinal));
        Assert.Contains(plan.Issues, issue => issue.Code == "video.checkpoint.quant_companion_missing");
        Assert.False(plan.IsValid);
    }

    [Fact]
    public async Task ComponentHeaders_AreValidatedByResolvedRole()
    {
        string root = Path.Combine(_tempDir, "component-tree");
        Directory.CreateDirectory(Path.Combine(root, "transformer"));
        Directory.CreateDirectory(Path.Combine(root, "video_vae"));
        Directory.CreateDirectory(Path.Combine(root, "text_encoder"));
        File.Copy(WriteHeaderOnlyH3("component-main.safetensors"),
            Path.Combine(root, "transformer", "model.safetensors"));
        File.Copy(WriteHeader("not-a-video-vae.safetensors", new Dictionary<string, object>
        {
            ["model.embed_tokens.weight"] = Descriptor("F32", [151936, 5120]),
        }), Path.Combine(root, "video_vae", "model.safetensors"));
        File.Copy(WriteHeader("not-a-text-encoder.safetensors", new Dictionary<string, object>
        {
            ["decoder.x_embedder.weight"] = Descriptor("F32", [2048, 24]),
        }), Path.Combine(root, "text_encoder", "model.safetensors"));

        VideoPlan plan = await VideoProfileResolver.ResolveAsync(new ModelSpec
        {
            Requested = root,
            LocalPath = root,
            Modality = Modality.Video,
        }, new VideoRequest { Prompt = "test" }, "minimax-h3", H3Defaults(), VideoFeatures.Lora,
            CancellationToken.None);

        Assert.Contains(plan.Issues, issue => issue.Code == "video.component.tensor_missing"
            && issue.Field == "videoVae");
        Assert.Contains(plan.Issues, issue => issue.Code == "video.component.tensor_missing"
            && issue.Field == "textEncoder");
        Assert.False(plan.IsValid);
    }

    [Fact]
    public async Task Sidecar_CannotActivatePddOrAnUnsupportedVsaMatrix()
    {
        string pddCheckpoint = WriteHeaderOnlyH3("sidecar-pdd.safetensors");
        string pddHash = await VideoCheckpointHashCache.GetSha256Async(pddCheckpoint, CancellationToken.None);
        await WriteSidecar(pddCheckpoint, pddHash, "Pdd", "Dense", "Fl2Va", 8, 12, 3);
        VideoPlan pdd = await Plan(pddCheckpoint);
        Assert.Contains(pdd.Issues, issue => issue.Code == "video.profile.sidecar_acceleration_invalid");
        Assert.False(pdd.Profile.IsSidecar);

        string vsaCheckpoint = WriteHeaderOnlyH3("sidecar-vsa-invalid.safetensors", gateCount: 50);
        string vsaHash = await VideoCheckpointHashCache.GetSha256Async(vsaCheckpoint, CancellationToken.None);
        await WriteSidecar(vsaCheckpoint, vsaHash, "Vsa", "Dense", "Hybrid", 8, 6, 3);
        VideoPlan vsa = await Plan(vsaCheckpoint);
        Assert.Contains(vsa.Issues, issue => issue.Code == "video.profile.sidecar_vsa_matrix_invalid");
        Assert.Contains(vsa.Issues, issue => issue.Code == "video.profile.sidecar_attention_invalid");
        Assert.False(vsa.Profile.IsSidecar);

        Task<VideoPlan> Plan(string path) => VideoProfileResolver.ResolveAsync(new ModelSpec
        {
            Requested = path,
            LocalPath = path,
            Modality = Modality.Video,
        }, new VideoRequest { Prompt = "test" }, "minimax-h3", H3Defaults(), VideoFeatures.Lora,
            CancellationToken.None);

        static Task WriteSidecar(string path, string hash, string acceleration, string attention, string task,
            int steps, double flowShift, double audioFlowShift) => File.WriteAllTextAsync(
            path + ".hartsy-video-profile.json", JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["sha256"] = hash,
                ["profileId"] = "operator-invalid-matrix",
                ["task"] = task,
                ["acceleration"] = acceleration,
                ["attention"] = attention,
                ["steps"] = steps,
                ["cfgScale"] = 1.0,
                ["flowShift"] = flowShift,
                ["audioFlowShift"] = audioFlowShift,
                ["sampler"] = "euler",
                ["scheduler"] = "normal",
            }));
    }

    [Fact]
    public async Task Sidecar_RejectsFilesystemAndCredentialBearingUrls()
    {
        string checkpoint = WriteHeaderOnlyH3("sidecar-private-urls.safetensors");
        string hash = await VideoCheckpointHashCache.GetSha256Async(checkpoint, CancellationToken.None);
        Dictionary<string, object> sidecar = new(StringComparer.Ordinal)
        {
            ["sha256"] = hash,
            ["profileId"] = "operator-private-urls",
            ["task"] = "Fl2Va",
            ["acceleration"] = "None",
            ["attention"] = "Dense",
            ["steps"] = 30,
            ["cfgScale"] = 1.0,
            ["flowShift"] = 12.0,
            ["audioFlowShift"] = 3.0,
            ["sampler"] = "euler",
            ["scheduler"] = "normal",
            ["provenanceUrl"] = "/srv/private/model-notes",
            ["licenseUrl"] = "https://operator-secret@example.invalid/license",
        };
        await File.WriteAllTextAsync(checkpoint + ".hartsy-video-profile.json",
            JsonSerializer.Serialize(sidecar));

        VideoPlan plan = await VideoProfileResolver.ResolveAsync(new ModelSpec
        {
            Requested = checkpoint,
            LocalPath = checkpoint,
            Modality = Modality.Video,
        }, new VideoRequest { Prompt = "test" }, "minimax-h3", H3Defaults(), VideoFeatures.Lora,
            CancellationToken.None);

        Assert.Contains(plan.Issues, issue => issue.Code == "video.profile.sidecar_url_invalid");
        Assert.False(plan.Profile.IsSidecar);
    }

    [Fact]
    public async Task CacheIdentity_IncludesTheResolvedSidecarExecutionContract()
    {
        string checkpoint = WriteHeaderOnlyH3("sidecar-contract-cache.safetensors");
        string hash = await VideoCheckpointHashCache.GetSha256Async(checkpoint, CancellationToken.None);
        string sidecarPath = checkpoint + ".hartsy-video-profile.json";
        await Write("Fl2Va", "None", 30, 12);
        VideoPlan dense = await Plan();

        await Write("T2Va", "Turbo", 8, 6);
        VideoPlan turbo = await Plan();

        Assert.Equal(dense.Profile.Id, turbo.Profile.Id);
        Assert.Equal(dense.Profile.ArtifactSha256, turbo.Profile.ArtifactSha256);
        Assert.NotEqual(dense.CacheIdentity, turbo.CacheIdentity);
        Assert.Contains("contract:task=", dense.CacheIdentity, StringComparison.Ordinal);

        Task<VideoPlan> Plan() => VideoProfileResolver.ResolveAsync(new ModelSpec
        {
            Requested = checkpoint,
            LocalPath = checkpoint,
            Modality = Modality.Video,
        }, new VideoRequest { Prompt = "test" }, "minimax-h3", H3Defaults(), VideoFeatures.Lora,
            CancellationToken.None);

        Task Write(string task, string acceleration, int steps, double flowShift) => File.WriteAllTextAsync(
            sidecarPath, JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["sha256"] = hash,
                ["profileId"] = "operator-stable-id",
                ["task"] = task,
                ["acceleration"] = acceleration,
                ["attention"] = "Dense",
                ["steps"] = steps,
                ["cfgScale"] = 1.0,
                ["flowShift"] = flowShift,
                ["audioFlowShift"] = 3.0,
                ["sampler"] = "euler",
                ["scheduler"] = "normal",
            }));
    }

    [Fact]
    public async Task InvalidH3Request_ProducesIssuesAndSafeEffectiveFallbacks()
    {
        VideoRequest request = new VideoRequest
        {
            Prompt = "test",
            Width = 0,
            Height = int.MaxValue,
            Frames = int.MaxValue,
            Fps = -1,
            Steps = 0,
            CfgScale = float.NaN,
            FlowShift = float.PositiveInfinity,
            AudioFlowShift = -3,
            Sampler = "dpmpp_2m",
            Scheduler = "karras",
            ReferenceSizing = (VideoReferenceSizing)99,
            SparseAttentionPolicy = (SparseAttentionPolicy)99,
            Loras = new LoraStack
            {
                Entries = [new LoraEntry { Model = "", Weight = double.NaN }],
            },
        };
        VideoPlan plan = await VideoProfileResolver.ResolveAsync(new ModelSpec
        {
            Requested = "missing",
            Modality = Modality.Video,
        }, request, "minimax-h3", H3Defaults(), VideoFeatures.Lora, CancellationToken.None);

        Assert.Contains(plan.Issues, issue => issue.Code == "video.request.value_invalid");
        Assert.Contains(plan.Issues, issue => issue.Code == "video.sampler.unsupported");
        Assert.Contains(plan.Issues, issue => issue.Code == "video.scheduler.unsupported");
        Assert.Contains(plan.Issues, issue => issue.Code == "video.reference_sizing.invalid");
        Assert.Contains(plan.Issues, issue => issue.Code == "video.vsa.policy_invalid");
        Assert.Contains(plan.Issues, issue => issue.Code == "video.lora.strength_invalid");
        Assert.Equal(1344, plan.EffectiveSettings.Width);
        Assert.Equal(768, plan.EffectiveSettings.Height);
        Assert.Equal(124, plan.EffectiveSettings.Frames);
        Assert.Equal(1f, plan.EffectiveSettings.CfgScale);
        Assert.Equal(12f, plan.EffectiveSettings.FlowShift);
        Assert.Equal(3f, plan.EffectiveSettings.AudioFlowShift);
        Assert.Equal("euler", plan.EffectiveSettings.Sampler);
        Assert.Equal("normal", plan.EffectiveSettings.Scheduler);
        Assert.False(plan.IsValid);
    }

    [Fact]
    public async Task ConvertedPdd_RejectsAFullWidthBaseEvenWhenItsHashMatchesMetadata()
    {
        string checkpoint = WriteHeaderOnlyH3("full-base-for-converted-pdd.safetensors");
        string baseHash = await VideoCheckpointHashCache.GetSha256Async(checkpoint, CancellationToken.None);
        Dictionary<string, string> metadata = new(StringComparer.Ordinal)
        {
            ["hartsy.pdd.format"] = "minimax_h3_pdd_hartsy_pruned_v1",
            ["hartsy.pdd.task"] = "fl2va",
            ["hartsy.pdd.adapter_sha256"] = new string('a', 64),
            ["hartsy.pdd.full_base_sha256"] = new string('b', 64),
            ["hartsy.pdd.target_base_sha256"] = baseHash,
            ["hartsy.pdd.affine_residual"] = "1.0E-05",
            ["hartsy.pdd.converter"] = "HartsyInference.MiniMaxH3PddPrunedConverter/v1",
            ["pdd_num_steps"] = "32",
            ["pdd_block_size"] = "4",
            ["lora_rank"] = "64",
            ["lora_alpha"] = "64",
        };
        string adapter = WriteHeaderOnlyPdd("converted-against-full.safetensors", metadata);
        VideoPlan plan = await VideoProfileResolver.ResolveAsync(new ModelSpec
        {
            Requested = checkpoint,
            LocalPath = checkpoint,
            Modality = Modality.Video,
        }, new VideoRequest
        {
            Prompt = "test",
            Loras = new LoraStack { Entries = [new LoraEntry { Model = adapter }] },
        }, "minimax-h3", H3Defaults(), VideoFeatures.Lora, CancellationToken.None);

        Assert.Contains(plan.Issues, issue => issue.Code == "video.pdd.converted_base_not_pruned");
        Assert.Equal(VideoAccelerationKind.None, plan.Profile.Acceleration);
        Assert.False(plan.IsValid);
    }

    [Fact]
    public async Task SourceFreeMaskVideo_MustDecodeToExactWhitePixels()
    {
        byte[] whitePng = Convert.FromBase64String(
            "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAACXBIWXMAAAABAAAAAQBPJcTWAAAADklEQVR4nGP4DwYMEAoAU7oL9ZisIGcAAAAASUVORK5CYII=");
        byte[] blackPng = Convert.FromBase64String(
            "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAACXBIWXMAAAABAAAAAQBPJcTWAAAAC0lEQVR4nGNgQAYAAA4AAamRc7EAAAAASUVORK5CYII=");

        VideoPlan white = await Plan(whitePng);
        Assert.DoesNotContain(white.Issues,
            issue => issue.Code is "video.mask.source_required" or "video.mask.decode_invalid");

        VideoPlan black = await Plan(blackPng);
        Assert.Contains(black.Issues, issue => issue.Code == "video.mask.source_required");

        VideoPlan malformed = await Plan([1, 2, 3, 4]);
        Assert.Contains(malformed.Issues, issue => issue.Code == "video.mask.decode_invalid");

        Task<VideoPlan> Plan(byte[] data) => VideoProfileResolver.ResolveAsync(new ModelSpec
        {
            Requested = "missing",
            Modality = Modality.Video,
        }, new VideoRequest
        {
            Prompt = "test",
            VideoDenoiseMask = new VideoDenoiseMask
            {
                MaskVideo = new VideoClip { Data = data, Format = "png" },
            },
        }, "minimax-h3", H3Defaults(), VideoFeatures.VideoDenoiseMask, CancellationToken.None);
    }

    private ImageData Image(int width, int height) =>
        new ImageData { Width = width, Height = height, Rgb = new byte[width * height * 3] };

    private string WriteHeaderOnlyH3(string name, int gateCount = 0, bool pruned = false)
    {
        Dictionary<string, object> header = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["__metadata__"] = new Dictionary<string, string> { ["task"] = "fl2va" },
            ["video_patch_proj.weight"] = Descriptor("F32", [5376, 96]),
            ["video_patch_proj.bias"] = Descriptor("F32", [5376]),
            ["audio_patch_proj.weight"] = Descriptor("F32", [5376, 32]),
            ["audio_patch_proj.bias"] = Descriptor("F32", [5376]),
            ["condition_proj.weight"] = Descriptor("F32", [5376, 5120]),
            ["condition_proj.bias"] = Descriptor("F32", [5376]),
            ["rope.inv_freq"] = Descriptor("F32", [16]),
            ["token_refiner.final_norm.weight"] = Descriptor("F32", [5376]),
            ["final_layer.norm.weight"] = Descriptor("F32", [5376]),
            ["final_layer.video_out.weight"] = Descriptor("F32", [96, 5376]),
            ["final_layer.video_out.bias"] = Descriptor("F32", [96]),
            ["final_layer.audio_out.weight"] = Descriptor("F32", [32, 5376]),
            ["final_layer.audio_out.bias"] = Descriptor("F32", [32]),
        };
        int timeDim = pruned ? 8 : 2688;
        if (pruned)
        {
            header["adaln_t_table"] = Descriptor("F32", [1025, 8]);
        }
        else
        {
            header["time_embedder.proj_in.weight"] = Descriptor("F32", [5376, 256]);
            header["time_embedder.proj_in.bias"] = Descriptor("F32", [5376]);
            header["time_embedder.proj_out.weight"] = Descriptor("F32", [2688, 5376]);
            header["time_embedder.proj_out.bias"] = Descriptor("F32", [2688]);
        }
        for (int i = 0; i < 50; i++)
        {
            string block = $"blocks.{i}";
            header[block + ".norm1.weight"] = Descriptor("F32", [5376]);
            header[block + ".norm2.weight"] = Descriptor("F32", [5376]);
            header[block + ".attn.qkv_proj.weight"] = Descriptor("F32", [21504, 5376]);
            header[block + ".attn.q_norm.weight"] = Descriptor("F32", [128]);
            header[block + ".attn.k_norm.weight"] = Descriptor("F32", [128]);
            header[block + ".attn.out_proj.weight"] = Descriptor("F32", [5376, 7168]);
            header[block + ".mlp.fc1.weight"] = Descriptor("F32", [28672, 5376]);
            header[block + ".mlp.fc2.weight"] = Descriptor("F32", [5376, 14336]);
            header[block + ".adaln_proj.linear.weight"] = Descriptor("F32", [96768, timeDim]);
            header[block + ".adaln_proj.linear.bias"] = Descriptor("F32", [96768]);
        }
        for (int i = 0; i < 2; i++)
        {
            string block = $"token_refiner.blocks.{i}";
            header[block + ".norm1.weight"] = Descriptor("F32", [5376]);
            header[block + ".norm2.weight"] = Descriptor("F32", [5376]);
            header[block + ".attn.qkv_proj.weight"] = Descriptor("F32", [21504, 5376]);
            header[block + ".attn.q_norm.weight"] = Descriptor("F32", [128]);
            header[block + ".attn.k_norm.weight"] = Descriptor("F32", [128]);
            header[block + ".attn.out_proj.weight"] = Descriptor("F32", [5376, 7168]);
            header[block + ".mlp.fc1.weight"] = Descriptor("F32", [28672, 5376]);
            header[block + ".mlp.fc2.weight"] = Descriptor("F32", [5376, 14336]);
        }
        header["final_layer.adaln_proj.linear.weight"] = Descriptor("F32", [10752, timeDim]);
        header["final_layer.adaln_proj.linear.bias"] = Descriptor("F32", [10752]);
        for (int i = 0; i < gateCount; i++)
        {
            header[$"blocks.{i}.attn.to_gate_compress.weight"] = Descriptor("F32", [7168, 5376]);
        }
        return WriteHeader(name, header);
    }

    private string WriteHeaderOnlyFunControl(string name, IReadOnlyDictionary<string, string> metadata, int timeDim)
    {
        Dictionary<string, object> header = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["__metadata__"] = metadata,
            ["control_proj_in.weight"] = Descriptor("F32", [5376, 196]),
            ["control_proj_in.bias"] = Descriptor("F32", [5376]),
        };
        for (int index = 0; index < 5; index++)
        {
            string prefix = $"control_blocks.{index}";
            header[prefix + ".norm1.weight"] = Descriptor("F32", [5376]);
            header[prefix + ".norm2.weight"] = Descriptor("F32", [5376]);
            header[prefix + ".attn.qkv_proj.weight"] = Descriptor("F32", [21504, 5376]);
            header[prefix + ".attn.q_norm.weight"] = Descriptor("F32", [128]);
            header[prefix + ".attn.k_norm.weight"] = Descriptor("F32", [128]);
            header[prefix + ".attn.out_proj.weight"] = Descriptor("F32", [5376, 7168]);
            header[prefix + ".mlp.fc1.weight"] = Descriptor("F32", [28672, 5376]);
            header[prefix + ".mlp.fc2.weight"] = Descriptor("F32", [5376, 14336]);
            header[prefix + ".adaln_proj.linear.weight"] = Descriptor("F32", [96768, timeDim]);
            header[prefix + ".adaln_proj.linear.bias"] = Descriptor("F32", [96768]);
            header[prefix + ".after_proj.weight"] = Descriptor("F32", [5376, 5376]);
            header[prefix + ".after_proj.bias"] = Descriptor("F32", [5376]);
            if (index == 0)
            {
                header[prefix + ".before_proj.weight"] = Descriptor("F32", [5376, 5376]);
                header[prefix + ".before_proj.bias"] = Descriptor("F32", [5376]);
            }
        }
        return WriteHeader(name, header);
    }

    private static async Task WriteFlSidecar(string checkpoint, string hash)
    {
        string json = JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["sha256"] = hash,
            ["profileId"] = "operator-fl2va-dense",
            ["task"] = "Fl2Va",
            ["acceleration"] = "None",
            ["attention"] = "Dense",
            ["steps"] = 30,
            ["cfgScale"] = 1.0,
            ["flowShift"] = 12.0,
            ["audioFlowShift"] = 3.0,
            ["sampler"] = "euler",
            ["scheduler"] = "normal",
        });
        await File.WriteAllTextAsync(checkpoint + ".hartsy-video-profile.json", json);
    }

    private string WriteHeaderOnlyPdd(string name, IReadOnlyDictionary<string, string> metadata)
    {
        Dictionary<string, object> header = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["__metadata__"] = metadata,
            ["proj_out.weight"] = Descriptor("F32", [32, 96, 5376]),
            ["proj_out.bias"] = Descriptor("F32", [32, 96]),
            ["audio_proj_out.weight"] = Descriptor("F32", [32, 32, 5376]),
            ["audio_proj_out.bias"] = Descriptor("F32", [32, 32]),
        };
        return WriteHeader(name, header);
    }

    private string WriteHeader(string name, Dictionary<string, object> header)
    {
        byte[] json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        string path = Path.Combine(_tempDir, name);
        using FileStream stream = File.Create(path);
        stream.Write(BitConverter.GetBytes((long)json.Length));
        stream.Write(json);
        return path;
    }

    private static VideoDefaults H3Defaults() => new VideoDefaults
    {
        Steps = 30,
        CfgScale = 1f,
        Width = 1344,
        Height = 768,
        Frames = 124,
        Fps = 24,
        FlowShift = 12f,
        AudioFlowShift = 3f,
        Sampler = "euler",
        Scheduler = "normal",
    };

    private static Dictionary<string, object> Descriptor(string dtype, long[] shape) =>
        new Dictionary<string, object>
        {
            ["dtype"] = dtype,
            ["shape"] = shape,
            ["data_offsets"] = new long[] { 0, 0 },
        };

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            foreach (string path in Directory.EnumerateFiles(_tempDir, "*", SearchOption.AllDirectories))
            {
                VideoCheckpointHashCache.RemovePersistent(path);
            }
            Directory.Delete(_tempDir, recursive: true);
        }
    }
}

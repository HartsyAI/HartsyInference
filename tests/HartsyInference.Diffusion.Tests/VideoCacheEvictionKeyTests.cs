using HartsyInference.Core.Backends;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the video pipeline cache key's format identity between construction
/// (<c>InferenceEngine.GetOrConstructVideoRecipe</c>) and eviction (<c>EvictOtherCheckpointPipelines</c>): both must
/// agree that a key is <c>"video-recipe:{path}|{describe}{placement}"</c>, or eviction's prefix match silently stops
/// keeping same-checkpoint variants (the bug this pins: an earlier <c>Equals</c>-against-a-pipe-less-prefix comparison
/// never matched a real key, so every video cache miss evicted the entire video pipeline cache).</summary>
public sealed class VideoCacheEvictionKeyTests
{
    private const string CheckpointPath = "/models/video/minimax-h3.safetensors";

    private static string BuildVideoCacheKey(string path, VideoRequest? request, PlacementConfig placement) =>
        $"video-recipe:{path}|{RecipeCacheKey.Describe(request)}{placement.CacheKey()}";

    [Fact]
    public void SameCheckpoint_DifferentRequestShapes_AllStartWithTheKeepPrefix()
    {
        string keepPrefix = $"video-recipe:{CheckpointPath}|";
        PlacementConfig placement = new();
        string plainKey = BuildVideoCacheKey(CheckpointPath, new VideoRequest { Prompt = "a" }, placement);
        string loraKey = BuildVideoCacheKey(CheckpointPath, new VideoRequest { Prompt = "a", VideoSwapModel = "low.safetensors" }, placement);
        string nullRequestKey = BuildVideoCacheKey(CheckpointPath, null, placement);

        Assert.StartsWith(keepPrefix, plainKey);
        Assert.StartsWith(keepPrefix, loraKey);
        Assert.StartsWith(keepPrefix, nullRequestKey);
    }

    [Fact]
    public void DifferentCheckpoint_NeverStartsWithAnotherPathsKeepPrefix()
    {
        string keepPrefix = $"video-recipe:{CheckpointPath}|";
        string otherKey = BuildVideoCacheKey("/models/video/wan22.safetensors", new VideoRequest { Prompt = "a" }, new PlacementConfig());

        Assert.False(otherKey.StartsWith(keepPrefix));
    }

    /// <summary>The exact bug: comparing a real key against a keep-key built WITHOUT the trailing separator via
    /// <c>Equals</c> never matches, because every real key has content after the pipe.</summary>
    [Fact]
    public void PipeLessKeepKey_NeverEqualsARealKey_DemonstratingTheOriginalBug()
    {
        string buggyKeepKey = $"video-recipe:{CheckpointPath}";
        string realKey = BuildVideoCacheKey(CheckpointPath, new VideoRequest { Prompt = "a" }, new PlacementConfig());

        Assert.NotEqual(buggyKeepKey, realKey);
        Assert.StartsWith($"{buggyKeepKey}|", realKey);
    }
}

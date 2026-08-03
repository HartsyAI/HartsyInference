using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Runs <see cref="MiniMaxH3Config.Detect"/> against the real MiniMax-H3 checkpoint and asserts every
/// derived field matches the published <c>transformer/config.json</c>. Skips when the checkpoint is absent.
/// <code>MINIMAX_H3_DIT=/path/minimax_h3_fl2va_bf16.safetensors dotnet test --filter MiniMaxH3RealCheckpoint</code></summary>
public class MiniMaxH3RealCheckpointTests
{
    private readonly ITestOutputHelper _output;

    public MiniMaxH3RealCheckpointTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "Integration")]
    public void DetectMatchesThePublishedConfig()
    {
        string? path = Environment.GetEnvironmentVariable("MINIMAX_H3_DIT");
        if (path is null || !File.Exists(path))
        {
            return;
        }

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(loader.GetAllTensors());
        _output.WriteLine($"tensors: {weights.Count}");

        Assert.True(MiniMaxH3Config.Matches(weights), "H3 signature (video_patch_proj + audio_patch_proj) not found");
        MiniMaxH3Config c = MiniMaxH3Config.Detect(weights);
        _output.WriteLine($"hidden={c.HiddenSize} layers={c.NumLayers} refiner={c.TokenRefinerNumLayers} "
            + $"heads={c.NumAttentionHeads}x{c.AttentionHeadDim} ffn={c.FfnHiddenSize} "
            + $"latents={c.LatentsDim}/{c.AudioLatentsDim} textDim={c.TextDim} rope={c.RopeInvFreqLen} "
            + $"timeEmbed={c.TimeEmbedDim} curves={c.UseAdalnCurves}");

        // Values published in MiniMaxAI/MiniMax-H3 FL2VA/transformer/config.json.
        Assert.Equal(5376, c.HiddenSize);
        Assert.Equal(50, c.NumLayers);
        Assert.Equal(2, c.TokenRefinerNumLayers);
        Assert.Equal(56, c.NumAttentionHeads);
        Assert.Equal(128, c.AttentionHeadDim);
        Assert.Equal(14336, c.FfnHiddenSize);
        Assert.Equal(24, c.LatentsDim);
        Assert.Equal(32, c.AudioLatentsDim);
        Assert.Equal(5120, c.TextDim);
        Assert.Equal(16, c.RopeInvFreqLen);
        Assert.Equal(96, c.VideoPatchDim);   // 24 channels x 1x2x2 patch

        // The bf16 release carries a real time embedder; the pruned release swaps it for the curve table.
        if (c.UseAdalnCurves)
        {
            Assert.NotNull(c.AdalnCurveGrid);
        }
        else
        {
            Assert.Equal(2688, c.TimeEmbedDim);
            Assert.Equal(256, c.TimestepInputDim);
        }
    }
}

using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Config detection on a tiny H3 fixture. The shapes are the image build's, scaled down: 2 heads of 4, 16-wide
/// FFN, 5-wide text. The grid and head count come from the metadata the real checkpoint declares.</summary>
public sealed class MiniMaxH3ImageConfigTests
{
    private const string SixStepGrid =
        "[1.0,0.9675236940383911,0.8875485062599182,0.75,0.5327739119529724,0.23200887441635132,0.0]";

    [Fact]
    public void SixHeadImageFixtureYieldsLatentWidth24AndSevenPointGrid()
    {
        MiniMaxH3Config config = MiniMaxH3Config.Detect(ImageWeights(rows: 96 * 6), Metadata(6, SixStepGrid));

        Assert.False(config.HasAudio);
        Assert.Equal(24, config.LatentsDim);
        Assert.Equal(6, config.OutputHeadCount);
        Assert.NotNull(config.OutputHeadSigmas);
        Assert.Equal(7, config.OutputHeadSigmas!.Count);
        Assert.Equal(1.0, config.OutputHeadSigmas[0]);
        Assert.Equal(0.0, config.OutputHeadSigmas[^1]);
        Assert.Equal(2, config.NumAttentionHeads);
        Assert.Equal(4, config.AttentionHeadDim);
    }

    [Fact]
    public void ImageConfigRefusesAudioChannels()
    {
        MiniMaxH3Config config = MiniMaxH3Config.Detect(ImageWeights(rows: 96 * 6), Metadata(6, SixStepGrid));

        Assert.Throws<InvalidOperationException>(() => config.AudioChannels);
    }

    [Fact]
    public void VideoFixtureKeepsAudioAndSingleHead()
    {
        MiniMaxH3Config config = MiniMaxH3Config.Detect(VideoWeights());

        Assert.True(config.HasAudio);
        Assert.Equal(24, config.LatentsDim);
        Assert.Equal(32, config.AudioChannels);
        Assert.Equal(1, config.OutputHeadCount);
        Assert.Null(config.OutputHeadSigmas);
    }

    [Fact]
    public void HeadCountNotMatchingRowsIsRefused()
    {
        // 6 heads declared, but video_out still has one head's rows.
        Assert.Throws<ArgumentException>(() => MiniMaxH3Config.Detect(ImageWeights(rows: 96), Metadata(6, SixStepGrid)));
    }

    [Fact]
    public void GridWithWrongPointCountIsRefused()
    {
        string fiveSteps = "[1.0,0.8,0.6,0.4,0.2,0.0]";

        Assert.Throws<ArgumentException>(() => MiniMaxH3Config.Detect(ImageWeights(rows: 96 * 6), Metadata(6, fiveSteps)));
    }

    [Fact]
    public void GridNotStartingAtOneIsRefused()
    {
        string grid = "[0.9,0.8,0.7,0.6,0.5,0.4,0.0]";

        Assert.Throws<ArgumentException>(() => MiniMaxH3Config.Detect(ImageWeights(rows: 96 * 6), Metadata(6, grid)));
    }

    [Fact]
    public void GridNotStrictlyDecreasingIsRefused()
    {
        string grid = "[1.0,0.9,0.95,0.7,0.5,0.2,0.0]";

        Assert.Throws<ArgumentException>(() => MiniMaxH3Config.Detect(ImageWeights(rows: 96 * 6), Metadata(6, grid)));
    }

    [Fact]
    public void MultiHeadRowsWithoutMetadataAreRefused()
    {
        Assert.Throws<ArgumentException>(() => MiniMaxH3Config.Detect(ImageWeights(rows: 96 * 6)));
    }

    private static IReadOnlyDictionary<string, string> Metadata(int steps, string grid) =>
        new Dictionary<string, string>
        {
            ["config"] = $"{{\"transformer\":{{\"pdd_steps\":{steps},\"pdd_sigmas\":{grid}}}}}",
        };

    private static Dictionary<string, Tensor> ImageWeights(int rows)
    {
        Dictionary<string, Tensor> weights = SharedWeights(rows);
        return weights;
    }

    private static Dictionary<string, Tensor> VideoWeights()
    {
        Dictionary<string, Tensor> weights = SharedWeights(rows: 96);
        weights["audio_patch_proj.weight"] = Zeros(8, 96);
        weights["final_layer.audio_out.weight"] = Zeros(32, 8);
        return weights;
    }

    private static Dictionary<string, Tensor> SharedWeights(int rows)
    {
        const int Hidden = 8;
        return new Dictionary<string, Tensor>
        {
            ["video_patch_proj.weight"] = Zeros(Hidden, 96),
            ["condition_proj.weight"] = Zeros(Hidden, 5),
            ["adaln_t_table"] = Zeros(1025, Hidden),
            ["final_layer.video_out.weight"] = Zeros(rows, Hidden),
            ["blocks.0.attn.q_norm.weight"] = Zeros(4),
            ["blocks.0.attn.qkv_proj.weight"] = Zeros(3 * 2 * 4, Hidden),
            ["blocks.0.mlp.fc1.weight"] = Zeros(2 * 16, Hidden),
            ["rope.inv_freq"] = Zeros(16),
        };
    }

    private static Tensor Zeros(params long[] dims) => new(new TensorShape(dims), DType.F32);
}

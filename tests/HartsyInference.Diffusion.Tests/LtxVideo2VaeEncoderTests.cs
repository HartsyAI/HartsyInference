using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using static HartsyInference.Diffusion.Tests.LtxVideo2VaeEncoderFixtures;

namespace HartsyInference.Diffusion.Tests;

public sealed class LtxVideo2VaeEncoderTests
{
    private const float LeakThreshold = 1e-4f;

    [Fact]
    public void ExpectedWeightShapes_RealConfig_MatchesCheckpointLayout()
    {
        IReadOnlyDictionary<string, TensorShape> shapes = LtxVideo2VaeEncoder.ExpectedWeightShapes(new LtxVideo2VaeEncoderConfig());

        Assert.Equal(92, shapes.Count);
        Assert.Equal(new TensorShape([128, 48, 3, 3, 3]), shapes["encoder.conv_in.conv.weight"]);
        Assert.Equal(new TensorShape([64, 128, 3, 3, 3]), shapes["encoder.down_blocks.0.downsamplers.0.conv.conv.weight"]);
        Assert.Equal(new TensorShape([128, 512, 3, 3, 3]), shapes["encoder.down_blocks.2.downsamplers.0.conv.conv.weight"]);
        Assert.Equal(new TensorShape([129, 2048, 3, 3, 3]), shapes["encoder.conv_out.conv.weight"]);
    }

    [Fact]
    public void Encode_LatentFrameIgnoresLaterFrames_WhenCausal()
    {
        using SafeTensorsLoader weights = Open(WeightsFile);
        using SafeTensorsLoader io = Open(IoFile);
        using IBackend backend = new CpuBackend();
        Dictionary<string, Tensor> tensors = weights.GetAllTensors();
        Tensor video = io.GetTensor("input");

        // Frame 3 only reaches latent frames 1 and up, since latent frame j depends on input frames 0..4j.
        using Tensor perturbed = Clone(video);
        AddToFrame(perturbed, 3, 0.25f);

        LtxVideo2VaeEncoder causal = LoadEncoder(TinyConfig(), tensors);
        using Tensor baseline = causal.Encode(backend, video);
        using Tensor after = causal.Encode(backend, perturbed);
        Assert.True(LatentFrame(baseline, 0).AsSpan().SequenceEqual(LatentFrame(after, 0)),
            "latent frame 0 changed when only input frame 3 changed");
        Assert.True(MaxAbsDiff(LatentFrame(baseline, 1), LatentFrame(after, 1)) > LeakThreshold,
            "latent frame 1 did not respond to input frame 3; the test would be vacuous");

        // Control: the same weights with encoder_causal=false leak frame 3 into latent frame 0.
        LtxVideo2VaeEncoder bidirectional = LoadEncoder(TinyConfig() with { IsCausal = false }, tensors);
        using Tensor leakBaseline = bidirectional.Encode(backend, video);
        using Tensor leakAfter = bidirectional.Encode(backend, perturbed);
        Assert.True(MaxAbsDiff(LatentFrame(leakBaseline, 0), LatentFrame(leakAfter, 0)) > LeakThreshold,
            "non-causal control did not leak; the causality check cannot be trusted");
    }

    [Fact]
    public void Encode_SingleFrame_ProducesOneLatentFrameMatchingClipFirstFrame()
    {
        using SafeTensorsLoader weights = Open(WeightsFile);
        using SafeTensorsLoader io = Open(IoFile);
        using IBackend backend = new CpuBackend();
        LtxVideo2VaeEncoder encoder = LoadEncoder(TinyConfig(), weights.GetAllTensors());
        Tensor video = io.GetTensor("input");

        using Tensor clip = encoder.Encode(backend, video);
        using Tensor image = CopyFrame(video, 0);
        using Tensor single = encoder.Encode(backend, image);

        Assert.Equal(new TensorShape([1, 8, 1, 2, 3]), single.Shape);
        float diff = MaxAbsDiff(LatentFrame(single, 0), LatentFrame(clip, 0));
        Assert.True(diff <= 1e-5f, $"single-frame latent differs from the clip's first latent frame by {diff:E3}");
    }
}

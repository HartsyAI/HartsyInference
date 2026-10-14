using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>CAM++'s context pool over a length that is not a multiple of the 100-frame segment: torch
/// <c>avg_pool1d(ceil_mode=True)</c> (upstream CAMPPlus, IndexTTS-2) averages the trailing partial window over the frames
/// it covers, the ONNX export CosyVoice 2 deploys over 100. A wrong divisor put 9% error on IndexTTS-2.5's style vector.</summary>
public sealed class CamPlusContextPoolTests
{
    [Theory]
    [InlineData(false, 124.5f)]
    [InlineData(true, 62.25f)]
    public void PartialSegment_DividesByTheDivisorOfItsExport(bool onnxSegmentPooling, float trailingSegmentMean)
    {
        const int t = 150;
        using Tensor x = new(new TensorShape(1, 1, t), DType.F32);
        Span<float> xs = x.AsSpan<float>();
        for (int j = 0; j < t; j++) xs[j] = j;

        using Tensor pooled = CamPlusSpeakerEncoder.ContextPool(x, onnxSegmentPooling);
        ReadOnlySpan<float> p = pooled.AsSpan<float>();
        const float mean = 74.5f;
        Assert.Equal(mean + 49.5f, p[0], 4);
        Assert.Equal(mean + 49.5f, p[99], 4);
        Assert.Equal(mean + trailingSegmentMean, p[100], 4);
        Assert.Equal(mean + trailingSegmentMean, p[t - 1], 4);
    }

    /// <summary>Chatterbox's upstream runs <c>CAMPPlus</c> in PyTorch; the encoder its loader builds must use that divisor.</summary>
    [Fact]
    public void ChatterboxSpeakerEncoder_UsesTheTorchDivisor()
    {
        using CamPlusSpeakerEncoder encoder = ChatterboxPipeline.CreateSpeakerEncoder(192);
        Assert.False(encoder.OnnxSegmentPooling);
    }
}

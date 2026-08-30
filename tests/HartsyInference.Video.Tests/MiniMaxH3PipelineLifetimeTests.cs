using HartsyInference.Core.Tensors;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Video.Tests;

public sealed class MiniMaxH3PipelineLifetimeTests
{
    [Fact]
    public void AudioDecodeFailureDisposesInputAndReleasesStagedWeights()
    {
        Tensor audioLatent = new Tensor(new TensorShape(2, 4), DType.F32);
        bool weightsReleased = false;

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            MiniMaxH3Pipeline.DecodeAudioWithCleanup(
                audioLatent,
                static () => throw new InvalidOperationException("decode failed"),
                () => weightsReleased = true));

        Assert.Equal("decode failed", error.Message);
        Assert.True(weightsReleased);
        Assert.Throws<ObjectDisposedException>(() => audioLatent.AsSpan<float>());
    }
}

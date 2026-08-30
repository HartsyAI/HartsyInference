using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.MiniMaxH3;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class PddHeadFusionTests
{
    [Fact]
    public void Fuse_FullHeadsUsesSeparateVideoAndAudioDeltaSigmaWeights()
    {
        using Tensor videoWeight = TensorOf(new TensorShape(32, 2, 2), index => (index / 4) + 1);
        using Tensor videoBias = TensorOf(new TensorShape(32, 2), index => (index / 2) + 1);
        using Tensor audioWeight = TensorOf(new TensorShape(32, 1, 2), index => 100 + index / 2);
        using Tensor audioBias = TensorOf(new TensorShape(32, 1), index => 100 + index);
        using PddHeadBank bank = new PddHeadBank(videoWeight, videoBias, audioWeight, audioBias,
            MiniMaxH3PddHeadLayout.FullHeads, requirePublishedShape: false);
        using CpuBackend backend = new CpuBackend();
        using PddHeadFusionSession session = new PddHeadFusionSession(backend, bank);
        MiniMaxH3PddSchedule schedule = MiniMaxH3PddSchedule.Create(Settings());
        MiniMaxH3PddStep step = schedule.Steps[0];

        using PddFusedHeads fused = session.Fuse(schedule, step.Sigma, step.SigmaNext);

        float expectedVideo = 0.0f;
        float expectedAudio = 0.0f;
        for (int i = 0; i < step.FineCount; i++)
        {
            expectedVideo += step.VideoWeights[i] * (i + 1);
            expectedAudio += step.AudioWeights[i] * (100 + i);
        }
        Assert.Equal(expectedVideo, fused.VideoWeight.AsSpan<float>()[0], 5);
        Assert.Equal(expectedAudio, fused.AudioWeight.AsSpan<float>()[0], 5);
        Assert.NotEqual(fused.VideoWeight.AsSpan<float>()[0] - 1.0f,
            fused.AudioWeight.AsSpan<float>()[0] - 100.0f);
    }

    [Fact]
    public void Fuse_BasePlusOffsetsKeepsBaseAtUnitWeight()
    {
        using Tensor videoWeight = TensorOf(new TensorShape(32, 2, 2), index =>
        {
            int head = index / 4;
            return head == 0 ? 10.0f : head;
        });
        using Tensor videoBias = TensorOf(new TensorShape(32, 2), index =>
        {
            int head = index / 2;
            return head == 0 ? 10.0f : head;
        });
        using Tensor audioWeight = TensorOf(new TensorShape(32, 1, 2), index =>
        {
            int head = index / 2;
            return head == 0 ? 20.0f : head;
        });
        using Tensor audioBias = TensorOf(new TensorShape(32, 1), index => index == 0 ? 20.0f : index);
        using PddHeadBank bank = new PddHeadBank(videoWeight, videoBias, audioWeight, audioBias,
            MiniMaxH3PddHeadLayout.BasePlusOffsets, requirePublishedShape: false);
        using CpuBackend backend = new CpuBackend();
        using PddHeadFusionSession session = new PddHeadFusionSession(backend, bank);
        MiniMaxH3PddSchedule schedule = MiniMaxH3PddSchedule.Create(Settings());
        MiniMaxH3PddStep step = schedule.Steps[0];

        using PddFusedHeads fused = session.Fuse(schedule, step.Sigma, step.SigmaNext);

        float expected = 10.0f;
        for (int i = 1; i < step.FineCount; i++) expected += step.VideoWeights[i] * i;
        Assert.Equal(expected, fused.VideoWeight.AsSpan<float>()[0], 5);
    }

    private static Tensor TensorOf(TensorShape shape, Func<int, float> value)
    {
        Tensor tensor = new Tensor(shape, DType.F32);
        Span<float> span = tensor.AsSpan<float>();
        for (int i = 0; i < span.Length; i++) span[i] = value(i);
        return tensor;
    }

    private static MiniMaxH3PddExecutionSettings Settings() => new()
    {
        Nfe = 8,
        Sampler = "euler",
        CfgScale = 1.0f,
        VideoFlowShift = 12.0,
        AudioFlowShift = 3.0,
        Strength = 1.0f,
    };
}

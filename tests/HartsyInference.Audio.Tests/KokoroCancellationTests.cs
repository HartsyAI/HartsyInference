using HartsyInference.Audio.Models.Kokoro;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Kokoro's stage boundaries without weights. The modules here are never loaded, so a stage that ran would fail
/// with something other than cancellation: a cancelled call has to stop at its first boundary, before any device work,
/// and a boundary that stops a call has to release the tensors its stage still held.</summary>
public sealed class KokoroCancellationTests
{
    [Fact]
    public void APreCancelledSynthesis_StopsAtTheFirstBoundary()
    {
        using KokoroPipeline pipeline = UnloadedPipeline();
        List<string> stages = [];
        pipeline.TestStageObserver = stages.Add;
        using CpuBackend backend = new();
        using Tensor style = new(new TensorShape(1, 256), DType.F32);

        Assert.Throws<OperationCanceledException>(() =>
            pipeline.SynthesizeFromStyle(backend, "ab", style, cancel: new CancellationToken(canceled: true)));

        Assert.Equal(["start"], stages);
    }

    [Fact]
    public void ACancelRequestedDuringTheCall_StopsItAtTheBoundaryThatSeesIt()
    {
        using KokoroPipeline pipeline = UnloadedPipeline();
        using CancellationTokenSource cancel = new();
        List<string> stages = [];
        pipeline.TestStageObserver = stage =>
        {
            stages.Add(stage);
            cancel.Cancel();
        };
        using CpuBackend backend = new();
        using Tensor style = new(new TensorShape(1, 256), DType.F32);

        Assert.Throws<OperationCanceledException>(() => pipeline.SynthesizeFromStyle(backend, "ab", style, cancel: cancel.Token));

        Assert.Equal(["start"], stages);
    }

    [Fact]
    public void ACancelledBoundary_DisposesTheStagesLiveTensors_ThenThrows()
    {
        using Tensor x = new(new TensorShape(1, 4, 8), DType.F32);
        using Tensor har = new(new TensorShape(1, 22, 8), DType.F32);
        List<string> seen = [];

        Assert.Throws<OperationCanceledException>(() =>
            KokoroOps.StageBoundary(seen.Add, "stage0", new CancellationToken(canceled: true), x, har));

        Assert.Equal(["stage0"], seen);
        Assert.Throws<ObjectDisposedException>(() => x.AsSpan<float>().Length);
        Assert.Throws<ObjectDisposedException>(() => har.AsSpan<float>().Length);
    }

    [Fact]
    public void APassedBoundary_LeavesTheStagesTensorsAlive()
    {
        using Tensor x = new(new TensorShape(1, 4, 8), DType.F32);
        using CancellationTokenSource live = new();
        List<string> seen = [];

        KokoroOps.StageBoundary(seen.Add, "encode", live.Token, x);

        Assert.Equal(["encode"], seen);
        Assert.Equal(32, x.AsSpan<float>().Length);
    }

    /// <summary>A pipeline over modules built from the v1 config with no weights loaded.</summary>
    private static KokoroPipeline UnloadedPipeline()
    {
        KokoroConfig config = KokoroConfig.V1;
        KokoroPhonemeTokenizer tokenizer = KokoroPhonemeTokenizer.FromVocab(new Dictionary<string, int> { ["a"] = 43, ["b"] = 44 });
        return new KokoroPipeline(config, tokenizer, new KokoroPlBert(config), new KokoroTextEncoder(config),
            new KokoroProsodyPredictor(config), new KokoroIStftNetDecoder(config));
    }
}

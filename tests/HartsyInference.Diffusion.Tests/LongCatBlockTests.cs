using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed unsafe class LongCatBlockTests
{
    private const int Frames = 5, GridH = 2, GridW = 3, TextRows = 4;

    /// <summary>The plain forward and the split-segment forward are the same attention when the segments cover the same query/key pairs.</summary>
    [Fact]
    public void DenseAndSplitSegmentsGiveTheSameBlockOutput()
    {
        LongCatConfig config = LongCatTestData.Tiny;
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, 4);
        try
        {
            LongCatBlock block = new(config);
            block.LoadWeights(weights, "blocks.0");
            int hw = GridH * GridW, tokens = Frames * hw;
            using Tensor hidden = LongCatTestData.Random(new TensorShape(tokens, config.HiddenSize), 21);
            using Tensor text = LongCatTestData.Random(new TensorShape(TextRows, config.HiddenSize), 22);
            using Tensor timeEmbed = LongCatTestData.Random(new TensorShape(Frames, config.AdalnTembedDim), 23);
            WanRope rope = new(config.HeadDim, LongCatConfig.RopeTheta, 0);
            (Tensor cos, Tensor sin) = rope.BuildCosSin(Frames, GridH, GridW);
            using Tensor index = new(new TensorShape(tokens), DType.I32);
            for (int i = 0; i < tokens; i++)
                ((int*)index.DataPointer)[i] = i / hw;

            AttentionSegment[] dense = [AttentionSegment.Dense(tokens)];
            AttentionSegment[] split = [new(0, 7, 0, tokens), new(7, tokens - 7, 0, tokens)];
            using Tensor a = block.Forward(backend, hidden, text, timeEmbed, new LongCatBlockInputs(Frames, hw, 0, cos, sin, dense, index));
            using Tensor b = block.Forward(backend, hidden, text, timeEmbed, new LongCatBlockInputs(Frames, hw, 0, cos, sin, split, index));

            Assert.True(LongCatTestData.RelRms(b.AsReadOnlySpan<float>(), a.AsReadOnlySpan<float>()) < 1e-6);
            cos.Dispose();
            sin.Dispose();
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }
}

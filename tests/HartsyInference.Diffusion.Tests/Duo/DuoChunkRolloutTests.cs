using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>Self-consistency of the chunk loop: the cached per-block denoise and commit reproduce a cache-free full forward over the
/// same history, block by block. These are not parity with upstream Duo; that is the L5/L6 step against the Python reference.</summary>
public sealed class DuoChunkRolloutTests
{
    private const int Hidden = 8, Heads = 2, HeadDim = 96, Inner = Heads * HeadDim, Ffn = 16, TextDim = 5;
    private const int VideoIn = 96, AudioIn = 32, TimeEmbedDim = 6, TimeEmbedHidden = 8;
    private const int Shared = 3, BodyRows = 2, RefRows = 4, VideoRows = 8, AudioRows = 6;
    private const float Tolerance = 1e-4f;

    [Fact]
    public void FirstBlockFromTheCacheMatchesTheFullForward()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoSampleInput sample = Sample();
        DuoStreamCache cache = new(blocks: 1, width: Inner, capacityRows: 64);
        DuoChunkRollout.Prefill(transformer, backend, sample, DuoRopePolicy.Old, cache, Shared0(), Values(30, RefRows * VideoIn));

        (float[] video, float[] audio) = DuoChunkRollout.GenerateBlock(transformer, backend, sample, 0, DuoRopePolicy.Old, cache,
            Shared0(), Body(0), Values(40, VideoRows * VideoIn), Values(41, AudioRows * AudioIn), steps: 2, shift: 1.0);
        (float[] fullVideo, float[] fullAudio) = DuoChunkRollout.FullBlock(transformer, backend, sample, 0, DuoRopePolicy.Old,
            Shared0(), [Body(0), Body(1)], Values(30, RefRows * VideoIn), [], [], Values(40, VideoRows * VideoIn),
            Values(41, AudioRows * AudioIn), steps: 2, shift: 1.0);

        AssertClose(fullVideo, video, "block 0 video");
        AssertClose(fullAudio, audio, "block 0 audio");
    }

    [Fact]
    public void SecondBlockReadsTheCommittedFirstBlock()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoSampleInput sample = Sample();
        DuoStreamCache cache = new(blocks: 1, width: Inner, capacityRows: 64);
        float[] reference = Values(30, RefRows * VideoIn);
        DuoChunkRollout.Prefill(transformer, backend, sample, DuoRopePolicy.Old, cache, Shared0(), reference);
        (float[] clean0Video, float[] clean0Audio) = DuoChunkRollout.GenerateBlock(transformer, backend, sample, 0, DuoRopePolicy.Old,
            cache, Shared0(), Body(0), Values(40, VideoRows * VideoIn), Values(41, AudioRows * AudioIn), steps: 2, shift: 1.0);
        // The cache keeps reference, shared and clean rows only; the block's body Intent rows are not retained.
        Assert.Equal(Shared + RefRows + VideoRows + AudioRows, cache.Length);

        (float[] video, float[] audio) = DuoChunkRollout.GenerateBlock(transformer, backend, sample, 1, DuoRopePolicy.Old, cache,
            Shared0(), Body(1), Values(50, VideoRows * VideoIn), Values(51, AudioRows * AudioIn), steps: 2, shift: 1.0);
        (float[] fullVideo, float[] fullAudio) = DuoChunkRollout.FullBlock(transformer, backend, sample, 1, DuoRopePolicy.Old,
            Shared0(), [Body(0), Body(1)], reference, clean0Video, clean0Audio, Values(50, VideoRows * VideoIn),
            Values(51, AudioRows * AudioIn), steps: 2, shift: 1.0);

        AssertClose(fullVideo, video, "block 1 video");
        AssertClose(fullAudio, audio, "block 1 audio");
    }

    private static DuoSampleInput Sample() => new()
    {
        SampleId = 3,
        SharedDescriptionRows = 2,
        SharedCueRows = 1,
        Reference = new DuoGrid(1, 4, 4),
        Blocks =
        [
            new DuoBlockInput { StartMs = 0.0, BodyTextRows = BodyRows, Video = new DuoGrid(2, 4, 4), AudioSteps = 3, FrameStartsMs = [0.0, 100.0] },
            new DuoBlockInput { StartMs = 708.0, BodyTextRows = BodyRows, Video = new DuoGrid(2, 4, 4), AudioSteps = 3, FrameStartsMs = [708.0, 808.0] },
        ],
        CleanBlocks = 0,
        NoisyBlocks = [0],
    };

    private static float[] Shared0() => Values(2, Shared * TextDim);

    private static float[] Body(int block) => Values(10 + block, BodyRows * TextDim);

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        float max = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            max = MathF.Max(max, MathF.Abs(expected[i] - actual[i]));
        }
        Assert.True(max <= Tolerance, $"{what} differs from the full forward by {max}.");
    }

    private static MiniMaxH3Transformer Transformer()
    {
        MiniMaxH3Transformer transformer = new(new MiniMaxH3Config
        {
            HiddenSize = Hidden,
            NumLayers = 1,
            TokenRefinerNumLayers = 1,
            NumAttentionHeads = Heads,
            AttentionHeadDim = HeadDim,
            FfnHiddenSize = Ffn,
            LatentsDim = 24,
            HasAudio = true,
            TextDim = TextDim,
            OutputHeadCount = 4,
            TimeEmbedHiddenSize = TimeEmbedHidden,
            TimeEmbedDim = TimeEmbedDim,
        });
        transformer.LoadWeights(new Dictionary<string, Tensor>
        {
            ["video_patch_proj.weight"] = Weight(Hidden, VideoIn, 10),
            ["audio_patch_proj.weight"] = Weight(Hidden, AudioIn, 11),
            ["condition_proj.weight"] = Weight(Hidden, TextDim, 12),
            ["time_embedder.proj_in.weight"] = Weight(TimeEmbedHidden, 256, 13),
            ["time_embedder.proj_out.weight"] = Weight(TimeEmbedDim, TimeEmbedHidden, 14),
            ["token_refiner.blocks.0.norm1.weight"] = Ones(Hidden),
            ["token_refiner.blocks.0.norm2.weight"] = Ones(Hidden),
            ["token_refiner.blocks.0.attn.qkv_proj.weight"] = Weight(3 * Inner, Hidden, 15),
            ["token_refiner.blocks.0.attn.q_norm.weight"] = Ones(HeadDim),
            ["token_refiner.blocks.0.attn.k_norm.weight"] = Ones(HeadDim),
            ["token_refiner.blocks.0.attn.out_proj.weight"] = Weight(Hidden, Inner, 16),
            ["token_refiner.blocks.0.mlp.fc1.weight"] = Weight(2 * Ffn, Hidden, 17),
            ["token_refiner.blocks.0.mlp.fc2.weight"] = Weight(Hidden, Ffn, 18),
            ["token_refiner.final_norm.weight"] = Ones(Hidden),
            ["blocks.0.norm1.weight"] = Ones(Hidden),
            ["blocks.0.norm2.weight"] = Ones(Hidden),
            ["blocks.0.adaln_proj.linear.weight"] = Weight(6 * Hidden * 3, TimeEmbedDim, 19),
            ["blocks.0.attn.qkv_proj.weight"] = Weight(3 * Inner, Hidden, 20),
            ["blocks.0.attn.q_norm.weight"] = Ones(HeadDim),
            ["blocks.0.attn.k_norm.weight"] = Ones(HeadDim),
            ["blocks.0.attn.out_proj.weight"] = Weight(Hidden, Inner, 21),
            ["blocks.0.mlp.fc1.weight"] = Weight(2 * Ffn, Hidden, 22),
            ["blocks.0.mlp.fc2.weight"] = Weight(Hidden, Ffn, 23),
            ["final_layer.norm.weight"] = Ones(Hidden),
            ["final_layer.adaln_proj.linear.weight"] = Weight(2 * Hidden, TimeEmbedDim, 24),
            ["final_layer.video_out.weight"] = Weight(VideoIn, Hidden, 25),
            ["final_layer.audio_out.weight"] = Weight(AudioIn, Hidden, 26),
        });
        return transformer;
    }

    private static Tensor Weight(int rows, int cols, int seed) => MakeTensor(new TensorShape(rows, cols), Values(seed, rows * cols, 0.2f));

    private static Tensor MakeTensor(TensorShape shape, float[] values)
    {
        Tensor tensor = new(shape, DType.F32);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static Tensor Ones(int count)
    {
        float[] data = new float[count];
        Array.Fill(data, 1f);
        return MakeTensor(new TensorShape(count), data);
    }

    private static float[] Values(int seed, int count, float scale = 1f)
    {
        float[] data = new float[count];
        for (int i = 0; i < count; i++)
        {
            data[i] = scale * MathF.Sin(seed * 1.7f + i * 0.37f);
        }
        return data;
    }
}

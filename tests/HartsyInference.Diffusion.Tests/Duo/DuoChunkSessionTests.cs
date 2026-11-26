using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>Self-consistency of the chunk loop: the cached per-block denoise and commit reproduce a cache-free full forward over the same
/// history. The model has two layers and weights large enough that the history changes the output well beyond the tolerance, and the
/// negative controls show the check fails when the history is wrong. These are not parity with upstream Duo (that is the L5/L6 step
/// against the Python reference, not run).</summary>
public sealed class DuoChunkSessionTests
{
    private const int Hidden = 8, Heads = 2, HeadDim = 96, Inner = Heads * HeadDim, Ffn = 16, TextDim = 5, Layers = 2;
    private const int VideoIn = 96, AudioIn = 32, TimeEmbedDim = 6, TimeEmbedHidden = 8;
    private const int Shared = 3, BodyRows = 2, RefRows = 4, VideoRows = 8, AudioRows = 6;
    private const float Tolerance = 1e-4f;
    /// <summary>A wrong history must move the output this far, or the agreement check would pass it.</summary>
    private const float Sensitivity = 1e-2f;

    [Fact]
    public void CachedBlocksMatchTheFullForwardOverTheSameHistory()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoSampleInput sample = Sample();
        DuoChunkSession session = new(transformer, backend, sample, DuoRopePolicy.Old, Cache());
        float[] reference = Values(30, RefRows * VideoIn);
        session.Prefill(Shared0(), reference);

        (float[] video0, float[] audio0) = session.GenerateBlock(0, Body(0), Noise(40, VideoRows), Values(41, AudioRows * AudioIn), 2, 1.0);
        (float[] video1, float[] audio1) = session.GenerateBlock(1, Body(1), Noise(50, VideoRows), Values(51, AudioRows * AudioIn), 2, 1.0);

        (float[] fullVideo0, float[] fullAudio0) = DuoChunkSession.FullBlock(transformer, backend, sample, 0, DuoRopePolicy.Old,
            Shared0(), [Body(0), Body(1)], reference, [], [], Noise(40, VideoRows), Values(41, AudioRows * AudioIn), 2, 1.0);
        (float[] fullVideo1, float[] fullAudio1) = DuoChunkSession.FullBlock(transformer, backend, sample, 1, DuoRopePolicy.Old,
            Shared0(), [Body(0), Body(1)], reference, video0, audio0, Noise(50, VideoRows), Values(51, AudioRows * AudioIn), 2, 1.0);

        AssertClose(fullVideo0, video0, Tolerance, "block 0 video");
        AssertClose(fullAudio0, audio0, Tolerance, "block 0 audio");
        AssertClose(fullVideo1, video1, Tolerance, "block 1 video");
        AssertClose(fullAudio1, audio1, Tolerance, "block 1 audio");
    }

    [Fact]
    public void ShiftedCleanBlockZeroIsDetected()
    {
        // The same check against a wrong history: block 0 shifted by 0.5 in the full forward. The cached block 1 must NOT match it.
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoSampleInput sample = Sample();
        DuoChunkSession session = new(transformer, backend, sample, DuoRopePolicy.Old, Cache());
        float[] reference = Values(30, RefRows * VideoIn);
        session.Prefill(Shared0(), reference);
        (float[] video0, float[] audio0) = session.GenerateBlock(0, Body(0), Noise(40, VideoRows), Values(41, AudioRows * AudioIn), 2, 1.0);
        (float[] video1, _) = session.GenerateBlock(1, Body(1), Noise(50, VideoRows), Values(51, AudioRows * AudioIn), 2, 1.0);

        float[] shifted = video0.Select(v => v + 0.5f).ToArray();
        (float[] wrongVideo1, _) = DuoChunkSession.FullBlock(transformer, backend, sample, 1, DuoRopePolicy.Old,
            Shared0(), [Body(0), Body(1)], reference, shifted, audio0, Noise(50, VideoRows), Values(51, AudioRows * AudioIn), 2, 1.0);

        Assert.True(MaxDiff(video1, wrongVideo1) > Sensitivity,
            $"A shifted block 0 changes block 1 by only {MaxDiff(video1, wrongVideo1)}; the agreement check cannot see the history.");
    }

    [Fact]
    public void EmptyHistoryIsDetectedByTheAgreementCheck()
    {
        // Block 1 against a forward with no committed block 0 at all (its clean media removed, text kept). It must NOT agree.
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoSampleInput sample = Sample();
        DuoChunkSession session = new(transformer, backend, sample, DuoRopePolicy.Old, Cache());
        float[] reference = Values(30, RefRows * VideoIn);
        session.Prefill(Shared0(), reference);
        session.GenerateBlock(0, Body(0), Noise(40, VideoRows), Values(41, AudioRows * AudioIn), 2, 1.0);
        (float[] video1, _) = session.GenerateBlock(1, Body(1), Noise(50, VideoRows), Values(51, AudioRows * AudioIn), 2, 1.0);

        // With no block 0 history: the full forward sees the reference and block 1 only.
        (float[] empty1, _) = DuoChunkSession.FullBlock(transformer, backend, sample, 1, DuoRopePolicy.Old,
            Shared0(), [Body(0), Body(1)], reference, Enumerable.Repeat(0f, VideoRows * VideoIn).ToArray(), Enumerable.Repeat(0f, AudioRows * AudioIn).ToArray(),
            Noise(50, VideoRows), Values(51, AudioRows * AudioIn), 2, 1.0);

        Assert.True(MaxDiff(video1, empty1) > Sensitivity,
            $"An empty history changes block 1 by only {MaxDiff(video1, empty1)}; the agreement check cannot see the history.");
    }

    [Fact]
    public void BlocksMustComeInOrderFromTheFirst()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoChunkSession session = new(transformer, backend, Sample(), DuoRopePolicy.Old, Cache());

        Assert.Throws<InvalidOperationException>(() => session.GenerateBlock(0, Body(0), Noise(40, VideoRows), Values(41, AudioRows * AudioIn), 2, 1.0));

        session.Prefill(Shared0(), Values(30, RefRows * VideoIn));
        Assert.Throws<InvalidOperationException>(() => session.GenerateBlock(1, Body(1), Noise(50, VideoRows), Values(51, AudioRows * AudioIn), 2, 1.0));

        session.GenerateBlock(0, Body(0), Noise(40, VideoRows), Values(41, AudioRows * AudioIn), 2, 1.0);
        Assert.Throws<InvalidOperationException>(() => session.GenerateBlock(0, Body(0), Noise(40, VideoRows), Values(41, AudioRows * AudioIn), 2, 1.0));
        Assert.Equal(1, session.NextBlock);
    }

    [Fact]
    public void WrongLengthsAreRefusedBeforeAnythingIsComputed()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoChunkSession session = new(transformer, backend, Sample(), DuoRopePolicy.Old, Cache());

        Assert.Throws<ArgumentException>(() => session.Prefill(Values(2, Shared * TextDim + 1), Values(30, RefRows * VideoIn)));
        Assert.Throws<ArgumentException>(() => session.Prefill(Shared0(), Values(30, RefRows * VideoIn + 3)));
        session.Prefill(Shared0(), Values(30, RefRows * VideoIn));
        Assert.Throws<InvalidOperationException>(() => session.Prefill(Shared0(), Values(30, RefRows * VideoIn)));

        Assert.Throws<ArgumentException>(() => session.GenerateBlock(0, Values(10, BodyRows * TextDim + 2), Noise(40, VideoRows), Values(41, AudioRows * AudioIn), 2, 1.0));
        Assert.Throws<ArgumentException>(() => session.GenerateBlock(0, Body(0), Noise(40, VideoRows + 1), Values(41, AudioRows * AudioIn), 2, 1.0));
        Assert.Throws<ArgumentException>(() => session.GenerateBlock(0, Body(0), Noise(40, VideoRows), Values(41, AudioRows * AudioIn - 1), 2, 1.0));
        Assert.Equal(0, session.NextBlock);
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

    private static DuoStreamCache Cache() => new(blocks: Layers, width: Inner, capacityRows: 64);

    /// <summary>Three blocks, so a rollout can outgrow a cache that holds the prefix and one block.</summary>
    private static DuoSampleInput ThreeBlockSample() => Sample() with
    {
        Blocks =
        [
            .. Sample().Blocks,
            new DuoBlockInput { StartMs = 1416.0, BodyTextRows = BodyRows, Video = new DuoGrid(2, 4, 4), AudioSteps = 3, FrameStartsMs = [1416.0, 1516.0] },
        ],
    };

    /// <summary>Reference and shared rows (7) plus one clean block (8 video and 6 audio rows): a cache of this size holds exactly one.</summary>
    private const int OneBlockCapacity = 7 + 14;

    [Fact]
    public void LongRolloutSlidesTheWindowAndEachBlockMatchesTheWindowedForward()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoSampleInput sample = ThreeBlockSample();
        float[] reference = Values(30, RefRows * VideoIn);
        DuoStreamCache cache = new(blocks: Layers, width: Inner, capacityRows: OneBlockCapacity);
        DuoChunkSession session = new(transformer, backend, sample, DuoRopePolicy.Old, cache);
        session.Prefill(Shared0(), reference);
        Assert.Equal(Shared + RefRows, cache.Length);

        float[][] video = new float[3][];
        float[][] audio = new float[3][];
        for (int b = 0; b < 3; b++)
        {
            (video[b], audio[b]) = session.GenerateBlock(b, Body(b), Noise(40 + 10 * b, VideoRows), Values(41 + 10 * b, AudioRows * AudioIn), 2, 1.0);
            Assert.True(cache.Length <= OneBlockCapacity, "The cache never exceeds its capacity.");
        }
        // Once block 1 is committed, block 0 has gone: the cache holds the prefix and block 2 is about to see block 1 only.
        Assert.Equal(OneBlockCapacity, cache.Length);

        // Block 2 is denoised against the windowed history: reference, shared rows and clean block 1 only.
        DuoSampleInput window = sample with { Blocks = [sample.Blocks[1], sample.Blocks[2]] };
        (float[] windowVideo, _) = DuoChunkSession.FullBlock(transformer, backend, window, 1, DuoRopePolicy.Old,
            Shared0(), [Body(1), Body(2)], reference, video[1], audio[1], Noise(60, VideoRows), Values(61, AudioRows * AudioIn), 2, 1.0);
        AssertClose(windowVideo, video[2], Tolerance, "block 2 video against the windowed forward");
    }

    [Fact]
    public void EvictionTakesTheWholeOldestBlockIncludingItsAudio()
    {
        // The prefix, one block, and 8 spare rows: the second block forces eviction, and a video-only eviction would leave 6 audio rows.
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoSampleInput sample = ThreeBlockSample();
        float[] reference = Values(30, RefRows * VideoIn);
        const int capacity = OneBlockCapacity + 8;
        DuoStreamCache cache = new(blocks: Layers, width: Inner, capacityRows: capacity);
        DuoChunkSession session = new(transformer, backend, sample, DuoRopePolicy.Old, cache);
        session.Prefill(Shared0(), reference);
        for (int b = 0; b < 3; b++)
        {
            session.GenerateBlock(b, Body(b), Noise(40 + 10 * b, VideoRows), Values(41 + 10 * b, AudioRows * AudioIn), 2, 1.0);
            // Only one block is ever held beside the prefix, and both of its groups go with it.
            Assert.Equal(Shared + RefRows + VideoRows + AudioRows, cache.Length);
            Assert.All(cache.Groups.Where(g => g.Kind == DuoGroupKind.Clean), g => Assert.Equal(b, g.Block));
        }
    }

    [Fact]
    public void EvictingWithNoCleanBlockIsRefused()
    {
        DuoStreamCache cache = new(blocks: Layers, width: Inner, capacityRows: 64);
        Assert.Throws<InvalidOperationException>(() => cache.EvictOldestClean());
    }

    [Fact]
    public void BlockThatCannotFitIsRefusedBeforeAnyEviction()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        // One row short: the prefix and a clean block do not fit, and no clean block is held to drop.
        DuoStreamCache cache = new(blocks: Layers, width: Inner, capacityRows: OneBlockCapacity - 1);
        DuoChunkSession session = new(transformer, backend, Sample(), DuoRopePolicy.Old, cache);
        session.Prefill(Shared0(), Values(30, RefRows * VideoIn));

        Assert.Throws<InvalidOperationException>(() => session.GenerateBlock(0, Body(0), Noise(40, VideoRows), Values(41, AudioRows * AudioIn), 2, 1.0));
        Assert.Equal(Shared + RefRows, cache.Length);
        Assert.Equal(0, session.NextBlock);
    }

    private static float[] Shared0() => Values(2, Shared * TextDim);

    private static float[] Body(int block) => Values(10 + block, BodyRows * TextDim);

    private static float[] Noise(int seed, int rows) => Values(seed, rows * VideoIn);

    private static float MaxDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float max = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        }
        return max;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        float max = MaxDiff(expected, actual);
        Assert.True(max <= tolerance, $"{what} differs from the full forward by {max}.");
    }

    private static MiniMaxH3Transformer Transformer()
    {
        MiniMaxH3Transformer transformer = new(new MiniMaxH3Config
        {
            HiddenSize = Hidden,
            NumLayers = Layers,
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
        Dictionary<string, Tensor> weights = new()
        {
            ["video_patch_proj.weight"] = Weight(Hidden, VideoIn, 10, 1f),
            ["audio_patch_proj.weight"] = Weight(Hidden, AudioIn, 11, 1f),
            ["condition_proj.weight"] = Weight(Hidden, TextDim, 12, 1f),
            ["time_embedder.proj_in.weight"] = Weight(TimeEmbedHidden, 256, 13, 1f),
            ["time_embedder.proj_out.weight"] = Weight(TimeEmbedDim, TimeEmbedHidden, 14, 1f),
            ["token_refiner.blocks.0.norm1.weight"] = Ones(Hidden),
            ["token_refiner.blocks.0.norm2.weight"] = Ones(Hidden),
            ["token_refiner.blocks.0.attn.qkv_proj.weight"] = Weight(3 * Inner, Hidden, 15, 1f),
            ["token_refiner.blocks.0.attn.q_norm.weight"] = Ones(HeadDim),
            ["token_refiner.blocks.0.attn.k_norm.weight"] = Ones(HeadDim),
            ["token_refiner.blocks.0.attn.out_proj.weight"] = Weight(Hidden, Inner, 16, 1f),
            ["token_refiner.blocks.0.mlp.fc1.weight"] = Weight(2 * Ffn, Hidden, 17, 1f),
            ["token_refiner.blocks.0.mlp.fc2.weight"] = Weight(Hidden, Ffn, 18, 1f),
            ["token_refiner.final_norm.weight"] = Ones(Hidden),
            ["final_layer.norm.weight"] = Ones(Hidden),
            ["final_layer.adaln_proj.linear.weight"] = Weight(2 * Hidden, TimeEmbedDim, 24, 0.5f),
            ["final_layer.video_out.weight"] = Weight(VideoIn, Hidden, 25, 1f),
            ["final_layer.audio_out.weight"] = Weight(AudioIn, Hidden, 26, 1f),
        };
        for (int layer = 0; layer < Layers; layer++)
        {
            int s = 100 * layer;
            string p = $"blocks.{layer}";
            weights[$"{p}.norm1.weight"] = Ones(Hidden);
            weights[$"{p}.norm2.weight"] = Ones(Hidden);
            weights[$"{p}.adaln_proj.linear.weight"] = Weight(6 * Hidden * 3, TimeEmbedDim, 19 + s, 0.5f);
            weights[$"{p}.attn.qkv_proj.weight"] = Weight(3 * Inner, Hidden, 20 + s, 1f);
            weights[$"{p}.attn.q_norm.weight"] = Ones(HeadDim);
            weights[$"{p}.attn.k_norm.weight"] = Ones(HeadDim);
            weights[$"{p}.attn.out_proj.weight"] = Weight(Hidden, Inner, 21 + s, 1f);
            weights[$"{p}.mlp.fc1.weight"] = Weight(2 * Ffn, Hidden, 22 + s, 1f);
            weights[$"{p}.mlp.fc2.weight"] = Weight(Hidden, Ffn, 23 + s, 1f);
        }
        transformer.LoadWeights(weights);
        return transformer;
    }

    private static Tensor Weight(int rows, int cols, int seed, float scale) =>
        MakeTensor(new TensorShape(rows, cols), Values(seed, rows * cols, scale));

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

    /// <summary>Hash-based values in [-scale, scale]: decorrelated across rows and seeds, unlike a short periodic sine.</summary>
    private static float[] Values(int seed, int count, float scale = 1f)
    {
        float[] data = new float[count];
        for (int i = 0; i < count; i++)
        {
            double h = Math.Sin(seed * 12.9898 + i * 78.233) * 43758.5453;
            double frac = h - Math.Floor(h);
            data[i] = scale * (float)(2.0 * frac - 1.0);
        }
        return data;
    }
}

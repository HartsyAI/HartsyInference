using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>Self-consistency of the Duo forward on the H3 stack, on a tiny geometry with random weights. These show that the
/// visibility rules reach the output; they are not parity with upstream Duo (that is the L0/L1 step, against the Python reference).</summary>
public sealed class MiniMaxH3DuoForwardTests
{
    private const int Hidden = 8, Heads = 2, HeadDim = 96, Inner = Heads * HeadDim, Ffn = 16, TextDim = 5;
    private const int VideoIn = 96, AudioIn = 32, TimeEmbedDim = 6, TimeEmbedHidden = 8;
    private const int VideoRowCount = 12, AudioRowCount = 6, TextRowCount = 5;

    [Fact]
    public void CausalReferenceRowsDoNotSeeNoisyRows()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        float[] video = Values(1, VideoRowCount * VideoIn);
        float[] perturbed = (float[])video.Clone();
        // Video row 4 of the layout is packed row 9, a noisy row of time 1.
        for (int c = 0; c < VideoIn; c++)
        {
            perturbed[4 * VideoIn + c] += 0.5f;
        }

        float[] baseline = Run(transformer, backend, layout, DuoAttentionMode.Causal, video);
        float[] changed = Run(transformer, backend, layout, DuoAttentionMode.Causal, perturbed);

        // Reference rows (video rows 0..3) read only reference and shared keys, so a later noisy row cannot reach them.
        AssertRowsEqual(baseline, changed, 0, 4);
        // The noisy row itself does move, so the perturbation is real.
        Assert.True(MaxDiff(baseline, changed, 4, 5) > 1e-4f);
    }

    [Fact]
    public void BidirectionalReferenceRowsSeeNoisyRows()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        float[] video = Values(1, VideoRowCount * VideoIn);
        float[] perturbed = (float[])video.Clone();
        for (int c = 0; c < VideoIn; c++)
        {
            perturbed[4 * VideoIn + c] += 0.5f;
        }

        float[] baseline = Run(transformer, backend, layout, DuoAttentionMode.Bidirectional, video);
        float[] changed = Run(transformer, backend, layout, DuoAttentionMode.Bidirectional, perturbed);

        // With no mask, the reference rows read every key, so the noisy row moves them. The effect is small (one key of 128 in a
        // softmax with small weights), so the bound is "moved at all", not a size; a mask that dropped the key would leave it at exactly zero.
        float moved = MaxDiff(baseline, changed, 0, 4);
        Assert.True(moved > 0f, $"reference rows did not move: {moved}.");
    }

    [Fact]
    public void RoleEmbeddingsAreRefused()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        using Tensor text = MakeTensor(new TensorShape(TextRowCount, TextDim), Values(2, TextRowCount * TextDim));
        using Tensor video = MakeTensor(new TensorShape(VideoRowCount, VideoIn), Values(3, VideoRowCount * VideoIn));
        using Tensor audio = MakeTensor(new TensorShape(AudioRowCount, AudioIn), Values(4, AudioRowCount * AudioIn));

        Assert.Throws<NotSupportedException>(() => transformer.ForwardDuo(backend, layout, DuoAttentionMode.Causal,
            RefinerGroups(), text, video, audio, Times(), roleEmbeddingEnabled: true));
    }

    [Fact]
    public void MissingAudioRowsAreRefusedWhenTheLayoutHasAudio()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        using Tensor text = MakeTensor(new TensorShape(TextRowCount, TextDim), Values(2, TextRowCount * TextDim));
        using Tensor video = MakeTensor(new TensorShape(VideoRowCount, VideoIn), Values(3, VideoRowCount * VideoIn));

        Assert.Throws<ArgumentException>(() => transformer.ForwardDuo(backend, layout, DuoAttentionMode.Causal,
            RefinerGroups(), text, video, audioRows: null, Times()));
    }

    [Fact]
    public void LargeClockPositionsStayFiniteAndCausal()
    {
        // A clock near 1e6 (about 11 hours of 25 ms ticks) must not lose the RoPE angle to double-to-F32 rounding
        // or overflow the path: the output stays finite, and the reference rows still do not see the noisy row.
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        double[] positions = layout.Positions;
        for (int r = 0; r < layout.Length; r++)
        {
            positions[r * 3] += 1_000_000.0;
        }
        float[] video = Values(1, VideoRowCount * VideoIn);
        float[] perturbed = (float[])video.Clone();
        for (int c = 0; c < VideoIn; c++)
        {
            perturbed[4 * VideoIn + c] += 0.5f;
        }

        float[] baseline = Run(transformer, backend, layout, DuoAttentionMode.Causal, video);
        float[] changed = Run(transformer, backend, layout, DuoAttentionMode.Causal, perturbed);

        Assert.All(baseline, value => Assert.True(float.IsFinite(value)));
        AssertRowsEqual(baseline, changed, 0, 4);
    }

    [Fact]
    public void AudioHeadReturnsOneRowPerAudioRowAtItsPatchWidth()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        using Tensor text = MakeTensor(new TensorShape(TextRowCount, TextDim), Values(2, TextRowCount * TextDim));
        using Tensor video = MakeTensor(new TensorShape(VideoRowCount, VideoIn), Values(3, VideoRowCount * VideoIn));
        using Tensor audio = MakeTensor(new TensorShape(AudioRowCount, AudioIn), Values(4, AudioRowCount * AudioIn));

        (Tensor v, Tensor? a) = transformer.ForwardDuo(backend, layout, DuoAttentionMode.Causal, RefinerGroups(), text, video, audio, Times());
        using (v)
        using (a)
        {
            Assert.NotNull(a);
            Assert.Equal(VideoRowCount, (int)v.Shape[0]);
            Assert.Equal(VideoIn, (int)v.Shape[1]);
            Assert.Equal(AudioRowCount, (int)a!.Shape[0]);
            Assert.Equal(AudioIn, (int)a.Shape[1]);
            Assert.All(a.AsSpan<float>().ToArray(), value => Assert.True(float.IsFinite(value)));
        }
    }

    [Fact]
    public void TimeIdBeyondTheGivenTimestepsIsRefused()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        using Tensor text = MakeTensor(new TensorShape(TextRowCount, TextDim), Values(2, TextRowCount * TextDim));
        using Tensor video = MakeTensor(new TensorShape(VideoRowCount, VideoIn), Values(3, VideoRowCount * VideoIn));
        using Tensor audio = MakeTensor(new TensorShape(AudioRowCount, AudioIn), Values(4, AudioRowCount * AudioIn));

        // The layout's noisy rows name timestep 1, so a single timestep is one too few.
        Assert.Throws<ArgumentOutOfRangeException>(() => transformer.ForwardDuo(backend, layout, DuoAttentionMode.Causal,
            RefinerGroups(), text, video, audio, times: [0f]));
    }

    [Fact]
    public void RefinerGroupBeyondTheTextRowsIsRefused()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        using Tensor text = MakeTensor(new TensorShape(TextRowCount, TextDim), Values(2, TextRowCount * TextDim));
        using Tensor video = MakeTensor(new TensorShape(VideoRowCount, VideoIn), Values(3, VideoRowCount * VideoIn));
        using Tensor audio = MakeTensor(new TensorShape(AudioRowCount, AudioIn), Values(4, AudioRowCount * AudioIn));
        IReadOnlyList<DuoGroup> tooLong = [new DuoGroup(0, 0, DuoGroupKind.Text, DuoModality.None, 0, TextRowCount + 4)];

        Assert.Throws<ArgumentException>(() => transformer.ForwardDuo(backend, layout, DuoAttentionMode.Causal,
            tooLong, text, video, audio, Times()));
    }

    [Fact]
    public void TiledBiasMatchesTheDenseBiasCausal()
    {
        // The dense route (one [S, S] bias) and the tiled route (a [tile, S] bias per query tile) must agree on the same layout.
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        float[] video = Values(1, VideoRowCount * VideoIn);

        float[] dense = Run(transformer, backend, layout, DuoAttentionMode.Causal, video);
        float[] tiled = Run(transformer, backend, layout, DuoAttentionMode.Causal, video, chunkRows: 32);

        Assert.True(MaxDiff(dense, tiled, 0, VideoRowCount) <= 1e-5f, $"tiled differs from dense by {MaxDiff(dense, tiled, 0, VideoRowCount)}.");
    }

    [Fact]
    public void TiledBiasHandlesAPartialLastTileAndBidirectionalMode()
    {
        // 40 does not divide the 128-row layout, so the last tile is 8 rows; the bidirectional mask has no zero tiles.
        using MiniMaxH3Transformer transformer = Transformer();
        using IBackend backend = new CpuBackend();
        DuoLayout layout = DuoLayoutBuilder.Build([OneBlock()], DuoRopePolicy.Old);
        float[] video = Values(1, VideoRowCount * VideoIn);

        float[] dense = Run(transformer, backend, layout, DuoAttentionMode.Bidirectional, video);
        float[] tiled = Run(transformer, backend, layout, DuoAttentionMode.Bidirectional, video, chunkRows: 40);

        Assert.True(MaxDiff(dense, tiled, 0, VideoRowCount) <= 1e-5f, $"tiled differs from dense by {MaxDiff(dense, tiled, 0, VideoRowCount)}.");
    }

    private static float[] Run(MiniMaxH3Transformer transformer, IBackend backend, DuoLayout layout, DuoAttentionMode mode,
        float[] videoValues, int? chunkRows = null)
    {
        using Tensor text = MakeTensor(new TensorShape(TextRowCount, TextDim), Values(2, TextRowCount * TextDim));
        using Tensor video = MakeTensor(new TensorShape(VideoRowCount, VideoIn), videoValues);
        using Tensor audio = MakeTensor(new TensorShape(AudioRowCount, AudioIn), Values(4, AudioRowCount * AudioIn));
        (Tensor v, Tensor? a) = transformer.ForwardDuo(backend, layout, mode, RefinerGroups(), text, video, audio, Times(),
            chunkRowsOverride: chunkRows);
        float[] result = v.AsSpan<float>().ToArray();
        v.Dispose();
        a?.Dispose();
        return result;
    }

    private static IReadOnlyList<DuoGroup> RefinerGroups() => [new DuoGroup(0, 0, DuoGroupKind.Text, DuoModality.None, 0, TextRowCount)];

    private static float[] Times() => [0f, 0.7f];

    private static DuoSampleInput OneBlock() => new()
    {
        SampleId = 7,
        SharedDescriptionRows = 2,
        SharedCueRows = 1,
        Reference = new DuoGrid(1, 4, 4),
        Blocks =
        [
            new DuoBlockInput
            {
                StartMs = 0.0,
                BodyTextRows = 2,
                Video = new DuoGrid(2, 4, 4),
                AudioSteps = 3,
                FrameStartsMs = [0.0, 100.0],
            },
        ],
        CleanBlocks = 0,
        NoisyBlocks = [0],
    };

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

    private static float[] Values(int seed, int count, float scale = 1f)
    {
        float[] data = new float[count];
        for (int i = 0; i < count; i++)
        {
            data[i] = scale * MathF.Sin(seed * 1.7f + i * 0.37f);
        }
        return data;
    }

    private static Tensor Ones(int count)
    {
        float[] data = new float[count];
        Array.Fill(data, 1f);
        return MakeTensor(new TensorShape(count), data);
    }

    private static float MaxDiff(float[] a, float[] b, int fromRow, int toRow)
    {
        float max = 0f;
        for (int i = fromRow * VideoIn; i < toRow * VideoIn; i++)
        {
            max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        }
        return max;
    }

    private static void AssertRowsEqual(float[] a, float[] b, int fromRow, int toRow)
    {
        float diff = MaxDiff(a, b, fromRow, toRow);
        Assert.True(diff <= 1e-6f, $"rows [{fromRow}, {toRow}) differ by {diff}.");
    }
}

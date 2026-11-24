using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using Xunit;
using static HartsyInference.Diffusion.Tests.TinyWanCheckpoint;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The frame-wise causal Wan forward that FastFrames (and the Causal Forcing generator it builds on) runs: one latent
/// frame per pass against a per-layer key/value cache. These run on a tiny random checkpoint on the CPU in F32 and show the cache
/// and RoPE bookkeeping equals one masked forward over the whole clip. They do not show the numbers match the upstream Python;
/// that needs the real weights and the reference dump.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class FastFramesFrameWiseTests
{
    private const int TextLength = 5;
    private const int Frames = 3;
    private const int Height = 4;
    private const int Width = 4;
    private const int TokensPerFrame = Height / 2 * (Width / 2);

    private static WanVideoConfig TinyConfig => new()
    {
        NumHeads = Heads, HeadDim = HeadDim, NumLayers = Layers, FfnDim = Ffn,
        InChannels = Channels, OutChannels = Channels, VaeLatentChannels = Channels,
        TextDim = TextDim, FreqDim = FreqDim, VaeSpatialCompression = 8,
    };

    [Fact]
    public void CachedFramesEqualOneFrameCausalForward()
    {
        float[] timesteps = [1000f, 833.3333f, 625f];
        using Rig rig = new();
        Tensor[] frames = rig.RandomFrames(Frames, seed: 21);

        using Tensor joint = rig.Transformer.ForwardFrameCausal(rig.Backend, Stack(frames), rig.Text, timesteps);
        using FrameKvCache cache = new(rig.Backend, Layers, Frames);
        for (int f = 0; f < Frames; f++)
        {
            WanVideoTransformer.FrameForward step = rig.Transformer.ForwardFrame(rig.Backend, frames[f], rig.Text, timesteps[f], f, cache);
            AssertClose(FrameOf(joint, f), step.Velocity.AsReadOnlySpan<float>(), $"frame {f} velocity");
            Assert.Equal(new TensorShape([(long)TokensPerFrame, Inner]), step.Hidden.Shape);
            Assert.Equal(new TensorShape([1L, Inner]), step.Temb.Shape);
            step.Velocity.Dispose();
            step.Hidden.Dispose();
            step.Temb.Dispose();
        }
        foreach (Tensor frame in frames)
            frame.Dispose();
    }

    /// <summary>A second pass of a finished frame, at timestep 0 on its clean latent, leaves that pass's keys and values in the
    /// slot: the next frame then reads them, as the clean-context pass of the generator does.</summary>
    [Fact]
    public void ARePassOfAFrameReplacesItsSlot()
    {
        using Rig rig = new();
        Tensor[] frames = rig.RandomFrames(2, seed: 31);
        using Tensor clean = rig.RandomFrames(1, seed: 32)[0];

        using FrameKvCache cache = new(rig.Backend, Layers, 2);
        Dispose(rig.Transformer.ForwardFrame(rig.Backend, frames[0], rig.Text, 1000f, 0, cache));
        Dispose(rig.Transformer.ForwardFrame(rig.Backend, clean, rig.Text, 0f, 0, cache));
        WanVideoTransformer.FrameForward second = rig.Transformer.ForwardFrame(rig.Backend, frames[1], rig.Text, 833.3333f, 1, cache);

        using Tensor joint = rig.Transformer.ForwardFrameCausal(rig.Backend, Stack([clean, frames[1]]), rig.Text, [0f, 833.3333f]);
        AssertClose(FrameOf(joint, 1), second.Velocity.AsReadOnlySpan<float>(), "frame 1 after a clean re-pass of frame 0");
        Dispose(second);
        foreach (Tensor frame in frames)
            frame.Dispose();
    }

    [Fact]
    public void ACacheJoinsItsFramesInOrderAndRefusesAnEmptySlot()
    {
        using CpuBackend backend = new();
        using FrameKvCache cache = new(backend, layers: 1, maxFrames: 3);
        for (int f = 0; f < 2; f++)
        {
            Tensor k = new(new TensorShape([2L, 4]), DType.F32);
            Tensor v = new(new TensorShape([2L, 4]), DType.F32);
            k.AsSpan<float>().Fill(f + 1);
            v.AsSpan<float>().Fill(-(f + 1));
            cache.Set(0, f, k, v);
        }

        (Tensor keys, Tensor values) = cache.JoinPrefix(0, 1);
        using (keys)
        using (values)
        {
            Assert.Equal(new TensorShape([4L, 4]), keys.Shape);
            Assert.Equal([1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 2f, 2f, 2f, 2f, 2f, 2f, 2f, 2f], keys.AsReadOnlySpan<float>().ToArray());
            Assert.Equal(-2f, values.AsReadOnlySpan<float>()[15]);
        }
        Assert.False(cache.Has(0, 2));
        Assert.Throws<InvalidOperationException>(() => cache.JoinPrefix(0, 2));
    }

    /// <summary>The Wan block hands the rewrite its K and V as <c>[S, C]</c> with no batch axis; a join on the second axis would
    /// widen the channels and corrupt the attention that reads it.</summary>
    [Fact]
    public void ASlotWithoutABatchAxisJoinsOnTheTokenAxis()
    {
        using CpuBackend backend = new();
        using FrameKvCache cache = new(backend, layers: 1, maxFrames: 2);
        for (int f = 0; f < 2; f++)
        {
            Tensor k = new(new TensorShape([2L, 3]), DType.F32);
            Tensor v = new(new TensorShape([2L, 3]), DType.F32);
            k.AsSpan<float>().Fill(f + 1);
            v.AsSpan<float>().Fill(f + 1);
            cache.Set(0, f, k, v);
        }

        (Tensor keys, Tensor values) = cache.JoinPrefix(0, 1);
        using (keys)
        using (values)
        {
            Assert.Equal(new TensorShape([4L, 3]), keys.Shape);
            Assert.Equal([1f, 1f, 1f, 1f, 1f, 1f, 2f, 2f, 2f, 2f, 2f, 2f], keys.AsReadOnlySpan<float>().ToArray());
        }
    }

    [Fact]
    public void AMismatchedKeyAndValueAreRefused()
    {
        using CpuBackend backend = new();
        using FrameKvCache cache = new(backend, layers: 1, maxFrames: 1);

        Assert.Throws<ArgumentException>(() => cache.Set(0, 0, new Tensor(new TensorShape([2L, 4]), DType.F32), new Tensor(new TensorShape([3L, 4]), DType.F32)));
    }

    [Fact]
    public void TheGridMapsTheReleasedTimestepsToTheirSigmas()
    {
        FastFramesFlowSchedule schedule = new(5f);

        // The generator's nominal [1000, 750, 500, 250], shifted by 5: 1000, 937.5, 833.33, 625.
        Assert.Equal(1f, schedule.Sigma(1000f), 6);
        Assert.Equal(0.9375f, schedule.Sigma(937.5f), 6);
        Assert.Equal(5f / 6f, schedule.Sigma(833.3333129882812f), 6);
        Assert.Equal(0.625f, schedule.Sigma(625f), 6);
        // The 14B generator's rounded 937 and 833 land on the neighbouring grid entries.
        Assert.InRange(Math.Abs(schedule.GridIndex(937f) - schedule.GridIndex(937.5f)), 0, 2);
        Assert.InRange(Math.Abs(schedule.GridIndex(833f) - schedule.GridIndex(833.3333f)), 0, 1);
        Assert.Equal(0, schedule.GridIndex(1000f));
        Assert.Equal(1000f, schedule.Timestep(0));
    }

    [Fact]
    public void TheLookupOfZeroPicksTheSmallestGridSigma()
    {
        FastFramesFlowSchedule schedule = new(5f);

        // The reference looks timestep 0 up on the grid, which ends at σ ≈ 0.005 (timestep ≈ 4.98), not at 0.
        Assert.Equal(999, schedule.GridIndex(0f));
        Assert.InRange(schedule.Sigma(0f), 0.0049f, 0.0051f);
    }

    [Fact]
    public void ASlotMustBeTokenMajorWithOneChannelCountAndTheCacheRefusesWorkAfterDisposal()
    {
        using CpuBackend backend = new();
        FrameKvCache cache = new(backend, layers: 1, maxFrames: 2);
        Assert.Throws<ArgumentException>(() => cache.Set(0, 0, new Tensor(new TensorShape([1L, 2, 4]), DType.F32), new Tensor(new TensorShape([1L, 2, 4]), DType.F32)));
        cache.Set(0, 0, new Tensor(new TensorShape([2L, 4]), DType.F32), new Tensor(new TensorShape([2L, 4]), DType.F32));
        Assert.Throws<ArgumentException>(() => cache.Set(0, 1, new Tensor(new TensorShape([2L, 3]), DType.F32), new Tensor(new TensorShape([2L, 3]), DType.F32)));
        cache.Dispose();
        Assert.Throws<ObjectDisposedException>(() => cache.Set(0, 1, new Tensor(new TensorShape([2L, 4]), DType.F32), new Tensor(new TensorShape([2L, 4]), DType.F32)));
    }

    /// <summary>A frame whose earlier frame is not cached is refused before anything is written: the cache is as it was, and the
    /// frame's own slot stays empty, so a later correct call still works.</summary>
    [Fact]
    public void AFrameWithAnUncachedEarlierFrameIsRefusedBeforeAnythingIsWritten()
    {
        using Rig rig = new();
        Tensor[] frames = rig.RandomFrames(3, seed: 51);
        using FrameKvCache cache = new(rig.Backend, Layers, 3);
        Assert.Throws<InvalidOperationException>(() => rig.Transformer.ForwardFrame(rig.Backend, frames[2], rig.Text, 625f, 2, cache));
        for (int layer = 0; layer < Layers; layer++)
            Assert.False(cache.Has(layer, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => rig.Transformer.ForwardFrame(rig.Backend, frames[0], rig.Text, 1000f, 3, cache));
        Dispose(rig.Transformer.ForwardFrame(rig.Backend, frames[0], rig.Text, 1000f, 0, cache));
        Dispose(rig.Transformer.ForwardFrame(rig.Backend, frames[1], rig.Text, 833f, 1, cache));
        Dispose(rig.Transformer.ForwardFrame(rig.Backend, frames[2], rig.Text, 625f, 2, cache));
        foreach (Tensor frame in frames)
            frame.Dispose();
    }

    /// <summary>The step-cache path is the one that passed no self-attention mask to block 0. With an empty step cache it computes
    /// every block, so it must equal the plain masked forward.</summary>
    [Fact]
    public void TheStepCachePathKeepsTheMask()
    {
        float[] timesteps = [1000f, 833.3333f, 625f];
        using Rig rig = new();
        Tensor[] frames = rig.RandomFrames(Frames, seed: 41);
        using Tensor clip = Stack(frames);
        using Tensor plain = rig.Transformer.ForwardFrameCausal(rig.Backend, clip, rig.Text, timesteps);
        using DeviceFeatureCache stepCache = new();
        using Tensor cached = rig.Transformer.ForwardFrameCausal(rig.Backend, clip, rig.Text, timesteps, stepCache);
        AssertClose(plain.AsReadOnlySpan<float>().ToArray(), cached.AsReadOnlySpan<float>(), "step-cache path against the plain masked forward");
        foreach (Tensor frame in frames)
            frame.Dispose();
    }

    private static void Dispose(WanVideoTransformer.FrameForward step)
    {
        step.Velocity.Dispose();
        step.Hidden.Dispose();
        step.Temb.Dispose();
    }

    private static Tensor Stack(Tensor[] frames)
    {
        int count = frames.Length;
        Tensor clip = new(new TensorShape([1L, Channels, count, Height, Width]), DType.F32);
        Span<float> dst = clip.AsSpan<float>();
        for (int f = 0; f < count; f++)
        {
            ReadOnlySpan<float> src = frames[f].AsReadOnlySpan<float>();
            for (int c = 0; c < Channels; c++)
                src.Slice(c * Height * Width, Height * Width).CopyTo(dst.Slice((c * count + f) * Height * Width, Height * Width));
        }
        return clip;
    }

    private static float[] FrameOf(Tensor clip, int frame)
    {
        int count = (int)clip.Shape[2];
        ReadOnlySpan<float> src = clip.AsReadOnlySpan<float>();
        float[] values = new float[Channels * Height * Width];
        for (int c = 0; c < Channels; c++)
            src.Slice((c * count + frame) * Height * Width, Height * Width).CopyTo(values.AsSpan(c * Height * Width, Height * Width));
        return values;
    }

    private static void AssertClose(float[] expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = 0, error = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{what} holds a non-finite value");
            scale += (double)expected[i] * expected[i];
            error += (double)(expected[i] - actual[i]) * (expected[i] - actual[i]);
        }
        double relative = Math.Sqrt(error / Math.Max(scale, 1e-30));
        Assert.True(relative < 1e-4, $"{what}: relative RMS error {relative:E2} against the one-pass causal forward");
    }

    private sealed class Rig : IDisposable
    {
        private readonly TinyWanCheckpoint _checkpoint = new();

        public Rig()
        {
            Transformer = new WanVideoTransformer(TinyConfig);
            Transformer.LoadWeights(_checkpoint.Weights);
            Text = IrisZTestData.Random(new Random(12), TextLength, TextDim);
        }

        public CpuBackend Backend { get; } = new();

        public WanVideoTransformer Transformer { get; }

        public Tensor Text { get; }

        public Tensor[] RandomFrames(int count, int seed)
        {
            Random rng = new(seed);
            return Enumerable.Range(0, count)
                .Select(_ => IrisZTestData.Random(rng, 1, Channels, 1, Height, Width))
                .ToArray();
        }

        public void Dispose()
        {
            Transformer.Dispose();
            Text.Dispose();
            _checkpoint.Dispose();
            Backend.Dispose();
        }
    }
}

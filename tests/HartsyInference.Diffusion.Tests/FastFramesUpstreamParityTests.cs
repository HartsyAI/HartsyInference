using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.FastFrames;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.Video;
using HartsyInference.Video.Pipelines;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The FastFrames rollout against Apple's own code. The fixture is a tiny random-weight model (a 2-layer, dim-64
/// causal Wan, a dim-16 bridged pixel decoder, a d_a-16 refiner) built, run and saved by <c>tools/parity/dump_fastframes_reference.py</c>
/// on the CPU reference image: ml-fastframes at a65c592, torch 2.7.1 CPU, FP32 (the dump replaces upstream's hard-coded BF16 with
/// FP32 and PyTorch's own attention kernel). Noise and prompt embeddings are saved inputs. The engine side loads the same weights
/// through the Wan converter, the refiner and decoder loaders, and runs the whole rollout: three latent frames (four denoising steps
/// for the first, two for the others, the clean pass for frame 0, the refinement network for frames 1 and 2) and the decoder, then
/// compares the latents and the pixels with upstream's. Tiny random weights show the pipeline implements the same algorithm; they do
/// not show the released checkpoints give the paper's videos.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class FastFramesUpstreamParityTests(Xunit.Abstractions.ITestOutputHelper output)
{
    private const int Frames = 3, Channels = 16, Height = 8, Width = 12;

    /// <summary>The released S recipe at the fixture's latent size: the same timesteps, so the test ties to what ships.</summary>
    private static readonly FastFramesConfig Tiny = FastFramesConfig.S with { Name = "tiny", LatentHeight = Height, LatentWidth = Width };

    [Fact]
    public void TheRolloutMatchesUpstreamsFp32Run()
    {
        using TinyModel model = new();
        using Tensor prompt = PaddedPrompt(model);
        SavedNoise noise = new(model.Io["noise"], model.Io["transition"]);

        using CpuBackend backend = new();
        FastFramesPipeline pipeline = new(backend, model.Transformer, model.Refiner, model.Decoder, Tiny);
        List<FastFramesFrame> frames = pipeline.Rollout(prompt, noise, Frames).ToList();

        Assert.Equal(Frames, frames.Count);
        float[] referenceLatents = model.Io["latents"].AsReadOnlySpan<float>().ToArray();   // [3, 16, 8, 12]
        float[] referencePixels = model.Io["pixels"].AsReadOnlySpan<float>().ToArray();     // [9, 3, 64, 96]
        int frameValues = Channels * Height * Width;
        for (int f = 0; f < Frames; f++)
            AssertClose(referenceLatents.AsSpan(f * frameValues, frameValues), frames[f].Latent, $"latent frame {f}", 1e-5);

        // 12 decoded frames, the first 3 dropped (upstream's align_frames).
        float[] decoded = frames.SelectMany(fr => fr.Pixels).ToArray();
        int pixelFrameValues = 3 * 64 * 96;
        AssertClose(referencePixels, decoded.AsSpan(3 * pixelFrameValues), "pixels (9 frames)", 1e-5);
        Assert.Equal(9, pipeline.PixelFrameCount(Frames));
    }

    [Fact]
    public void GenerateFramesYieldsTheEightBitFramesWithTheLeadingOnesDropped()
    {
        using TinyModel model = new();
        using Tensor prompt = new(new TensorShape(8, 16), DType.F32);
        for (int i = 0; i < prompt.ElementCount; i++)
            prompt.AsSpan<float>()[i] = (i % 7 - 3) * 0.2f;

        using CpuBackend backend = new();
        FastFramesPipeline pipeline = new(backend, model.Transformer, model.Refiner, model.Decoder, Tiny);
        List<VideoFrame> video = pipeline.GenerateFrames(prompt, new FastFramesSeededNoise(11, Tiny), Frames).ToList();

        Assert.Equal(9, video.Count);
        Assert.All(video, f => Assert.Equal(64 * 96 * 3, f.Rgb.Length));
        Assert.Equal(Enumerable.Range(0, 9), video.Select(f => f.Index));
        Assert.Contains(video, f => f.Rgb.Distinct().Count() > 1);
        // Seeded noise is deterministic.
        List<VideoFrame> again = pipeline.GenerateFrames(prompt, new FastFramesSeededNoise(11, Tiny), Frames).ToList();
        Assert.True(video[4].Rgb.AsSpan().SequenceEqual(again[4].Rgb));
    }

    /// <summary>The constructor refuses a recipe or a component that does not fit the others, naming both sides, before any frame runs.</summary>
    [Fact]
    public void ThePipelineRefusesComponentsThatDoNotFitItsRecipe()
    {
        using TinyModel model = new();
        using CpuBackend backend = new();
        Assert.Throws<ArgumentException>(() => new FastFramesPipeline(backend, model.Transformer, model.Refiner, model.Decoder,
            Tiny with { LatentChannels = 8 }));
        Assert.Throws<ArgumentException>(() => new FastFramesPipeline(backend, model.Transformer, model.Refiner, model.Decoder,
            Tiny with { LatentHeight = 9 }));
        using FastFramesKvRefiner threeLayers = new(model.Refiner.Config with { Layers = 3 });
        Assert.Throws<ArgumentException>(() => new FastFramesPipeline(backend, model.Transformer, threeLayers, model.Decoder, Tiny));
    }

    /// <summary>The arguments are checked when the call is made, not when the consumer first asks for a frame.</summary>
    [Fact]
    public void TheArgumentsAreCheckedWhenTheCallIsMade()
    {
        using TinyModel model = new();
        using Tensor prompt = PaddedPrompt(model);
        using CpuBackend backend = new();
        FastFramesPipeline pipeline = new(backend, model.Transformer, model.Refiner, model.Decoder, Tiny);
        SavedNoise noise = new(model.Io["noise"], model.Io["transition"]);
        Assert.Throws<ArgumentOutOfRangeException>(() => pipeline.Rollout(prompt, noise, 0));
        Assert.Throws<ArgumentNullException>(() => pipeline.GenerateFrames(null!, noise, Frames));
    }

    /// <summary>A rollout cancelled after its first frame leaves nothing behind that stops the next one: the same pipeline then
    /// generates a full clip.</summary>
    [Fact]
    public void ACancelledRolloutLeavesThePipelineUsable()
    {
        using TinyModel model = new();
        using Tensor prompt = PaddedPrompt(model);
        using CpuBackend backend = new();
        FastFramesPipeline pipeline = new(backend, model.Transformer, model.Refiner, model.Decoder, Tiny);
        using CancellationTokenSource cancel = new();
        Assert.Throws<OperationCanceledException>(() => pipeline.Rollout(prompt, new FastFramesSeededNoise(11, Tiny), Frames,
            (done, _) => { if (done == 1) cancel.Cancel(); }, cancel.Token).ToList());

        List<VideoFrame> video = pipeline.GenerateFrames(prompt, new FastFramesSeededNoise(11, Tiny), Frames).ToList();
        Assert.Equal(9, video.Count);
    }

    /// <summary>A diverged rollout (non-finite pixels) is refused, not rendered as black.</summary>
    [Fact]
    public void ANonFiniteRolloutIsRefusedNotRendered()
    {
        using TinyModel model = new();
        using Tensor prompt = PaddedPrompt(model);
        using CpuBackend backend = new();
        FastFramesPipeline pipeline = new(backend, model.Transformer, model.Refiner, model.Decoder, Tiny);
        Assert.Throws<InvalidOperationException>(() => pipeline.GenerateFrames(prompt, new NaNNoise(), Frames).ToList());
    }

    private static Tensor PaddedPrompt(TinyModel model) =>
        // An owned copy: the fixture's prompt embeddings are already padded to the generator's 8 rows.
        FastFramesOps.CopyF32(model.Io["prompt_embeds"], new TensorShape(8, 16));

    private void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = 0, error = 0, maxAbs = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{what} holds a non-finite value");
            scale += (double)expected[i] * expected[i];
            double d = expected[i] - actual[i];
            error += d * d;
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
        }
        Assert.True(scale > 1e-3, $"{what}: the reference is all zeros, so the comparison proves nothing");
        double relative = Math.Sqrt(error / scale);
        output.WriteLine($"{what}: relative RMS {relative:E2}, max abs {maxAbs:E2}, reference RMS {Math.Sqrt(scale / expected.Length):E2}");
        Assert.True(relative < tolerance, $"{what}: relative RMS error {relative:E2} (max abs {maxAbs:E2}) against upstream, tolerance {tolerance:E1}");
    }

    /// <summary>The fixture's checkpoints, loaded once per test. The sources stay open until the components are gone, because a
    /// loaded component may read the checkpoint's memory.</summary>
    private sealed class TinyModel : IDisposable
    {
        private readonly CheckpointSource _generator, _refinerSource, _decoderSource, _io;

        public TinyModel()
        {
            string dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "fastframes_tiny");
            _generator = CheckpointSource.Open(Path.Combine(dir, "generator.safetensors"));
            _refinerSource = CheckpointSource.Open(Path.Combine(dir, "refiner.safetensors"));
            _decoderSource = CheckpointSource.Open(Path.Combine(dir, "decoder.safetensors"));
            _io = CheckpointSource.Open(Path.Combine(dir, "io.safetensors"));

            WanVideoConfig wan = new()
            {
                NumHeads = 2, HeadDim = 32, NumLayers = 2, FfnDim = 32, InChannels = Channels, OutChannels = Channels,
                VaeLatentChannels = Channels, TextDim = 16, FreqDim = 256, VaeSpatialCompression = 8,
            };
            Transformer = new WanVideoTransformer(wan);
            Transformer.LoadWeights(WanVideoCheckpointConverter.Convert(_generator.Weights, _generator.Header.Metadata).Transformer);
            Refiner = new FastFramesKvRefiner(FastFramesKvRefiner.DetectConfig(_refinerSource.Weights, heads: 2) with { StemHeads = 8, FrequencyDim = 256 });
            Refiner.LoadWeights(_refinerSource.Weights);
            FastFramesPixelDecoderConfig decoderConfig = FastFramesPixelDecoder.DetectConfig(_decoderSource.Weights);
            Assert.True(decoderConfig.HasBridge);
            Decoder = new FastFramesPixelDecoder(decoderConfig);
            Decoder.LoadWeights(_decoderSource.Weights);
        }

        public WanVideoTransformer Transformer { get; }

        public FastFramesKvRefiner Refiner { get; }

        public FastFramesPixelDecoder Decoder { get; }

        public IReadOnlyDictionary<string, Tensor> Io => _io.Weights;

        public void Dispose()
        {
            Decoder.Dispose();
            Refiner.Dispose();
            Transformer.Dispose();
            _io.Dispose();
            _decoderSource.Dispose();
            _refinerSource.Dispose();
            _generator.Dispose();
        }
    }

    private sealed class SavedNoise(Tensor initial, Tensor transition) : IFastFramesNoise
    {
        private readonly float[] _initial = initial.AsReadOnlySpan<float>().ToArray();       // [frames, 16, 8, 12]
        private readonly float[] _transition = transition.AsReadOnlySpan<float>().ToArray(); // [3, frames, 16, 8, 12]
        private const int FrameValues = Channels * Height * Width;

        public float[] Initial(int frame) => _initial.AsSpan(frame * FrameValues, FrameValues).ToArray();

        public float[] Transition(int step, int frame) => _transition.AsSpan((step * Frames + frame) * FrameValues, FrameValues).ToArray();
    }

    private sealed class NaNNoise : IFastFramesNoise
    {
        public float[] Initial(int frame) => Enumerable.Repeat(float.NaN, Channels * Height * Width).ToArray();

        public float[] Transition(int step, int frame) => Enumerable.Repeat(float.NaN, Channels * Height * Width).ToArray();
    }
}

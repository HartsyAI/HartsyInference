using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Memory.Vram;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.FastFrames;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.Tests.Shared;
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

    /// <summary>Block streaming must not change the clip: the tiny model on CUDA with every block resident, then with every block
    /// streamed (ForceOn and a headroom larger than the device, so the planner keeps no prefix resident). A CUDA fact: without a
    /// device it is reported as skipped.</summary>
    [CudaDeviceFact]
    [Trait("Category", "GpuIntegration")]
    public void TheStreamedRolloutMatchesTheResidentOneOnCuda()
    {
        using TinyModel model = new();
        using Tensor prompt = PaddedPrompt(model);
        SavedNoise noise = new(model.Io["noise"], model.Io["transition"]);

        using IBackend cuda = HartsyInference.Engine.BackendFactory.CreateCuda(0);
        FastFramesPipeline resident = new(cuda, model.Transformer, model.Refiner, model.Decoder, Tiny)
        {
            StreamingMode = LowVramMode.ForceOff,
        };
        List<FastFramesFrame> expected = resident.Rollout(prompt, noise, Frames).ToList();
        // The resident run left every block on the device. Free them so the streamed run starts cold and really uploads.
        cuda.FreeWeights(model.Transformer.EnumerateWeights());

        FastFramesPipeline streamed = new(cuda, model.Transformer, model.Refiner, model.Decoder, Tiny)
        {
            StreamingMode = LowVramMode.ForceOn,
            // Belt-and-braces: ForceOn keeps no prefix whatever the headroom, so this only documents that nothing is kept resident.
            StreamingHeadroomOverride = long.MaxValue / 4,
        };
        List<FastFramesFrame> actual = streamed.Rollout(prompt, noise, Frames).ToList();

        // The streamed run must have streamed every block, or the test proves nothing about streaming.
        Assert.Equal(model.Transformer.BlockCount, streamed.LastStreamedBlocks);
        Assert.Equal(Frames, actual.Count);
        for (int f = 0; f < Frames; f++)
        {
            AssertClose(expected[f].Latent, actual[f].Latent, $"streamed latent frame {f}", 1e-5);
            AssertClose(expected[f].Pixels, actual[f].Pixels, $"streamed pixels, latent frame {f}", 1e-5);
        }
    }

    /// <summary>A cached pipeline runs clip after clip: the second clip reproduces the first from the same inputs. On CPU the placement is
    /// resident-only, so this checks the bookkeeping between clips; the CUDA variants below stream.</summary>
    [Fact]
    public void TwoClipsInARowOnOnePipelineMatch()
    {
        using TinyModel model = new();
        using Tensor prompt = PaddedPrompt(model);
        SavedNoise noise = new(model.Io["noise"], model.Io["transition"]);

        using CpuBackend backend = new();
        FastFramesPipeline pipeline = new(backend, model.Transformer, model.Refiner, model.Decoder, Tiny);
        List<FastFramesFrame> first = pipeline.Rollout(prompt, noise, Frames).ToList();
        // CPU is resident-only, so a clip that stayed resident leaves the generator marked resident for the next clip.
        Assert.True(pipeline.GeneratorResident, "a resident clip must leave the generator marked resident");
        List<FastFramesFrame> second = pipeline.Rollout(prompt, noise, Frames).ToList();

        Assert.Equal(Frames, second.Count);
        for (int f = 0; f < Frames; f++)
            Assert.True(first[f].Latent.AsSpan().SequenceEqual(second[f].Latent), $"latent frame {f} changed between clips.");
    }

    /// <summary>A resident clip, then a streamed clip, on one CUDA pipeline. The residency query is the check. After the resident clip the
    /// generator's weights are on the device and the pin is marked resident. After the streamed clip no generator weight is still on the
    /// device. Without the release in ReleaseOrKeepGenerator the resident clip's weights stay bound, and this test fails.</summary>
    [CudaDeviceFact]
    [Trait("Category", "GpuIntegration")]
    public void ResidentThenStreamedClipsOnCudaReleaseTheGenerator()
    {
        using TinyModel model = new();
        using Tensor prompt = PaddedPrompt(model);
        SavedNoise noise = new(model.Io["noise"], model.Io["transition"]);

        using IBackend cuda = HartsyInference.Engine.BackendFactory.CreateCuda(0);
        FastFramesPipeline pipeline = new(cuda, model.Transformer, model.Refiner, model.Decoder, Tiny)
        {
            StreamingMode = LowVramMode.ForceOff,
        };
        List<FastFramesFrame> resident = pipeline.Rollout(prompt, noise, Frames).ToList();

        Assert.True(pipeline.GeneratorResident, "a resident clip must leave the pin resident");
        Assert.True(model.Transformer.EnumerateWeights().Any(cuda.IsWeightResident),
            "a resident clip must leave generator weights on the device");

        pipeline.StreamingMode = LowVramMode.ForceOn;
        pipeline.StreamingHeadroomOverride = long.MaxValue / 4;
        List<FastFramesFrame> streamed = pipeline.Rollout(prompt, noise, Frames).ToList();

        Assert.Equal(model.Transformer.BlockCount, pipeline.LastStreamedBlocks);
        Assert.False(pipeline.GeneratorResident, "a streamed clip must release the generator");
        Assert.False(model.Transformer.EnumerateWeights().Any(cuda.IsWeightResident),
            "no generator weight may stay on the device after a streamed clip");
        for (int f = 0; f < Frames; f++)
            AssertClose(resident[f].Latent, streamed[f].Latent, $"streamed latent frame {f}", 1e-5);
    }

    /// <summary>The CUDA rollout (resident, TF32 default GEMMs) against the CPU rollout on the same tiny model and inputs. The tolerance
    /// is the TF32 lane's: a looser bound than the exact-GEMM decoder test, because the refiner and the blocks run the default GEMMs.</summary>
    [CudaDeviceFact]
    [Trait("Category", "GpuIntegration")]
    public void TheRolloutOnCudaMatchesTheCpuRollout()
    {
        using TinyModel model = new();
        using Tensor prompt = PaddedPrompt(model);
        SavedNoise noise = new(model.Io["noise"], model.Io["transition"]);

        using CpuBackend cpu = new();
        List<FastFramesFrame> expected = new FastFramesPipeline(cpu, model.Transformer, model.Refiner, model.Decoder, Tiny)
            .Rollout(prompt, noise, Frames).ToList();

        using IBackend cuda = HartsyInference.Engine.BackendFactory.CreateCuda(0);
        FastFramesPipeline pipeline = new(cuda, model.Transformer, model.Refiner, model.Decoder, Tiny)
        {
            StreamingMode = LowVramMode.ForceOff,
        };
        List<FastFramesFrame> actual = pipeline.Rollout(prompt, noise, Frames).ToList();

        Assert.Equal(Frames, actual.Count);
        for (int f = 0; f < Frames; f++)
        {
            AssertClose(expected[f].Latent, actual[f].Latent, $"CUDA latent frame {f}", 1e-3);
            AssertClose(expected[f].Pixels, actual[f].Pixels, $"CUDA pixels, latent frame {f}", 1e-3);
        }
    }

    /// <summary>A resident clip leaves the generator marked resident. Disposing the pipeline releases it and makes a further clip refuse; a second
    /// dispose does nothing.</summary>
    [Fact]
    public void DisposingAResidentPipelineReleasesItsGenerator()
    {
        using TinyModel model = new();
        using Tensor prompt = PaddedPrompt(model);
        SavedNoise noise = new(model.Io["noise"], model.Io["transition"]);

        using CpuBackend backend = new();
        FastFramesPipeline pipeline = new(backend, model.Transformer, model.Refiner, model.Decoder, Tiny);
        pipeline.Rollout(prompt, noise, Frames).ToList();
        Assert.True(pipeline.GeneratorResident);

        pipeline.Dispose();
        Assert.False(pipeline.GeneratorResident, "disposing must release a resident generator");
        pipeline.Dispose();
        Assert.Throws<ObjectDisposedException>(() => pipeline.Rollout(prompt, noise, Frames).ToList());
    }

    /// <summary>A pipeline disposed after the first frame of a clip stops at the next frame: the enumerator refuses to continue.</summary>
    [Fact]
    public void DisposingMidClipStopsAtTheNextFrame()
    {
        using TinyModel model = new();
        using Tensor prompt = PaddedPrompt(model);
        SavedNoise noise = new(model.Io["noise"], model.Io["transition"]);

        using CpuBackend backend = new();
        FastFramesPipeline pipeline = new(backend, model.Transformer, model.Refiner, model.Decoder, Tiny);
        using IEnumerator<FastFramesFrame> clip = pipeline.Rollout(prompt, noise, Frames).GetEnumerator();
        Assert.True(clip.MoveNext());

        pipeline.Dispose();
        Assert.Throws<ObjectDisposedException>(() => clip.MoveNext());
    }

    /// <summary>A rollout created before Dispose and enumerated after it is refused, not run on a released pipeline.</summary>
    [Fact]
    public void EnumeratingARolloutAfterDisposeRefuses()
    {
        using TinyModel model = new();
        using Tensor prompt = PaddedPrompt(model);
        SavedNoise noise = new(model.Io["noise"], model.Io["transition"]);

        using CpuBackend backend = new();
        FastFramesPipeline pipeline = new(backend, model.Transformer, model.Refiner, model.Decoder, Tiny);
        IEnumerable<FastFramesFrame> clip = pipeline.Rollout(prompt, noise, Frames);
        pipeline.Dispose();

        Assert.Throws<ObjectDisposedException>(() => clip.ToList());
    }

    /// <summary>The CUDA form of the dispose test: after a resident clip some generator weight is on the device, and disposing the pipeline
    /// frees it. Without the release in Dispose this test fails.</summary>
    [CudaDeviceFact]
    [Trait("Category", "GpuIntegration")]
    public void DisposingAResidentPipelineFreesTheGeneratorOnCuda()
    {
        using TinyModel model = new();
        using Tensor prompt = PaddedPrompt(model);
        SavedNoise noise = new(model.Io["noise"], model.Io["transition"]);

        using IBackend cuda = HartsyInference.Engine.BackendFactory.CreateCuda(0);
        FastFramesPipeline pipeline = new(cuda, model.Transformer, model.Refiner, model.Decoder, Tiny)
        {
            StreamingMode = LowVramMode.ForceOff,
        };
        pipeline.Rollout(prompt, noise, Frames).ToList();
        Assert.True(model.Transformer.EnumerateWeights().Any(cuda.IsWeightResident),
            "a resident clip must leave generator weights on the device");

        pipeline.Dispose();
        Assert.False(model.Transformer.EnumerateWeights().Any(cuda.IsWeightResident),
            "disposing must free the generator's device weights");
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

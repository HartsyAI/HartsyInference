using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>The Duo session driver on the tiny synthetic model, with fake window decoders that depend on their input. The windows the video
/// decoder is handed must hold the chunks in order, so consecutive windows agree on the chunks they share; the published blocks must depend on
/// the noise; and the driver refuses a bad grid or audio length before anything is committed, and runs only once. These are self-consistency
/// checks on the driver: nothing here runs upstream code or released weights.</summary>
public sealed class DuoSessionTests
{
    private const int Hidden = 8, Heads = 2, HeadDim = 96, Inner = Heads * HeadDim, Ffn = 16, TextDim = 5, Layers = 2;
    private const int VideoIn = 96, AudioIn = 32, TimeEmbedDim = 6, TimeEmbedHidden = 8;
    private const int Shared = 3, BodyRows = 2, RefRows = 4, VideoRows = 20, AudioSteps = 3, AudioRows = 2 * AudioSteps;
    private const int Blocks = 3, SamplesPerStep = 4, Height = 4, Width = 4;

    [Fact]
    public void TheVideoWindowsHoldTheChunksInOrder()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        List<Window> windows = [];
        RunOnce(transformer, windows, seed: 0);

        // OnChunk(1), OnChunk(2) and the End window: 7, 12 and 12 latent steps.
        Assert.Equal(new[] { 7, 12, 12 }, windows.Select(w => w.Steps).ToArray());
        // Chunk 0 is steps 0-4 of the first window and of the second; chunk 1 is steps 5-6 of the first window (its first two) and steps 5-9 of
        // the second, and the End window opens on chunk 1 too.
        AssertSameSteps(windows[0], 0, windows[1], 0, 5);
        AssertSameSteps(windows[0], 5, windows[1], 5, 2);
        AssertSameSteps(windows[1], 5, windows[2], 0, 5);
    }

    [Fact]
    public void ThePublishedBlocksDependOnTheNoise()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        DuoOutput first = RunOnce(transformer, [], seed: 0);
        DuoOutput second = RunOnce(transformer, [], seed: 1);

        Assert.NotEqual(first.Video[0].Rgb, second.Video[0].Rgb);
        Assert.NotEqual(first.Audio[0], second.Audio[0]);
    }

    [Fact]
    public void ARunPublishesEveryVideoAndAudioBlockInOrder()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        DuoOutput output = RunOnce(transformer, [], seed: 0);

        Assert.Equal(Blocks, output.Video.Count);
        Assert.All(output.Video, block =>
        {
            Assert.Equal(17, block.Frames);
            Assert.Equal(Height, block.Height);
            Assert.Equal(Width, block.Width);
        });
        Assert.Equal(Blocks, output.Audio.Count);
        Assert.All(output.Audio, samples => Assert.Equal(2 * AudioSteps * SamplesPerStep, samples.Length));
    }

    [Fact]
    public void ABlockWithTheWrongGridIsRefusedBeforeAnythingIsCommitted()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        IBackend backend = TestBackends.Cpu;
        using DuoStreamCache cache = new(backend, blocks: Layers, width: Inner, capacityRows: 256);
        using DuoVideoStreamDecoder video = new(backend, window => Fake(window, []));
        using DuoAudioStreamDecoder audio = Audio(backend);
        DuoSampleInput sample = Sample() with
        {
            Blocks = Sample().Blocks.Select((b, i) => i == 1 ? b with { Video = new DuoGrid(4, Height, Width) } : b).ToArray(),
        };
        DuoSession session = new(transformer, backend, Config(), sample, DuoRopePolicy.Old, cache, video, audio);

        Assert.Throws<ArgumentException>(() => session.Run(Run(seed: 0)));
        Assert.Equal(0, cache.Length);
    }

    [Fact]
    public void ABlockWithTheWrongAudioStepsIsRefusedBeforeAnythingIsCommitted()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        IBackend backend = TestBackends.Cpu;
        using DuoStreamCache cache = new(backend, blocks: Layers, width: Inner, capacityRows: 256);
        using DuoVideoStreamDecoder video = new(backend, window => Fake(window, []));
        using DuoAudioStreamDecoder audio = Audio(backend);
        DuoSampleInput sample = Sample() with
        {
            Blocks = Sample().Blocks.Select((b, i) => i == 2 ? b with { AudioSteps = AudioSteps + 1 } : b).ToArray(),
        };
        DuoSession session = new(transformer, backend, Config(), sample, DuoRopePolicy.Old, cache, video, audio);

        Assert.Throws<ArgumentException>(() => session.Run(Run(seed: 0)));
        Assert.Equal(0, cache.Length);
    }

    [Fact]
    public void ASessionRunsOnce()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        IBackend backend = TestBackends.Cpu;
        using DuoStreamCache cache = new(backend, blocks: Layers, width: Inner, capacityRows: 256);
        using DuoVideoStreamDecoder video = new(backend, window => Fake(window, []));
        using DuoAudioStreamDecoder audio = Audio(backend);
        DuoSession session = new(transformer, backend, Config(), Sample(), DuoRopePolicy.Old, cache, video, audio);
        session.Run(Run(seed: 0));

        Assert.Throws<InvalidOperationException>(() => session.Run(Run(seed: 0)));
    }

    [Fact]
    public void ARunWithTheWrongNumberOfBlockInputsIsRefused()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        IBackend backend = TestBackends.Cpu;
        using DuoStreamCache cache = new(backend, blocks: Layers, width: Inner, capacityRows: 256);
        using DuoVideoStreamDecoder video = new(backend, window => Fake(window, []));
        using DuoAudioStreamDecoder audio = Audio(backend);
        DuoSession session = new(transformer, backend, Config(), Sample(), DuoRopePolicy.Old, cache, video, audio);
        DuoRun run = Run(seed: 0) with { BodyText = Run(seed: 0).BodyText.Take(Blocks - 1).ToArray() };

        Assert.Throws<ArgumentException>(() => session.Run(run));
    }

    /// <summary>One run on a fresh cache and fresh decoders. The windows the video decoder saw are appended to <paramref name="windows"/>.</summary>
    private static DuoOutput RunOnce(MiniMaxH3Transformer transformer, List<Window> windows, int seed)
    {
        IBackend backend = TestBackends.Cpu;
        using DuoStreamCache cache = new(backend, blocks: Layers, width: Inner, capacityRows: 256);
        using DuoVideoStreamDecoder video = new(backend, window => Fake(window, windows));
        using DuoAudioStreamDecoder audio = Audio(backend);
        DuoSession session = new(transformer, backend, Config(), Sample(), DuoRopePolicy.Old, cache, video, audio);
        return session.Run(Run(seed));
    }

    private static DuoAudioStreamDecoder Audio(IBackend backend) =>
        new(backend, FakeAudio, AudioIn, SamplesPerStep, AudioSteps, lookaheadSteps: 2, contextSteps: 4);

    /// <summary>A recorded window: its latent steps, and the values in [channel][step][height][width] order.</summary>
    private sealed record Window(int Steps, float[] Values);

    /// <summary>A fake video decoder. It records the window it was given and returns pixels that depend only on the window's values.</summary>
    private static Tensor Fake(Tensor window, List<Window> windows)
    {
        int steps = (int)window.Shape[2];
        float[] values = window.AsSpan<float>().ToArray();
        windows.Add(new Window(steps, values));
        int frames = DuoVideoDecodePlan.RawFramesFor(steps);
        float mean = values.Average();
        Tensor output = new(new TensorShape(new long[] { 1, 3, frames, Height, Width }), DType.F32);
        Span<float> o = output.AsSpan<float>();
        for (int c = 0; c < 3; c++)
        {
            for (int f = 0; f < frames; f++)
            {
                for (int p = 0; p < Height * Width; p++)
                {
                    o[((c * frames) + f) * Height * Width + p] = MathF.Tanh(mean * 8f + 0.1f * c + 0.01f * f + 0.001f * p);
                }
            }
        }
        return output;
    }

    /// <summary>A fake audio decoder: <c>[1, 2, steps · samplesPerStep]</c>, pure in its window's values.</summary>
    private static Tensor FakeAudio(Tensor window)
    {
        int steps = (int)window.Shape[3];
        float[] values = window.AsSpan<float>().ToArray();
        float mean = values.Average();
        Tensor output = new(new TensorShape(new long[] { 1, 2, steps * SamplesPerStep }), DType.F32);
        Span<float> o = output.AsSpan<float>();
        for (int s = 0; s < steps * SamplesPerStep; s++)
        {
            o[s] = MathF.Tanh(mean * 8f + 0.01f * s);
            o[(steps * SamplesPerStep) + s] = MathF.Tanh(mean * 4f - 0.01f * s);
        }
        return output;
    }

    /// <summary>Asserts that <paramref name="count"/> latent steps of two windows agree, for every channel and every height and width position.</summary>
    private static void AssertSameSteps(Window a, int aStart, Window b, int bStart, int count)
    {
        for (int c = 0; c < 24; c++)
        {
            for (int t = 0; t < count; t++)
            {
                for (int p = 0; p < Height * Width; p++)
                {
                    float left = a.Values[((c * a.Steps) + aStart + t) * Height * Width + p];
                    float right = b.Values[((c * b.Steps) + bStart + t) * Height * Width + p];
                    Assert.Equal(left, right);
                }
            }
        }
    }

    private static DuoRun Run(int seed) => new(
        SharedText: Values(2, Shared * TextDim),
        ReferenceVideo: Values(30, RefRows * VideoIn),
        BodyText: Enumerable.Range(0, Blocks).Select(b => Values(10 + b, BodyRows * TextDim)).ToArray(),
        NoiseVideo: Enumerable.Range(0, Blocks).Select(b => Values(40 + b + (100 * seed), VideoRows * VideoIn)).ToArray(),
        NoiseAudio: Enumerable.Range(0, Blocks).Select(b => Values(50 + b + (100 * seed), AudioRows * AudioIn)).ToArray(),
        Steps: 2,
        Shift: 1.0);

    private static DuoSampleInput Sample() => new()
    {
        SampleId = 4,
        SharedDescriptionRows = 2,
        SharedCueRows = 1,
        Reference = new DuoGrid(1, 4, 4),
        Blocks = Enumerable.Range(0, Blocks).Select(b => new DuoBlockInput
        {
            StartMs = 708.0 * b,
            BodyTextRows = BodyRows,
            Video = new DuoGrid(5, Height, Width),
            AudioSteps = AudioSteps,
            FrameStartsMs = Enumerable.Range(0, 5).Select(f => (708.0 * b) + (100.0 * f)).ToArray(),
        }).ToArray(),
        CleanBlocks = 0,
        NoisyBlocks = [0],
    };

    private static MiniMaxH3Config Config() => new() { LatentsDim = 24, HasAudio = true, AudioLatentsDim = AudioIn };

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
        Make(new TensorShape(rows, cols), Values(seed, rows * cols, scale));

    private static Tensor Make(TensorShape shape, float[] values)
    {
        Tensor tensor = new(shape, DType.F32);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static Tensor Ones(int count)
    {
        float[] data = new float[count];
        Array.Fill(data, 1f);
        return Make(new TensorShape(count), data);
    }

    /// <summary>Hash-based values in [-scale, scale], the generator the other Duo tests use.</summary>
    private static float[] Values(int seed, int count, float scale = 1f)
    {
        float[] data = new float[count];
        for (int i = 0; i < count; i++)
        {
            double h = Math.Sin((seed * 12.9898) + (i * 78.233)) * 43758.5453;
            double frac = h - Math.Floor(h);
            data[i] = scale * (float)((2.0 * frac) - 1.0);
        }
        return data;
    }
}

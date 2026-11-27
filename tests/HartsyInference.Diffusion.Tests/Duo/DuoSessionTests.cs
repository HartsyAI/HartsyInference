using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>The Duo session driver on the tiny synthetic model, with fake window decoders. The run publishes one video block of 17 frames and one
/// audio block per Duo block, in order, and refuses a run whose inputs do not match the sample. These are self-consistency checks on the
/// driver: nothing here runs upstream code or released weights.</summary>
public sealed class DuoSessionTests
{
    private const int Hidden = 8, Heads = 2, HeadDim = 96, Inner = Heads * HeadDim, Ffn = 16, TextDim = 5, Layers = 2;
    private const int VideoIn = 96, AudioIn = 32, TimeEmbedDim = 6, TimeEmbedHidden = 8;
    private const int Shared = 3, BodyRows = 2, RefRows = 4, VideoRows = 20, AudioSteps = 3, AudioRows = 2 * AudioSteps;
    private const int Blocks = 3, SamplesPerStep = 4;

    [Fact]
    public void ARunPublishesEveryVideoAndAudioBlockInOrder()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        IBackend backend = TestBackends.Cpu;
        using DuoStreamCache cache = new(backend, blocks: Layers, width: Inner, capacityRows: 256);
        using DuoVideoStreamDecoder video = new(backend, FakeVideo);
        using DuoAudioStreamDecoder audio = new(backend, FakeAudio, AudioIn, SamplesPerStep, AudioSteps, lookaheadSteps: 2, contextSteps: 4);
        DuoSession session = new(transformer, backend, Config(), Sample(), DuoRopePolicy.Old, cache, video, audio);

        DuoOutput output = session.Run(Run());

        Assert.Equal(Blocks, output.Video.Count);
        Assert.All(output.Video, block =>
        {
            Assert.Equal(17, block.Frames);
            Assert.Equal(4, block.Height);
            Assert.Equal(4, block.Width);
        });
        Assert.Equal(Blocks, output.Audio.Count);
        Assert.All(output.Audio, samples => Assert.Equal(2 * AudioSteps * SamplesPerStep, samples.Length));
    }

    [Fact]
    public void ARunWithTheWrongNumberOfBlockInputsIsRefused()
    {
        using MiniMaxH3Transformer transformer = Transformer();
        IBackend backend = TestBackends.Cpu;
        using DuoStreamCache cache = new(backend, blocks: Layers, width: Inner, capacityRows: 256);
        using DuoVideoStreamDecoder video = new(backend, FakeVideo);
        using DuoAudioStreamDecoder audio = new(backend, FakeAudio, AudioIn, SamplesPerStep, AudioSteps, lookaheadSteps: 2, contextSteps: 4);
        DuoSession session = new(transformer, backend, Config(), Sample(), DuoRopePolicy.Old, cache, video, audio);
        DuoRun run = Run() with { BodyText = Run().BodyText.Take(Blocks - 1).ToArray() };

        Assert.Throws<ArgumentException>(() => session.Run(run));
    }

    private static DuoRun Run() => new(
        SharedText: Values(2, Shared * TextDim),
        ReferenceVideo: Values(30, RefRows * VideoIn),
        BodyText: Enumerable.Range(0, Blocks).Select(b => Values(10 + b, BodyRows * TextDim)).ToArray(),
        NoiseVideo: Enumerable.Range(0, Blocks).Select(b => Values(40 + b, VideoRows * VideoIn)).ToArray(),
        NoiseAudio: Enumerable.Range(0, Blocks).Select(b => Values(50 + b, AudioRows * AudioIn)).ToArray(),
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
            Video = new DuoGrid(5, 4, 4),
            AudioSteps = AudioSteps,
            FrameStartsMs = Enumerable.Range(0, 5).Select(f => 708.0 * b + 100.0 * f).ToArray(),
        }).ToArray(),
        CleanBlocks = 0,
        NoisyBlocks = [0],
    };

    private static MiniMaxH3Config Config() => new() { LatentsDim = 24, HasAudio = true, AudioLatentsDim = AudioIn };

    /// <summary>A fake video decoder: a pure function of the window's shape, with the frame count the schedule needs.</summary>
    private static Tensor FakeVideo(Tensor window)
    {
        int frames = DuoVideoDecodePlan.RawFramesFor((int)window.Shape[2]);
        int h = (int)window.Shape[3], w = (int)window.Shape[4];
        Tensor output = new(new TensorShape(new long[] { 1, 3, frames, h, w }), DType.F32);
        output.AsSpan<float>().Fill(0.25f);
        return output;
    }

    /// <summary>A fake audio decoder: <c>[1, 2, steps · samplesPerStep]</c>, pure in its shape.</summary>
    private static Tensor FakeAudio(Tensor window)
    {
        int steps = (int)window.Shape[3];
        Tensor output = new(new TensorShape(new long[] { 1, 2, steps * SamplesPerStep }), DType.F32);
        output.AsSpan<float>().Fill(0.5f);
        return output;
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
            double h = Math.Sin(seed * 12.9898 + i * 78.233) * 43758.5453;
            double frac = h - Math.Floor(h);
            data[i] = scale * (float)(2.0 * frac - 1.0);
        }
        return data;
    }
}

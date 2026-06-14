using Xunit;
using Xunit.Abstractions;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Video;
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>Full end-to-end structural test for the Lance T2V pipeline: a tiny-config <see cref="LanceTransformer"/> + streaming <see cref="Wan22VaeDecoder"/> driven through <see cref="LanceVideoPipeline.GenerateFromTokens"/> on the CPU backend. Proves the video wiring (T&gt;1 grid, streaming VAE decode, per-frame output) composes and yields the right number of correctly-shaped frames. Numeric correctness vs the real checkpoint is validation-pending.</summary>
public class LanceVideoPipelineTests
{
    private readonly ITestOutputHelper _output;
    public LanceVideoPipelineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void GenerateFromTokens_TinyConfig_ProducesRequestedFrames()
    {
        CpuBackend backend = new();
        LanceConfig cfg = new()
        {
            HiddenSize = 32, NumLayers = 2, NumHeads = 4, NumKvHeads = 2,
            IntermediateSize = 64, MropeSection = (2, 1, 1), VocabSize = 64, QkNorm = false,
        };

        LanceTransformer transformer = new(cfg);
        transformer.LoadWeights(LanceSyntheticWeights.BuildTransformer(cfg));

        int[] dimMult = [1, 2, 4, 4];
        bool[] tUp = [false, true, true];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: tUp);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, tUp));

        LanceVideoPipeline pipeline = new(backend, transformer, vae, cfg);

        int[] prompt = [1, 2, 3, 4, 5];
        int[] neg = [1, 2];
        int numFrames = 5;   // (5-1) % 4 == 0 → T_lat = 2
        TextToImageRequest req = new()
        {
            Prompt = "x", NegativePrompt = "", Width = 64, Height = 64, Steps = 2, CfgScale = 4.0f, Seed = 42,
        };

        (byte[][] frames, int w, int h, int seed) = pipeline.GenerateFromTokens(prompt, neg, req, numFrames,
            p => _output.WriteLine($"step {p.Step}/{p.TotalSteps}"));

        Assert.Equal(numFrames, frames.Length);
        Assert.Equal(64, w);
        Assert.Equal(64, h);
        foreach (byte[] f in frames)
            Assert.Equal(64 * 64 * 3, f.Length);
        _ = seed;
    }

    [Fact]
    public void GenerateFromTokens_RejectsBadFrameCount()
    {
        CpuBackend backend = new();
        LanceConfig cfg = new()
        {
            HiddenSize = 32, NumLayers = 1, NumHeads = 4, NumKvHeads = 2,
            IntermediateSize = 64, MropeSection = (2, 1, 1), VocabSize = 64,
        };
        LanceTransformer transformer = new(cfg);
        transformer.LoadWeights(LanceSyntheticWeights.BuildTransformer(cfg));
        int[] dimMult = [1, 2, 4, 4];
        bool[] tUp = [false, true, true];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: tUp);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, tUp));
        LanceVideoPipeline pipeline = new(backend, transformer, vae, cfg);

        TextToImageRequest req = new() { Prompt = "x", Width = 64, Height = 64, Steps = 1, CfgScale = 4.0f, Seed = 1 };
        // num_frames = 4 → (4-1) % 4 != 0 → rejected.
        Assert.Throws<ArgumentException>(() => pipeline.GenerateFromTokens([1, 2], [1], req, 4));
    }

    [Fact]
    public async Task GenerateFramesAsync_StreamsAllFrames()
    {
        LanceVideoPipeline pipeline = BuildTinyPipeline();
        TextToImageRequest req = new() { Prompt = "x", Width = 64, Height = 64, Steps = 2, CfgScale = 4.0f, Seed = 42 };

        List<VideoFrame> frames = new();
        await foreach (VideoFrame f in pipeline.GenerateFramesAsync([1, 2, 3], [1], req, numFrames: 5))
            frames.Add(f);

        Assert.Equal(5, frames.Count);
        for (int i = 0; i < frames.Count; i++)
        {
            Assert.Equal(i, frames[i].Index);          // sequential indices
            Assert.Equal(64 * 64 * 3, frames[i].Rgb.Length);
        }
    }

    [Fact]
    public async Task BmpSequenceEncoder_WritesOneFilePerFrame()
    {
        string dir = Path.Combine(Path.GetTempPath(), "lance_vid_" + Guid.NewGuid().ToString("N"));
        try
        {
            await new BmpSequenceEncoder().EncodeAsync(SyntheticFrames(3, 8, 8), dir, fps: 24);
            Assert.Equal(3, Directory.GetFiles(dir, "frame_*.bmp").Length);
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }

    private static async IAsyncEnumerable<VideoFrame> SyntheticFrames(int n, int w, int h)
    {
        for (int i = 0; i < n; i++)
        {
            yield return new VideoFrame(i, w, h, new byte[w * h * 3]);
            await Task.Yield();
        }
    }

    private static LanceVideoPipeline BuildTinyPipeline()
    {
        CpuBackend backend = new();
        LanceConfig cfg = new()
        {
            HiddenSize = 32, NumLayers = 2, NumHeads = 4, NumKvHeads = 2,
            IntermediateSize = 64, MropeSection = (2, 1, 1), VocabSize = 64,
        };
        LanceTransformer transformer = new(cfg);
        transformer.LoadWeights(LanceSyntheticWeights.BuildTransformer(cfg));
        int[] dimMult = [1, 2, 4, 4];
        bool[] tUp = [false, true, true];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: tUp);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, tUp));
        return new LanceVideoPipeline(backend, transformer, vae, cfg);
    }
}

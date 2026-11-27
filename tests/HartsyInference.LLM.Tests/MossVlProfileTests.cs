using System.Diagnostics;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.LLM.Models.MossVl;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Vulkan;

namespace HartsyInference.LLM.Tests;

/// <summary>MOSS-VL timing on the GGUF: vision tower per frame, prefill, and decode speed as the stream holds more frames. Opt-in
/// (<c>HARTSY_MOSS_VL_PROFILE=1</c>, frame counts in <c>HARTSY_MOSS_VL_PROFILE_FRAMES</c>) besides the usual weight variables; prints a
/// table, asserts only that decode stays flat up to 32 frames (attention over vision is a small share of a step). Runs the backend as
/// the engine does for MOSS-VL (no cached widened copies of quantized weights).</summary>
[Trait("Category", "GpuBenchmark")]
public sealed class MossVlProfileTests
{
    private const string CheckpointVariable = "HARTSY_MOSS_VL_INDEX";
    private const string GgufVariable = "HARTSY_MOSS_VL_GGUF";
    private const int Steps = 32;

    [RealWeightsFact(CheckpointVariable, GgufVariable)]
    public void DecodeStaysFlatAsFramesAccumulate()
    {
        if (Environment.GetEnvironmentVariable("HARTSY_MOSS_VL_PROFILE") != "1")
            return;   // reported as passed with nothing measured; the variable opts in
        string checkpoint = Path.GetDirectoryName(Environment.GetEnvironmentVariable(CheckpointVariable)!)!;
        MossVlConfig config;
        using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(checkpoint, "config.json"))))
            config = MossVlConfig.FromJson(doc.RootElement);
        ILlmTokenizer tokenizer;
        using (FileStream stream = File.OpenRead(Path.Combine(checkpoint, "tokenizer.json")))
            tokenizer = HfTokenizerJson.LoadByteLevelBpe(stream);
        string gguf = Environment.GetEnvironmentVariable(GgufVariable)!;
        bool cuda = string.Equals(Environment.GetEnvironmentVariable("HARTSY_MOSS_VL_BACKEND"), "cuda", StringComparison.OrdinalIgnoreCase);
        using IBackend backend = cuda ? new CudaBackend(0, CudaPtxDirectory.Resolve()) : new VulkanBackend();
        // As the engine runs MOSS-VL (TextService.LoadMossVl): decode reads the packed blocks, so no widened copy is cached for prefill.
        if (!cuda) backend.CacheQuantizedWeightCasts = false;
        using MossVlVisionTower tower = MossVlVisionTower.OpenGguf(gguf, backend);
        using MossVlTextDecoder decoder = MossVlTextDecoder.OpenGguf(gguf, config, new MossVlLoadOptions { Backend = backend });
        backend.PreloadWeightGroups(decoder.Model.EnumerateWeightGroups(includeRedundantSplits: false));
        backend.PreloadWeights(tower.Encoder.EnumerateWeights());

        string dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "MossVl", "Video");
        RawRgbFrame frame = new(File.ReadAllBytes(Path.Combine(dir, "frame_002.rgb")), 256, 256);
        MossVlRealtimeProtocol protocol = new(config, tokenizer);

        // Tower: one 256x256 frame, warmed once, then timed.
        using VisionPatchGrid grid = MossVlImageProcessor.Default.Preprocess(frame);
        tower.Encoder.Forward(backend, grid.PixelValues, grid.GridT, grid.GridH, grid.GridW).MergedTokens.Dispose();
        Stopwatch towerClock = Stopwatch.StartNew();
        const int TowerReps = 5;
        for (int i = 0; i < TowerReps; i++)
            tower.Encoder.Forward(backend, grid.PixelValues, grid.GridT, grid.GridH, grid.GridW).MergedTokens.Dispose();
        backend.Sync();
        double towerMs = towerClock.Elapsed.TotalMilliseconds / TowerReps;
        Console.WriteLine($"Vision tower, 256x256 frame ({grid.MergedHeight}x{grid.MergedWidth} merged tokens): {towerMs:F1} ms/frame");

        List<(int Frames, double PrefillMs, double TokS)> rows = [];
        int[] counts = Environment.GetEnvironmentVariable("HARTSY_MOSS_VL_PROFILE_FRAMES") is { Length: > 0 } list
            ? [.. list.Split(',').Select(int.Parse)] : [1, 8, 32];
        foreach (int frames in counts)
        {
            using Tensor merged = tower.Encoder.Forward(backend, grid.PixelValues, grid.GridT, grid.GridH, grid.GridW).MergedTokens;
            List<MossVlImageFeatures> images = [.. Enumerable.Repeat(new MossVlImageFeatures(merged, grid.MergedHeight, grid.MergedWidth), frames)];
            int[] ids = tokenizer.Encode(protocol.OfflinePrompt(0, Enumerable.Range(0, frames).Select(i => (double)i).ToArray(),
                "Describe what happens in this video."), addSpecial: true);
            using MossVlDecodeSession session = new(decoder, config, backend, new MossVlSessionLimits(InitialTextCapacity: ids.Length + Steps + 8));
            SamplerChain sampler = SamplerChain.FromOptions(SamplingOptions.GreedyPreset, tokenizer, config.Text.VocabSize);
            List<int> history = [];
            Stopwatch prefill = Stopwatch.StartNew();
            int token = session.ForwardSample(ids, images, sampler, history);
            double prefillMs = prefill.Elapsed.TotalMilliseconds;
            token = session.ForwardSample([token], [], sampler, history);   // warm the step shapes
            Stopwatch decode = Stopwatch.StartNew();
            for (int s = 0; s < Steps; s++) token = session.ForwardSample([token], [], sampler, history);
            double tokS = Steps / decode.Elapsed.TotalSeconds;
            rows.Add((frames, prefillMs, tokS));
            Console.WriteLine($"{frames,3} frames ({session.VisionTokens} vision tokens, {ids.Length} prompt tokens): prefill {prefillMs:F0} ms, decode {tokS:F1} tok/s");
        }
        // Decode attends every frame through 12 cross layers; up to 32 frames of 65 tokens that is a small read next to the 48-layer
        // GEMVs. Longer streams are reported, not asserted (64 frames measured 20 tok/s on the RTX 3060).
        List<(int Frames, double PrefillMs, double TokS)> asserted = [.. rows.Where(r => r.Frames <= 32).OrderBy(r => r.Frames)];
        if (asserted.Count > 1)
        {
            (int Frames, double PrefillMs, double TokS) first = asserted[0], last = asserted[^1];
            Assert.True(last.TokS > first.TokS * 0.85,
                $"Decode slowed from {first.TokS:F1} tok/s at {first.Frames} frames to {last.TokS:F1} tok/s at {last.Frames} frames.");
        }
    }
}

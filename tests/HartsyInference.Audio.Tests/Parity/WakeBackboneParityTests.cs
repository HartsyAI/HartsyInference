using HartsyInference.Audio.Models.Wake;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.Onnx;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Numeric parity for the wake-word front-end, backbone and classifier heads against openWakeWord's
/// shipped ONNX graphs run under onnxruntime. Fixtures come from
/// <c>tests/python-reference/wake_backbone_ref.py</c>; point <c>HARTSYINFERENCE_WAKE_REF_DIR</c> at its output
/// and <c>HARTSYINFERENCE_WAKE_MODELS</c> at the wake model root to run.
///
/// <para>These cover exactly the things that fail silently: the mel front-end's undocumented constants and its
/// global dB floor, the backbone's clipped-LeakyReLU activation, and the three different head architectures
/// (one of which bundles a second network in the same file).</para></summary>
public sealed class WakeBackboneParityTests
{
    private static string? RefDir => Environment.GetEnvironmentVariable("HARTSYINFERENCE_WAKE_REF_DIR");
    private static string? ModelsDir => Environment.GetEnvironmentVariable("HARTSYINFERENCE_WAKE_MODELS");

    [Fact]
    public void MelFrontend_MatchesOnnxMelspectrogram()
    {
        if (!TryPaths(out string refDir, out string modelsDir)) return;

        float[] audio = ReadF32(Path.Combine(refDir, "wake_mel_input.bin"));
        float[] reference = ReadF32(Path.Combine(refDir, "wake_mel_output.bin"));

        using CpuBackend backend = new();
        using OnnxWeightLoader loader = new();
        loader.Load(Path.Combine(modelsDir, "backbone", "melspectrogram.onnx"));
        using WakeMelFrontend mel = new();
        mel.LoadWeights(loader.GetAllTensors());

        int frames = WakeMelFrontend.FrameCount(audio.Length);
        Assert.Equal(8, frames);
        float[] output = new float[frames * WakeMelFrontend.MelBins];
        Assert.Equal(frames, mel.Compute(backend, audio, output));

        AssertClose(reference, output, 2e-3f, "mel");
    }

    [Fact]
    public void Backbone_MatchesOnnxSpeechEmbedding()
    {
        if (!TryPaths(out string refDir, out string modelsDir)) return;

        float[] window = ReadF32(Path.Combine(refDir, "wake_embedding_input.bin"));
        float[] reference = ReadF32(Path.Combine(refDir, "wake_embedding_output.bin"));

        using CpuBackend backend = new();
        using OnnxWeightLoader loader = new();
        loader.Load(Path.Combine(modelsDir, "backbone", "embedding_model.onnx"));
        using SpeechEmbeddingModel model = new();
        model.LoadWeights(loader.GetAllTensors());

        float[] embedding = new float[SpeechEmbeddingModel.EmbeddingDim];
        model.Forward(backend, window, embedding);

        // Embeddings reach ±48, so compare relatively: 19 convolutions of float accumulation in a
        // different order than onnxruntime's will not be bit-exact.
        AssertRelative(reference, embedding, 2e-3f, "embedding");
    }

    [Fact]
    public void Heads_MatchOnnxAcrossAllThreeArchitectures()
    {
        if (!TryPaths(out string refDir, out string modelsDir)) return;

        float[] features = ReadF32(Path.Combine(refDir, "wake_head_input.bin"));
        float[] reference = ReadF32(Path.Combine(refDir, "wake_head_scores.bin"));
        string[] names = ["oww_alexa_v0.1", "oww_hey_mycroft_v0.1", "oww_hey_jarvis_v0.1"];
        Assert.Equal(names.Length, reference.Length);

        using CpuBackend backend = new();
        for (int i = 0; i < names.Length; i++)
        {
            using OnnxWeightLoader loader = new();
            loader.Load(Path.Combine(modelsDir, "heads", names[i] + ".onnx"));
            using WakeHead head = new(names[i]);
            head.LoadWeights(loader.GetAllTensors());

            float score = head.Score(backend, features);
            Assert.True(MathF.Abs(score - reference[i]) < 1e-4f,
                $"head '{names[i]}' scored {score}, reference {reference[i]}");
        }
    }

    [Fact]
    public void StreamingPipeline_MatchesReferenceScoresOverRealSpeech()
    {
        if (!TryPaths(out string refDir, out string modelsDir)) return;

        float[] pcm = ReadF32(Path.Combine(refDir, "wake_stream_input.bin"));
        float[] reference = ReadF32(Path.Combine(refDir, "wake_stream_scores.bin"));

        using CpuBackend backend = new();
        using WakeDetectionPipeline pipeline = new(backend, LoadMel(modelsDir), LoadEmbedding(modelsDir));

        // Threshold 0 with no smoothing or refractory turns every step into a reported score, which is how
        // the per-step sequence is compared; production settings are exercised by the detection tests.
        using OnnxWeightLoader headLoader = new();
        headLoader.Load(Path.Combine(modelsDir, "heads", "oww_alexa_v0.1.onnx"));
        WakeHead head = new("oww_alexa_v0.1");
        head.LoadWeights(headLoader.GetAllTensors());
        pipeline.AddWord(head, new WakeWordSettings { Threshold = 0f, SmoothingWindow = 1, RefractorySeconds = 0 });

        List<WakeDetection> detections = [];
        List<float> scores = [];
        // Push in odd-sized blocks so the pipeline's own chunk accumulation is exercised, not just
        // conveniently pre-aligned 1280-sample writes.
        for (int offset = 0; offset < pcm.Length; offset += 997)
        {
            int take = Math.Min(997, pcm.Length - offset);
            pipeline.Push(pcm.AsSpan(offset, take), detections);
            foreach (WakeDetection d in detections) scores.Add(d.Score);
        }

        Assert.Equal(reference.Length, scores.Count);
        float maxAbs = 0f;
        for (int i = 0; i < reference.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[i] - scores[i]));
        Assert.True(maxAbs < 1e-4f, $"streaming score max abs diff {maxAbs} exceeds 1e-4 over {reference.Length} steps");
    }

    [Fact]
    public void StreamingPipeline_DoesNotFireOnUnrelatedSpeech()
    {
        if (!TryPaths(out string refDir, out string modelsDir)) return;

        float[] pcm = ReadF32(Path.Combine(refDir, "wake_stream_input.bin"));
        using CpuBackend backend = new();
        using WakeDetectionPipeline pipeline = new(backend, LoadMel(modelsDir), LoadEmbedding(modelsDir));

        using OnnxWeightLoader headLoader = new();
        headLoader.Load(Path.Combine(modelsDir, "heads", "oww_alexa_v0.1.onnx"));
        WakeHead head = new("alexa");
        head.LoadWeights(headLoader.GetAllTensors());
        pipeline.AddWord(head);

        List<WakeDetection> detections = [];
        int fired = 0;
        for (int offset = 0; offset < pcm.Length; offset += WakeDetectionPipeline.ChunkSamples)
        {
            int take = Math.Min(WakeDetectionPipeline.ChunkSamples, pcm.Length - offset);
            pipeline.Push(pcm.AsSpan(offset, take), detections);
            fired += detections.Count;
        }
        Assert.Equal(0, fired);
    }

    private static WakeMelFrontend LoadMel(string modelsDir)
    {
        using OnnxWeightLoader loader = new();
        loader.Load(Path.Combine(modelsDir, "backbone", "melspectrogram.onnx"));
        WakeMelFrontend mel = new();
        mel.LoadWeights(loader.GetAllTensors());
        return mel;
    }

    private static SpeechEmbeddingModel LoadEmbedding(string modelsDir)
    {
        using OnnxWeightLoader loader = new();
        loader.Load(Path.Combine(modelsDir, "backbone", "embedding_model.onnx"));
        SpeechEmbeddingModel model = new();
        model.LoadWeights(loader.GetAllTensors());
        return model;
    }

    private static bool TryPaths(out string refDir, out string modelsDir)
    {
        refDir = RefDir ?? "";
        modelsDir = ModelsDir ?? "";
        if (refDir.Length == 0 || modelsDir.Length == 0)
        {
            Assert.True(true, "set HARTSYINFERENCE_WAKE_REF_DIR and HARTSYINFERENCE_WAKE_MODELS to run");
            return false;
        }
        return true;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxAbs = 0f;
        for (int i = 0; i < expected.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(expected[i] - actual[i]));
        Assert.True(maxAbs < tolerance, $"{what} max abs diff {maxAbs} exceeds {tolerance}");
    }

    private static void AssertRelative(float[] expected, float[] actual, float tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = expected[i] - actual[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
        }
        double rel = Math.Sqrt(num / Math.Max(den, 1e-12));
        Assert.True(rel < tolerance, $"{what} relative L2 {rel} exceeds {tolerance}");
    }

    private static float[] ReadF32(string path)
    {
        Assert.True(File.Exists(path), $"missing reference fixture {path}");
        byte[] bytes = File.ReadAllBytes(path);
        float[] f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, f.Length * 4);
        return f;
    }
}

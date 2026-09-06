using HartsyInference.Audio.Models.Wake;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Numeric parity for Silero VAD against <c>silero_vad.onnx</c>'s 16 kHz branch run under
/// onnxruntime. Fixtures come from <c>tests/python-reference/silerovad_ref.py</c>; point
/// <c>HARTSYINFERENCE_WAKE_REF_DIR</c> at its output and <c>HARTSYINFERENCE_WAKE_MODELS</c> at the wake model
/// root to run.
///
/// <para>Everything this covers fails silently: a wrong reflection pad shifts the STFT by a frame, a power
/// spectrum instead of a magnitude is still a plausible-looking feature, and dropping the ReLU between the
/// LSTM hidden state and the final convolution just moves the probabilities. All three produce a model that
/// loads, runs, and is wrong.</para></summary>
public sealed class SileroVadParityTests
{
    private static string? RefDir => Environment.GetEnvironmentVariable("HARTSYINFERENCE_WAKE_REF_DIR");
    private static string? ModelsDir => Environment.GetEnvironmentVariable("HARTSYINFERENCE_WAKE_MODELS");

    [Fact]
    public void PerChunkProbabilities_MatchOnnxOverRealSpeech()
    {
        if (!TryPaths(out string refDir, out string modelsDir)) return;

        float[] audio = ReadF32(Path.Combine(refDir, "silero_input.bin"));
        float[] reference = ReadF32(Path.Combine(refDir, "silero_probs.bin"));
        Assert.Equal(reference.Length * SileroVad.WindowSamples, audio.Length);

        using CpuBackend backend = new();
        using SileroVad vad = Load(modelsDir);

        float[] probabilities = new float[reference.Length];
        for (int i = 0; i < reference.Length; i++)
            probabilities[i] = vad.Process(backend, audio.AsSpan(i * SileroVad.WindowSamples, SileroVad.WindowSamples));

        float maxAbs = 0f;
        for (int i = 0; i < reference.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[i] - probabilities[i]));
        Assert.True(maxAbs < 1e-3f, $"per-chunk probability max abs diff {maxAbs} exceeds 1e-3 over {reference.Length} chunks");

        // jfk.wav is continuous speech, so a model stuck near zero would still pass a diff-only check if the
        // reference were ever regenerated against a broken port.
        int speech = 0;
        foreach (float p in probabilities) if (p >= 0.5f) speech++;
        Assert.True(speech > reference.Length / 2, $"only {speech} of {reference.Length} chunks scored as speech on continuous speech");
    }

    [Fact]
    public void ResetClearsState_SoTheSecondPassRepeatsTheFirst()
    {
        if (!TryPaths(out string refDir, out string modelsDir)) return;

        float[] audio = ReadF32(Path.Combine(refDir, "silero_input.bin"));
        int chunks = Math.Min(40, audio.Length / SileroVad.WindowSamples);

        using CpuBackend backend = new();
        using SileroVad vad = Load(modelsDir);

        float[] first = new float[chunks];
        for (int i = 0; i < chunks; i++)
            first[i] = vad.Process(backend, audio.AsSpan(i * SileroVad.WindowSamples, SileroVad.WindowSamples));

        vad.Reset();
        for (int i = 0; i < chunks; i++)
        {
            float again = vad.Process(backend, audio.AsSpan(i * SileroVad.WindowSamples, SileroVad.WindowSamples));
            Assert.Equal(first[i], again);
        }
    }

    [Fact]
    public void Hysteresis_SegmentsRealSpeechAndIgnoresSilence()
    {
        if (!TryPaths(out string refDir, out string modelsDir)) return;

        float[] audio = ReadF32(Path.Combine(refDir, "silero_input.bin"));
        using CpuBackend backend = new();
        using SileroVad vad = Load(modelsDir);
        SileroVadStream stream = new(vad);

        List<SileroVadSegment> segments = [];
        for (int offset = 0; offset + SileroVad.WindowSamples <= audio.Length; offset += SileroVad.WindowSamples)
            if (stream.Push(backend, audio.AsSpan(offset, SileroVad.WindowSamples), out SileroVadSegment segment))
                segments.Add(segment);
        if (stream.Flush(out SileroVadSegment tail)) segments.Add(tail);

        Assert.NotEmpty(segments);
        long speechSamples = 0;
        foreach (SileroVadSegment s in segments)
        {
            Assert.True(s.StartSample >= 0 && s.EndSample > s.StartSample, $"degenerate segment {s.StartSample}..{s.EndSample}");
            speechSamples += s.LengthSamples;
        }
        Assert.True(speechSamples > audio.Length / 2, $"segments cover {speechSamples} of {audio.Length} samples of continuous speech");

        // Digital silence must not open a segment, whatever the LSTM state was left in.
        stream.Reset();
        float[] silence = new float[SileroVad.WindowSamples];
        for (int i = 0; i < 200; i++)
            Assert.False(stream.Push(backend, silence, out _));
        Assert.False(stream.InSpeech);
    }

    /// <summary>The per-chunk verdict has to drop as soon as the speaker stops, even though the segment stays
    /// open for the stream's whole minimum-silence window.
    ///
    /// <para>A caller timing end-of-speech reads one of these two, and reading the wrong one costs it the
    /// stream's minimum silence on top of its own — the speaker waits for the sum. Both numbers are 500 ms in
    /// the wake path, so the bug is a doubling and it looks exactly like a slow transcriber.</para></summary>
    [Fact]
    public void LastChunkWasSpeech_DropsImmediately_WhileTheSegmentStaysOpen()
    {
        if (!TryPaths(out string refDir, out string modelsDir)) return;

        float[] audio = ReadF32(Path.Combine(refDir, "silero_input.bin"));
        using CpuBackend backend = new();
        using SileroVad vad = Load(modelsDir);
        SileroVadStream stream = new(vad, minSilenceMs: 500);

        int offset = 0;
        while (!stream.InSpeech && offset + SileroVad.WindowSamples <= audio.Length)
        {
            stream.Push(backend, audio.AsSpan(offset, SileroVad.WindowSamples), out _);
            offset += SileroVad.WindowSamples;
        }
        Assert.True(stream.InSpeech, "never entered speech on continuous speech");
        Assert.True(stream.LastChunkWasSpeech, "in speech, but the chunk that opened the segment was not speech");

        // 500 ms of silence at 512 samples a chunk is ~15 chunks; the segment must still be open at three.
        float[] silence = new float[SileroVad.WindowSamples];
        int quietAfter = -1;
        for (int i = 0; i < 3; i++)
        {
            stream.Push(backend, silence, out _);
            if (quietAfter < 0 && !stream.LastChunkWasSpeech) quietAfter = i + 1;
        }
        Assert.True(quietAfter > 0, "the per-chunk verdict never dropped over three chunks of digital silence");
        Assert.True(stream.InSpeech,
            "the segment closed within three chunks, so this test no longer distinguishes the two signals");
    }

    private static SileroVad Load(string modelsDir)
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(modelsDir, "vad", "silero_vad_16k.safetensors"));
        SileroVad vad = new();
        vad.LoadWeights(loader.GetAllTensors());
        return vad;
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

    private static float[] ReadF32(string path)
    {
        Assert.True(File.Exists(path), $"missing reference fixture {path}");
        byte[] bytes = File.ReadAllBytes(path);
        float[] f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, f.Length * 4);
        return f;
    }
}

using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using HartsyInference.Audio.Models.Separation;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine;
using HartsyInference.ModelAssets.CheckpointConverters;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Separation.Parity;

/// <summary>Skips unless <c>HARTSY_KIM_VOCAL2_ONNX</c> names Kim_Vocal_2.onnx and <c>HARTSY_KIM_VOCAL2_REF</c> names a folder written by
/// <c>tests/python-reference/kim_vocal2_reference/dump_kim_vocal2_reference.py</c> or, without audio-separator, <c>standin_kim_vocal2_reference.py</c>.</summary>
public sealed class KimVocal2RealFactAttribute : FactAttribute
{
    internal const string OnnxVariable = "HARTSY_KIM_VOCAL2_ONNX";
    internal const string RefVariable = "HARTSY_KIM_VOCAL2_REF";

    public KimVocal2RealFactAttribute()
    {
        string? onnx = Environment.GetEnvironmentVariable(OnnxVariable);
        string? reference = Environment.GetEnvironmentVariable(RefVariable);
        if (string.IsNullOrWhiteSpace(onnx) || !File.Exists(onnx))
        {
            Skip = $"SKIPPED: {OnnxVariable} is unset or not a file; the Kim_Vocal_2 parity run did not happen.";
        }
        else if (string.IsNullOrWhiteSpace(reference) || !File.Exists(Path.Combine(reference, "meta.json")))
        {
            Skip = $"SKIPPED: {RefVariable} is unset or holds no meta.json from the onnxruntime dump; the Kim_Vocal_2 parity run did not happen.";
        }
    }
}

/// <summary>Skips unless <c>HARTSY_KIM_VOCAL2_REF</c> holds a reference whose 16 kHz output is soxr_hq: the dumper's (librosa), or a stand-in
/// that records <c>vocals_16k_resampler</c>. Needs no weights.</summary>
public sealed class KimVocal2SoxrFactAttribute : FactAttribute
{
    public KimVocal2SoxrFactAttribute()
    {
        string? reference = Environment.GetEnvironmentVariable(KimVocal2RealFactAttribute.RefVariable);
        string meta = string.IsNullOrWhiteSpace(reference) ? "" : Path.Combine(reference, "meta.json");
        if (!File.Exists(meta))
        {
            Skip = $"SKIPPED: {KimVocal2RealFactAttribute.RefVariable} is unset or holds no meta.json; the 16 kHz comparison did not happen.";
        }
        else if (!KimVocal2ParityTests.ResampledWithSoxr(File.ReadAllText(meta)))
        {
            Skip = "SKIPPED: this reference's 16 kHz output is not soxr_hq; the 16 kHz comparison did not happen.";
        }
    }
}

/// <summary>Match of the C# separator against audio-separator 0.30.2 driving LongCat-Video <c>separate_vocals</c> on a saved clip, in two parts:
/// the network and 44.1 kHz stem (a stand-in reference is refused unless <c>HARTSY_KIM_VOCAL2_ALLOW_STANDIN=1</c>, and then says nothing about
/// audio-separator), and separately the 16 kHz resampler, fed the reference's own stem so that it does not depend on the network or the backend.</summary>
public sealed unsafe class KimVocal2ParityTests(ITestOutputHelper output)
{
    private sealed class Recorder(IMdxSpectrogramModel inner) : IMdxSpectrogramModel
    {
        public float[]? FirstSpectrogram { get; private set; }
        public float[]? FirstPrediction { get; private set; }

        public void Run(float[] spectrogram, float[] prediction)
        {
            inner.Run(spectrogram, prediction);
            FirstSpectrogram ??= (float[])spectrogram.Clone();
            FirstPrediction ??= (float[])prediction.Clone();
        }
    }

    [KimVocal2RealFact]
    [Trait("Category", "Integration")]
    public void SeparatorMatchesTheReference()
    {
        string onnx = Environment.GetEnvironmentVariable(KimVocal2RealFactAttribute.OnnxVariable)!;
        string reference = Environment.GetEnvironmentVariable(KimVocal2RealFactAttribute.RefVariable)!;
        MdxSettings settings = MdxSettings.KimVocal2;
        string meta = File.ReadAllText(Path.Combine(reference, "meta.json"));
        output.WriteLine("reference: " + meta.ReplaceLineEndings(" "));
        bool official = WrittenByTheRealDriver(meta);
        Assert.True(official || Environment.GetEnvironmentVariable("HARTSY_KIM_VOCAL2_ALLOW_STANDIN") == "1",
            "This reference was not written by dump_kim_vocal2_reference.py, so it says nothing about audio-separator or soxr_hq. "
            + "Set HARTSY_KIM_VOCAL2_ALLOW_STANDIN=1 to compare the network and stem against a stand-in.");

        (long[] mixShape, float[] mixData) = Npy.Read(Path.Combine(reference, "mix_normalized.npy"));
        int n = (int)mixShape[^1];
        float[] left = mixData.AsSpan(0, n).ToArray();
        float[] right = mixData.AsSpan(n, n).ToArray();

        string backendName = Environment.GetEnvironmentVariable("HARTSY_KIM_VOCAL2_BACKEND") ?? "cpu";
        using IBackend backend = BackendFactory.Create(backendName);
        output.WriteLine($"backend: {backendName}");
        Dictionary<string, Tensor> weights = KimVocal2OnnxConverter.Convert(onnx, out _);
        try
        {
            using KimVocal2Network network = new(weights);
            Dictionary<string, float[]> stages = [];
            network.StageProbe = (name, t) =>
            {
                if (stages.ContainsKey(name)) return;               // chunk 0 only
                stages[name] = new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();
            };
            Recorder recorder = new(new KimVocal2SpectrogramModel(backend, network, settings.Frames));
            MdxSeparator separator = new(settings, recorder);

            long started = Environment.TickCount64;
            float[][] stem = separator.Separate(left, right,
                progress: (done, total) => output.WriteLine($"chunk {done}/{total} at {(Environment.TickCount64 - started) / 1000.0:0.0}s"));

            double spec = Compare("chunk0 STFT input", reference, "chunk0_spek", recorder.FirstSpectrogram!);
            double pred = Compare("chunk0 network output", reference, "chunk0_pred", recorder.FirstPrediction!);
            double worstStage = 0;
            foreach (KeyValuePair<string, float[]> stage in stages)
            {
                worstStage = Math.Max(worstStage, Compare($"stage {stage.Key}", reference, "stage_" + stage.Key, stage.Value));
            }
            double stem0 = Compare("stem normalized, left", reference, "stem_normalized", stem[0], row: 0);
            double stem1 = Compare("stem normalized, right", reference, "stem_normalized", stem[1], row: 1);

            Assert.True(spec < 1e-5, $"STFT relRMS {spec:E2}");
            Assert.True(worstStage < 1e-4, $"worst block relRMS {worstStage:E2}");
            Assert.True(pred < 1e-4, $"network output relRMS {pred:E2}");
            Assert.True(stem0 < 1e-3 && stem1 < 1e-3, $"stem relRMS {stem0:E2} / {stem1:E2}");
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }

    [KimVocal2SoxrFact]
    [Trait("Category", "Integration")]
    public void SixteenKilohertzVocalsMatchSoxrHq()
    {
        string reference = Environment.GetEnvironmentVariable(KimVocal2RealFactAttribute.RefVariable)!;
        (long[] shape, float[] data) = Npy.Read(Path.Combine(reference, "stem_normalized.npy"));
        int n = (int)shape[^1];
        float[][] stem = [data.AsSpan(0, n).ToArray(), data.AsSpan(n, n).ToArray()];
        MdxSeparator separator = new(MdxSettings.KimVocal2, new NoModel());

        double mono = Compare("vocals 16 kHz mono (C# soxr_hq stand-in)", reference, "vocals_16k", separator.ToMono(stem, 16_000));

        // The engine polyphase resampler was 3.4e-3 from soxr_hq on the 8 s test mix, so this limit also catches a return to it.
        Assert.True(mono < 1e-3, $"16 kHz vocals relRMS {mono:E2}");
    }

    private sealed class NoModel : IMdxSpectrogramModel
    {
        public void Run(float[] spectrogram, float[] prediction) => throw new NotSupportedException("the resampler comparison runs no network");
    }

    /// <summary>Only the real dumper records the replay check and the audio-separator version.</summary>
    private static bool WrittenByTheRealDriver(string meta)
    {
        using JsonDocument document = JsonDocument.Parse(meta);
        JsonElement root = document.RootElement;
        return root.TryGetProperty("official_matches_replay", out JsonElement replay) && replay.ValueKind == JsonValueKind.True
            && root.TryGetProperty("versions", out JsonElement versions) && versions.TryGetProperty("audio_separator", out _);
    }

    internal static bool ResampledWithSoxr(string meta)
    {
        using JsonDocument document = JsonDocument.Parse(meta);
        return WrittenByTheRealDriver(meta)
            || (document.RootElement.TryGetProperty("vocals_16k_resampler", out JsonElement resampler) && resampler.GetString() == "soxr_hq");
    }

    private double Compare(string label, string reference, string file, float[] actual, int? row = null)
    {
        (long[] shape, float[] expectedAll) = Npy.Read(Path.Combine(reference, file + ".npy"));
        float[] expected = expectedAll;
        if (row is not null)
        {
            int length = (int)shape[^1];
            expected = expectedAll.AsSpan(row.Value * length, length).ToArray();
        }
        int count = Math.Min(expected.Length, actual.Length);
        if (expected.Length != actual.Length)
        {
            output.WriteLine($"{label}: length {actual.Length} vs reference {expected.Length}; comparing the first {count}");
        }
        double num = 0, den = 0, maxAbs = 0;
        for (int i = 0; i < count; i++)
        {
            double d = expected[i] - actual[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
        }
        Assert.True(den > 0, $"{label}: the reference tensor is all zero");
        double rel = Math.Sqrt(num / den);
        output.WriteLine($"{label}: relRMS {rel:E3}, max|diff| {maxAbs:E3}, n={count}");
        return rel;
    }

    /// <summary>Reads little-endian float32 C-order <c>.npy</c> files.</summary>
    private static class Npy
    {
        public static (long[] Shape, float[] Data) Read(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            int headerLength = bytes[8] | (bytes[9] << 8);
            string header = Encoding.ASCII.GetString(bytes, 10, headerLength);
            Assert.Contains("'<f4'", header);
            Assert.Contains("'fortran_order': False", header);
            Match shapeMatch = Regex.Match(header, @"'shape':\s*\(([^)]*)\)");
            long[] shape = shapeMatch.Groups[1].Value
                .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(long.Parse).ToArray();
            float[] data = new float[(bytes.Length - 10 - headerLength) / sizeof(float)];
            Buffer.BlockCopy(bytes, 10 + headerLength, data, 0, data.Length * sizeof(float));
            return (shape, data);
        }
    }
}

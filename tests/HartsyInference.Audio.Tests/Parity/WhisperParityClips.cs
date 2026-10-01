using System.Globalization;
using System.Text;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Io;
using HartsyInference.Tests.Common;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>The clips Whisper's front-end and token parity run on, built here and written as raw little-endian float32
/// so the Python reference (<c>tests/python-reference/whisper_logmel_parity/whisper_reference.py</c>) reads the very
/// samples the engine does: the narrowband variant comes from the engine's own resampler, as in
/// <c>WhisperBenchTests</c>. JFK at 16 kHz and 8 k → 16 k narrowband, whole and sliced to the gate's 2 / 5 / 10 s,
/// 5 s of digital silence, 5 s of white noise, exactly 30 s of speech (the right reflection reads real audio), and 44 s
/// (truncated to its first 30 s by both sides).</summary>
internal static class WhisperParityClips
{
    /// <summary>Directory holding <c>clips/</c>, the reference's <c>hf/</c> and each engine run's output.</summary>
    public const string DirEnvVar = "HARTSYINFERENCE_WHISPER_PARITY_DIR";

    public const int SampleRate = 16_000;
    private const int NarrowbandRate = 8_000;

    /// <summary>The JFK clip the bench and the English-only tests use.</summary>
    public static string JfkPath => Path.Combine(RepoRoot.Path, "tests", "python-reference", "silerovad_reference", "jfk.wav");

    public static IReadOnlyList<(string Name, float[] Audio)> Build()
    {
        WavFile.DecodedAudio decoded = WavFile.Read(JfkPath);
        float[] mono = decoded.ToMono();
        float[] jfk = decoded.SampleRate == SampleRate ? mono : Resampler.Create(decoded.SampleRate, SampleRate).Resample(mono);
        float[] down = Resampler.Create(SampleRate, NarrowbandRate).Resample(jfk);
        float[] up = Resampler.Create(NarrowbandRate, SampleRate).Resample(down);
        float[] narrowband = up.Length > jfk.Length ? up[..jfk.Length] : up;

        uint state = DeterministicRng.Seed(20260930);
        float[] noise = new float[5 * SampleRate];
        for (int i = 0; i < noise.Length; i++)
        {
            noise[i] = 0.1f * DeterministicRng.NextGaussian(ref state);
        }
        float[] thirty = new float[30 * SampleRate];
        for (int i = 0; i < thirty.Length; i++)
        {
            thirty[i] = jfk[i % jfk.Length];
        }

        return
        [
            ("jfk_16k", jfk),
            ("jfk_nb", narrowband),
            ("jfk_2s_16k", jfk[..(2 * SampleRate)]),
            ("jfk_5s_16k", jfk[..(5 * SampleRate)]),
            ("jfk_10s_16k", jfk[..(10 * SampleRate)]),
            ("jfk_2s_nb", narrowband[..(2 * SampleRate)]),
            ("jfk_5s_nb", narrowband[..(5 * SampleRate)]),
            ("jfk_10s_nb", narrowband[..(10 * SampleRate)]),
            ("silence_5s", new float[5 * SampleRate]),
            ("noise_5s", noise),
            ("speech_30s", thirty),
            ("speech_44s", [.. narrowband, .. jfk, .. narrowband, .. jfk]),
        ];
    }

    /// <summary>Writes <c>clips/{name}.f32</c> and the <c>clips/clips.tsv</c> manifest (name, samples).</summary>
    public static void Write(string dir, IReadOnlyList<(string Name, float[] Audio)> clips)
    {
        string clipDir = Path.Combine(dir, "clips");
        Directory.CreateDirectory(clipDir);
        StringBuilder manifest = new();
        foreach ((string name, float[] audio) in clips)
        {
            File.WriteAllBytes(Path.Combine(clipDir, name + ".f32"), MemoryMarshal.AsBytes(audio.AsSpan()).ToArray());
            manifest.Append(name).Append('\t').Append(audio.Length.ToString(CultureInfo.InvariantCulture)).Append('\n');
        }
        File.WriteAllText(Path.Combine(clipDir, "clips.tsv"), manifest.ToString());
    }

    /// <summary>Reads a raw little-endian float32 file.</summary>
    public static float[] ReadF32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, values.Length * sizeof(float));
        return values;
    }
}

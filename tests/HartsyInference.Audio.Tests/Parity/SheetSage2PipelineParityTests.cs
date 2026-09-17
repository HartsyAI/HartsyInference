using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>The whole model end to end: a recording in, the two renderings of its score out, against what the
/// released implementation produced from the same file.
///
/// <para>Every other gate in this family checks one stage against a dump of that stage. This one checks that the
/// stages compose — mel to encoder to decode to stitch to ABC — which is the only thing that can catch an error
/// living in the handover rather than in a part. It is the strictest form the check can take: the score is
/// compared as text, so a single wrong note, bar line or chord symbol fails it.</para>
///
/// <para>Gated on <c>SHEETSAGE2_CHECKPOINT</c>, plus <c>SHEETSAGE2_AUDIO</c> and <c>SHEETSAGE2_REF_DIR</c>
/// pointing at a mono 24 kHz recording and the reference rendering of it. The audio is not committed — it is
/// megabytes of waveform whose only job is to be fed to the model — so this skips unless a machine has been set
/// up for it. Regenerate the reference with <c>tests/python-reference/ref_e2e.py</c>.</para></summary>
[Trait("Category", "Integration")]
public sealed class SheetSage2PipelineParityTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _out = output;

    [Fact]
    public void Transcription_MatchesReleasedImplementation()
    {
        string? checkpoint = Environment.GetEnvironmentVariable("SHEETSAGE2_CHECKPOINT");
        string? audioPath = Environment.GetEnvironmentVariable("SHEETSAGE2_AUDIO");
        string? referenceDir = Environment.GetEnvironmentVariable("SHEETSAGE2_REF_DIR");
        if (string.IsNullOrEmpty(checkpoint) || string.IsNullOrEmpty(audioPath) || string.IsNullOrEmpty(referenceDir))
        {
            _out.WriteLine("Skipped: set SHEETSAGE2_CHECKPOINT, SHEETSAGE2_AUDIO and SHEETSAGE2_REF_DIR.");
            return;
        }

        using IBackend? backend = CreateBackend();
        if (backend is null)
        {
            _out.WriteLine("Skipped: no usable CUDA device.");
            return;
        }
        float[] audio = ReadMono24kWav(audioPath);
        // The guarded checkpoint is the one under test; going through the downloader would validate
        // whichever copy the cache happens to hold, and would reach the network on a machine set up offline.
        using SheetSage2Pipeline pipeline = SheetSage2Pipeline.LoadFrom(checkpoint);
        ScoreTranscription score = pipeline.Transcribe(backend, audio);

        _out.WriteLine($"{score.Duration:0.0}s in {score.WindowCount} window(s), truncated={score.Truncated}");
        // A whole score is too long to read in an assertion message, so on a mismatch the produced text is
        // written beside the reference and the first differing line is named.
        Compare(referenceDir, "ref_abc_melody.abc", score.MelodyAbc);
        Compare(referenceDir, "ref_abc_full.abc", score.FullAbc);
    }

    private void Compare(string referenceDir, string name, string produced)
    {
        string expected = File.ReadAllText(Path.Combine(referenceDir, name));
        if (expected == produced) return;
        string dump = Path.Combine(Path.GetTempPath(), "sheetsage2-produced-" + name);
        File.WriteAllText(dump, produced);
        string[] want = expected.Split('\n');
        string[] got = produced.Split('\n');
        int line = 0;
        while (line < want.Length && line < got.Length && want[line] == got[line]) line++;
        _out.WriteLine($"{name}: differs at line {line + 1} of {want.Length} (produced {got.Length}); wrote {dump}");
        _out.WriteLine($"  want: {(line < want.Length ? want[line] : "<eof>")}");
        _out.WriteLine($"  got : {(line < got.Length ? got[line] : "<eof>")}");
        Assert.Fail($"{name} differs at line {line + 1}; produced text written to {dump}");
    }

    /// <summary>CUDA, or skip. The encoder attends over 7,500 tokens across 24 layers, so the host path is not a
    /// slower run, it is one that does not finish. <c>SHEETSAGE2_FORCE_CPU=1</c> opts it in anyway.</summary>
    private static IBackend? CreateBackend()
    {
        if (Environment.GetEnvironmentVariable("SHEETSAGE2_FORCE_CPU") == "1")
        {
            return new CpuBackend();   // tier-lint: guarded
        }
        string ptxDirectory = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDirectory))
        {
            return null;
        }
        try
        {
            return new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDirectory);
        }
        catch (Exception)
        {
            return null;   // no usable device -- the gate skips, as a missing checkpoint does
        }
    }

    /// <summary>Reads the 16-bit mono 24 kHz WAV the model consumes, so the test feeds the same samples the
    /// reference was given rather than anything this repository resampled.</summary>
    private static float[] ReadMono24kWav(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int position = 12;
        while (position + 8 <= bytes.Length)
        {
            string id = System.Text.Encoding.ASCII.GetString(bytes, position, 4);
            int size = BitConverter.ToInt32(bytes, position + 4);
            if (id == "data")
            {
                float[] samples = new float[size / 2];
                for (int i = 0; i < samples.Length; i++)
                {
                    samples[i] = BitConverter.ToInt16(bytes, position + 8 + (i * 2)) / 32768f;
                }
                return samples;
            }
            position += 8 + size + (size & 1);
        }
        throw new InvalidDataException($"No data chunk in '{path}'.");
    }
}

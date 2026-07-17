using System.Text.Json;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Layer-by-layer parity harness for the ACE-Step ADaMoS HiFi-GAN vocoder (<see cref="AdaMosHiFiGanV1"/>): mel
/// <c>[1, melBins, T]</c> → mono waveform <c>[1, 1, T · rateProduct]</c>. Mirrors <see cref="AceStepDcaeDiffTests"/>:
/// this test GENERATES the tiny synthetic vocoder checkpoint (<see cref="AceStepSyntheticWeights.BuildVocoder"/>) + a
/// fixed mel, writes them to <c>Output/ace_step_vocoder_parity/</c>, runs <see cref="AdaMosHiFiGanV1.Decode"/> with
/// <c>ACE_STEP_DEBUG_DIR</c> set so the vocoder dumps stage / backbone-norm / conv_pre / upsample / MRF-sum / waveform
/// taps, then the Python side (<c>dump_ace_step_vocoder.py</c>) independently re-runs the vocoder math in float64 numpy
/// and <c>diff_ace_step_vocoder.py</c> compares. Validates: the ConvNeXt backbone (stem k7 conv → channels-first
/// LayerNorm → depthwise/pointwise GELU blocks + γ layer-scale, k1 channel transitions), and the HiFi-GAN head
/// (conv_pre → 2 × [SiLU + ConvTranspose1d upsample + multi-receptive-field ResBlock1 average] → SiLU → conv_post →
/// tanh).
///
/// Run order: this test (writes weights + C# dumps) → <c>python3 tests/python-reference/dump_ace_step_vocoder.py</c>
/// → <c>python3 tests/python-reference/diff_ace_step_vocoder.py</c>.</summary>
[Collection("AceStepParity")]
[Trait("Category", "SyntheticSmoke")]
public unsafe class AceStepVocoderDiffTests
{
    private readonly ITestOutputHelper _output;
    public AceStepVocoderDiffTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Vocoder_Decode_WritesParityArtifacts_Cpu()
    {
        const int MelBins = 32;
        int[] dims = [4, 6, 8, 8];
        int[] depths = [1, 1, 1, 1];
        int[] upsampleRates = [2, 2];
        int[] upsampleKernels = [4, 4];
        int[] resblockKernels = [3];
        const int T = 10;

        string root = Path.Combine(RepoRoot.Path, "Output", "ace_step_vocoder_parity");
        string inputsDir = Path.Combine(root, "inputs");
        string csDir = Path.Combine(root, "cs");
        if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        Directory.CreateDirectory(inputsDir);
        Directory.CreateDirectory(Path.Combine(csDir, "layers"));
        Environment.SetEnvironmentVariable("ACE_STEP_DEBUG_DIR", csDir);

        // ── Weights (source of truth — Python consumes the SAME file) ──
        Dictionary<string, Tensor> weights = AceStepSyntheticWeights.BuildVocoder(melBins: MelBins, dims: dims);
        SafeTensorsWriter.Save(Path.Combine(root, "vocoder.safetensors"), weights);

        // ── Deterministic fixed mel input ──
        Tensor mel = MakeTensor(new TensorShape(1, MelBins, T), seed: 13);
        WriteBin(Path.Combine(inputsDir, "mel.bin"), mel);

        var meta = new
        {
            mel_bins = MelBins,
            dims,
            depths,
            upsample_rates = upsampleRates,
            upsample_kernels = upsampleKernels,
            resblock_kernels = resblockKernels,
            t = T,
        };
        File.WriteAllText(Path.Combine(inputsDir, "meta.json"), JsonSerializer.Serialize(meta));

        // ── Run the vocoder decode (dumps voc.stage.*, voc.backbone_norm, voc.conv_pre, voc.upsample.*, voc.mrf.*, voc.waveform) ──
        AdaMosHiFiGanV1 vocoder = new(depths: depths, dims: dims,
            upsampleRates: upsampleRates, upsampleKernels: upsampleKernels, resblockKernels: resblockKernels);
        vocoder.LoadWeights(weights);
        using CpuBackend backend = new();
        Tensor wav = vocoder.Decode(backend, mel);

        int totalUp = upsampleRates.Aggregate(1, (a, r) => a * r);
        _output.WriteLine($"Vocoder decode done: wav shape={wav.Shape} (expect [1,1,{T * totalUp}])");
        Assert.Equal(1, (int)wav.Shape[0]);
        Assert.Equal(1, (int)wav.Shape[1]);
        Assert.Equal(T * totalUp, (int)wav.Shape[2]);

        wav.Dispose();
        mel.Dispose();
        Environment.SetEnvironmentVariable("ACE_STEP_DEBUG_DIR", null);

        Assert.True(File.Exists(Path.Combine(csDir, "layers", "voc_waveform.bin")),
            "voc.waveform not dumped — ACE_STEP_DEBUG_DIR may have been resolved before this test set it.");
        _output.WriteLine($"Artifacts written to {root}.");

        // ── Shell out to the Python reference + diff and assert parity. ──
        RunPython(Path.Combine("tests", "python-reference", "dump_ace_step_vocoder.py"));
        string diff = RunPython(Path.Combine("tests", "python-reference", "diff_ace_step_vocoder.py"));
        _output.WriteLine(diff);
        Assert.DoesNotContain("FIRST DIVERGENCE", diff);
        Assert.Contains("PASS", diff);
    }

    private string RunPython(string relScript)
    {
        System.Diagnostics.ProcessStartInfo psi = new("/usr/bin/python3", relScript)
        {
            WorkingDirectory = RepoRoot.Path,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        using System.Diagnostics.Process proc = System.Diagnostics.Process.Start(psi)!;
        string stdout = proc.StandardOutput.ReadToEnd();
        string stderr = proc.StandardError.ReadToEnd();
        proc.WaitForExit();
        if (proc.ExitCode != 0)
            throw new Exception($"python3 {relScript} exited {proc.ExitCode}:\n{stdout}\n{stderr}");
        return stdout + stderr;
    }

    /// <summary>Deterministic small-magnitude tensor (Python reads the written bin, not this formula).</summary>
    private static Tensor MakeTensor(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        long n = shape.ElementCount;
        uint s = (uint)(seed * 2654435761u + 1u);
        for (long i = 0; i < n; i++)
        {
            s = s * 1664525u + 1013904223u;
            p[i] = ((s >> 8) / (float)(1 << 24) - 0.5f) * 2.0f; // ~U(-1,1)
        }
        return t;
    }

    private static void WriteBin(string path, Tensor t)
    {
        long bytes = t.Shape.ElementCount * sizeof(float);
        byte[] buf = new byte[bytes];
        fixed (byte* dst = buf) Buffer.MemoryCopy((void*)t.DataPointer, dst, bytes, bytes);
        File.WriteAllBytes(path, buf);
    }
}

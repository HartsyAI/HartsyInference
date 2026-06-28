using System.Text.Json;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Layer-by-layer parity harness for the ACE-Step Music-DCAE decoder (latent <c>[1,8,H,W]</c> → mel
/// <c>[1,2,H·8,W·8]</c>). Mirrors <see cref="AceStepDitDiffTests"/>: this test GENERATES the tiny synthetic DCAE
/// checkpoint (<see cref="AceStepSyntheticWeights.BuildDcaeDecoder"/>) + a fixed latent, writes them to
/// <c>Output/ace_step_dcae_parity/</c>, runs <see cref="MusicDcaeDecoder.Decode"/> with <c>ACE_STEP_DEBUG_DIR</c> set so
/// the decoder dumps conv_in / per-stage / upsample / norm_out / conv_out taps, then the Python side
/// (<c>dump_ace_step_dcae.py</c>) independently re-runs the DCAE math in float64 numpy and
/// <c>diff_ace_step_dcae.py</c> compares. Validates the repeat-interleave conv_in shortcut, ResBlock2d (3×3 conv →
/// SiLU → 3×3 → channel-RMSNorm → residual), the EfficientViT ReLU-linear attention + GLUMBConv, the
/// nearest-upsample + pixel-shuffle shortcut, and the final RMSNorm → ReLU → conv_out.
///
/// Run order: this test (writes weights + C# dumps) → <c>python3 tests/python-reference/dump_ace_step_dcae.py</c>
/// → <c>python3 tests/python-reference/diff_ace_step_dcae.py</c>.</summary>
[Collection("AceStepParity")]
public unsafe class AceStepDcaeDiffTests
{
    private readonly ITestOutputHelper _output;
    public AceStepDcaeDiffTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Dcae_Decode_WritesParityArtifacts_Cpu()
    {
        const int LatentChannels = 8;
        const int OutChannels = 2;
        const int GlumbHidden = 8;
        const int HeadDim = 4;   // deepest stage = 32 channels → 8 heads of 4 (must divide dims[^1])
        int[] blockOut = [4, 8, 16, 32];
        int[] layersPerBlock = [1, 1, 1, 1];
        const int Hlat = 4, Wlat = 6;

        string root = Path.Combine(RepoRoot.Path, "Output", "ace_step_dcae_parity");
        string inputsDir = Path.Combine(root, "inputs");
        string csDir = Path.Combine(root, "cs");
        if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        Directory.CreateDirectory(inputsDir);
        Directory.CreateDirectory(Path.Combine(csDir, "layers"));
        Environment.SetEnvironmentVariable("ACE_STEP_DEBUG_DIR", csDir);

        // ── Weights (source of truth — Python consumes the SAME file) ──
        Dictionary<string, Tensor> weights = AceStepSyntheticWeights.BuildDcaeDecoder(
            latentChannels: LatentChannels, blockOut: blockOut, glumbHidden: GlumbHidden);
        SafeTensorsWriter.Save(Path.Combine(root, "decoder.safetensors"), weights);

        // ── Deterministic fixed latent input ──
        Tensor latent = MakeTensor(new TensorShape(1, LatentChannels, Hlat, Wlat), seed: 7);
        WriteBin(Path.Combine(inputsDir, "latent.bin"), latent);

        var meta = new
        {
            latent_channels = LatentChannels,
            out_channels = OutChannels,
            block_out = blockOut,
            layers_per_block = layersPerBlock,
            glumb_hidden = GlumbHidden,
            head_dim = HeadDim,
            attention_stages = 1,
            h_lat = Hlat,
            w_lat = Wlat,
        };
        File.WriteAllText(Path.Combine(inputsDir, "meta.json"), JsonSerializer.Serialize(meta));

        // ── Run the DCAE decode (dumps conv_in, stage.*, upsample.*, norm_out, conv_out) ──
        MusicDcaeDecoder dcae = new(latentChannels: LatentChannels, outChannels: OutChannels,
            blockOutChannels: blockOut, layersPerBlock: layersPerBlock, attentionStages: 1, attentionHeadDim: HeadDim);
        dcae.LoadWeights(weights);
        using CpuBackend backend = new();
        Tensor mel = dcae.Decode(backend, latent);

        _output.WriteLine($"DCAE decode done: mel shape={mel.Shape} (expect [1,{OutChannels},{Hlat * 8},{Wlat * 8}])");
        Assert.Equal(1, (int)mel.Shape[0]);
        Assert.Equal(OutChannels, (int)mel.Shape[1]);
        Assert.Equal(Hlat * 8, (int)mel.Shape[2]);
        Assert.Equal(Wlat * 8, (int)mel.Shape[3]);

        mel.Dispose();
        latent.Dispose();
        Environment.SetEnvironmentVariable("ACE_STEP_DEBUG_DIR", null);

        Assert.True(File.Exists(Path.Combine(csDir, "layers", "dcae_conv_out.bin")),
            "dcae.conv_out not dumped — ACE_STEP_DEBUG_DIR may have been resolved before this test set it.");
        _output.WriteLine($"Artifacts written to {root}.");

        // ── Shell out to the Python reference + diff and assert parity. ──
        RunPython(Path.Combine("tests", "python-reference", "dump_ace_step_dcae.py"));
        string diff = RunPython(Path.Combine("tests", "python-reference", "diff_ace_step_dcae.py"));
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

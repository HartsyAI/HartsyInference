using System.Text.Json;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Layer-by-layer parity harness for the ACE-Step v1 DiT core (latent + precomputed context → velocity).
/// Self-contained and CPU-only: this test GENERATES the tiny synthetic checkpoint (<see cref="AceStepSyntheticWeights"/>)
/// + fixed inputs, writes them to <c>Output/ace_step_dit_parity/</c>, and runs <see cref="AceStepDit"/> with
/// <c>ACE_STEP_DEBUG_DIR</c> set. The Python side (<c>dump_ace_step_dit.py</c>) loads the SAME weights+inputs, runs an
/// independent PyTorch re-implementation of the DiT forward, and <c>diff_ace_step_layers.py</c> compares. Validates the
/// LiteLA linear-attention, the interleaved-pair/half-concat RoPE quirk, GLUMBConv, cross-attn, AdaLN-6, patch-embed
/// and final layer — the documented first-run hotspots.
///
/// Run order: this test (writes weights + C# dumps) → <c>python3 tests/python-reference/dump_ace_step_dit.py</c>
/// → <c>python3 tests/python-reference/diff_ace_step_layers.py</c>. (Lyric Conformer + BuildContext are validated
/// separately; here the context is fed directly so the DiT math is isolated.)</summary>
[Collection("AceStepParity")]
public unsafe class AceStepDitDiffTests
{
    private readonly ITestOutputHelper _output;
    public AceStepDitDiffTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Dit_Forward_WritesParityArtifacts_Cpu()
    {
        const int F = 6;   // latent time frames
        const int L = 5;   // context tokens
        const float Timestep = 0.7f;

        AceStepConfig cfg = AceStepSyntheticWeights.TinyConfig;
        int dim = cfg.InnerDim;

        string root = Path.Combine(RepoRoot.Path, "Output", "ace_step_dit_parity");
        string inputsDir = Path.Combine(root, "inputs");
        string csDir = Path.Combine(root, "cs");
        if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        Directory.CreateDirectory(inputsDir);
        Directory.CreateDirectory(Path.Combine(csDir, "layers"));
        Environment.SetEnvironmentVariable("ACE_STEP_DEBUG_DIR", csDir);

        // ── Weights (source of truth — Python consumes the SAME file) ──
        Dictionary<string, Tensor> weights = AceStepSyntheticWeights.BuildDit(cfg);
        SafeTensorsWriter.Save(Path.Combine(root, "transformer.safetensors"), weights);

        // ── Deterministic fixed inputs ──
        Tensor latent = MakeTensor(new TensorShape(1, cfg.InChannels, cfg.LatentHeight, F), seed: 1);
        Tensor context = MakeTensor(new TensorShape(L, dim), seed: 2);
        WriteBin(Path.Combine(inputsDir, "latent.bin"), latent);
        WriteBin(Path.Combine(inputsDir, "context.bin"), context);
        File.WriteAllBytes(Path.Combine(inputsDir, "timestep.bin"), BitConverter.GetBytes(Timestep));

        var meta = new
        {
            in_channels = cfg.InChannels,
            num_heads = cfg.NumHeads,
            head_dim = cfg.HeadDim,
            inner_dim = dim,
            num_layers = cfg.NumLayers,
            mlp_ratio = cfg.MlpRatio,
            patch_h = cfg.PatchSize.H,
            patch_w = cfg.PatchSize.W,
            latent_height = cfg.LatentHeight,
            freq_dim = cfg.FreqDim,
            rope_theta = cfg.RopeTheta,
            ff_hidden = (int)(dim * cfg.MlpRatio),
            F,
            L,
            timestep = Timestep,
        };
        File.WriteAllText(Path.Combine(inputsDir, "meta.json"), JsonSerializer.Serialize(meta));

        // ── Run the DiT (dumps patch_embed, tblock, layers.i, output_velocity) ──
        using AceStepDit dit = new(cfg);
        dit.LoadWeights(weights);
        using CpuBackend backend = new();
        Tensor velocity = dit.Forward(backend, latent, context, Timestep);

        _output.WriteLine($"DiT forward done: velocity shape={velocity.Shape} (expect [1,{cfg.InChannels},{cfg.LatentHeight},{F}])");
        Assert.Equal(1, (int)velocity.Shape[0]);
        Assert.Equal(cfg.InChannels, (int)velocity.Shape[1]);
        Assert.Equal(cfg.LatentHeight, (int)velocity.Shape[2]);
        Assert.Equal(F, (int)velocity.Shape[3]);

        velocity.Dispose();
        latent.Dispose();
        context.Dispose();
        Environment.SetEnvironmentVariable("ACE_STEP_DEBUG_DIR", null);

        Assert.True(File.Exists(Path.Combine(csDir, "output_velocity.bin")), "output_velocity not dumped — ACE_STEP_DEBUG_DIR may have been resolved before this test set it.");
        _output.WriteLine($"Artifacts written to {root}.");
        _output.WriteLine("Next: python3 tests/python-reference/dump_ace_step_dit.py && python3 tests/python-reference/diff_ace_step_layers.py");
    }

    /// <summary>Deterministic small-magnitude tensor (same formula is NOT needed in Python — it reads the written bin).</summary>
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

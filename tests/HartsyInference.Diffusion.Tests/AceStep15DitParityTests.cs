using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight numeric parity for the ACE-Step v1.5 turbo (2B) DiT + condition encoder, against a standalone
/// f32 PyTorch oracle (<c>tests/python-reference/dump_acestep15_reference.py</c>, logic copied verbatim from
/// <c>modeling_acestep_v15_turbo.py</c>). The oracle loads the Comfy-Org turbo safetensors (bf16→f32), runs fixed
/// inputs through <c>condition_encode</c>, a single <c>dit_velocity</c> forward, and the full 8-step shift-3 turbo
/// Euler loop, and dumps IO to <c>tests/python-reference/acestep15_ref/*.bin</c> (vendored). This test runs the SAME
/// fixed inputs through the C# CPU F32 path (<see cref="AceStep15ConditionEncoder"/> + <see cref="AceStep15Dit"/>)
/// and diffs.
///
/// Verified components (CPU F32 vs f32 oracle): condition encoder maxAbs 1.9e-6 · DiT velocity maxAbs 5.1e-6
/// (corr 1.0) · full 8-step loop final latent maxAbs 6.6e-6 (corr 1.0). The turbo scheduler
/// (<see cref="AceStep15Config.GetTimesteps"/> + the Euler update) is bit-identical to the oracle by construction.
///
/// GATED on the real turbo checkpoint (it is not in CI). Set <c>ACESTEP15_DIT</c> to the safetensors path (defaults
/// to the local hartsyinference cache). To regenerate the reference bins: install torch + run
/// <c>TORCHDYNAMO_DISABLE=1 python3 tests/python-reference/dump_acestep15_reference.py</c> (it writes to
/// <c>/tmp/ace15ref</c>; copy <c>*.bin</c> into <c>tests/python-reference/acestep15_ref</c>).</summary>
public unsafe class AceStep15DitParityTests
{
    private readonly ITestOutputHelper _output;
    public AceStep15DitParityTests(ITestOutputHelper output) => _output = output;

    private static string? CheckpointPath()
    {
        string env = Environment.GetEnvironmentVariable("ACESTEP15_DIT") ?? "";
        if (env.Length > 0 && File.Exists(env)) return env;
        string def = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "hartsyinference", "models",
            "Comfy-Org--ace_step_1.5_ComfyUI_files", "split_files", "diffusion_models",
            "acestep_v1.5_turbo.safetensors");
        return File.Exists(def) ? def : null;
    }

    private static string RefDir() =>
        Path.Combine(RepoRoot.Path, "tests", "python-reference", "acestep15_ref");

    [Fact]
    public void ConditionEncoder_And_Dit_MatchTorchOracle_Cpu()
    {
        string? ckpt = CheckpointPath();
        if (ckpt is null)
        {
            _output.WriteLine("SKIP: ACE-Step 1.5 turbo checkpoint not found (set ACESTEP15_DIT). " +
                "This is a real-weight parity test and is not run in CI.");
            return;
        }
        string refDir = RefDir();
        Assert.True(File.Exists(Path.Combine(refDir, "cond.bin")),
            $"reference dumps missing in {refDir}; regenerate via dump_acestep15_reference.py.");

        CpuBackend backend = new();
        AceStep15Config cfg = new();
        (Dictionary<string, Tensor> weights, HartsyInference.ModelAssets.SafeTensors.SafeTensorsLoader loader) =
            AceStepCheckpointConverter.LoadModel15(ckpt, castToF32: true);
        AceStep15Dit dit = new(cfg); dit.LoadWeights(weights);
        AceStep15ConditionEncoder encoder = new(cfg); encoder.LoadWeights(weights);

        // ── Condition encoder ──
        Tensor textHidden = LoadBin(refDir, "text_hidden");
        Tensor lyricHidden = LoadBin(refDir, "lyric_hidden");
        Tensor condRef = LoadBin(refDir, "cond");
        Tensor cond = encoder.EncodeConditions(backend, textHidden, lyricHidden, null);
        (double cm, double cc) = Diff(cond, condRef);
        _output.WriteLine($"CONDITION ENCODER: maxAbs={cm:E3} corr={cc:F8}");
        Assert.Equal(condRef.Shape, cond.Shape);
        Assert.True(cm < 5e-3, $"condition encoder maxAbs {cm:E3} exceeds 5e-3");
        Assert.True(cc > 0.9999, $"condition encoder corr {cc:F6} below 0.9999");

        // ── DiT velocity (single forward) ──
        Tensor noisy = LoadBin(refDir, "noisy");
        Tensor velRef = LoadBin(refDir, "velocity");
        Tensor context = BuildContext(LoadBin(refDir, "src"), LoadBin(refDir, "chunk"));
        Tensor vel = dit.Forward(backend, noisy, context, condRef, 0.75f, 0.75f);
        (double vm, double vc) = Diff(vel, velRef);
        _output.WriteLine($"DiT VELOCITY (ref-cond): maxAbs={vm:E3} corr={vc:F8}");
        Assert.Equal(velRef.Shape, vel.Shape);
        Assert.True(vm < 5e-3, $"DiT velocity maxAbs {vm:E3} exceeds 5e-3");
        Assert.True(vc > 0.9999, $"DiT velocity corr {vc:F6} below 0.9999");

        // ── Full 8-step turbo (shift-3) Euler loop ──
        Tensor lText = LoadBin(refDir, "loop_text_hidden");
        Tensor lLyric = LoadBin(refDir, "loop_lyric_hidden");
        Tensor lCond = encoder.EncodeConditions(backend, lText, lLyric, null);
        Tensor lCtx = BuildContext(LoadBin(refDir, "loop_src"), LoadBin(refDir, "loop_chunk"));
        Tensor lNoise = LoadBin(refDir, "loop_noise");
        Tensor lFinalRef = LoadBin(refDir, "loop_final");
        int lt = (int)lNoise.Shape[1];
        float[] sched = AceStep15Config.GetTimesteps(3f);
        Tensor xt = new Tensor(new TensorShape(1, lt, 64), DType.F32);
        Buffer.MemoryCopy((float*)lNoise.DataPointer, (float*)xt.DataPointer, (long)lt * 64 * 4, (long)lt * 64 * 4);
        float* xtp = (float*)xt.DataPointer;
        for (int i = 0; i < sched.Length; i++)
        {
            float tc = sched[i];
            float next = i < sched.Length - 1 ? sched[i + 1] : 0f;
            Tensor vt = dit.Forward(backend, xt, lCtx, lCond, tc, tc);
            float* vtp = (float*)vt.DataPointer;
            float ddt = next - tc;
            for (long e = 0; e < xt.Shape.ElementCount; e++) xtp[e] += ddt * vtp[e];
            vt.Dispose();
        }
        (double lm, double lc) = Diff(xt, lFinalRef);
        _output.WriteLine($"FULL 8-STEP LOOP (final latent): maxAbs={lm:E3} corr={lc:F8}");
        Assert.True(lm < 5e-3, $"full-loop final latent maxAbs {lm:E3} exceeds 5e-3");
        Assert.True(lc > 0.9999, $"full-loop final latent corr {lc:F6} below 0.9999");

        cond.Dispose(); vel.Dispose(); xt.Dispose();
        textHidden.Dispose(); lyricHidden.Dispose(); condRef.Dispose(); noisy.Dispose();
        velRef.Dispose(); context.Dispose(); lText.Dispose(); lLyric.Dispose(); lCond.Dispose();
        lCtx.Dispose(); lNoise.Dispose(); lFinalRef.Dispose();
        dit.Dispose();
        GC.KeepAlive(loader);
    }

    /// <summary>context = [src ‖ chunk] along channels → [1, T, 128].</summary>
    private static Tensor BuildContext(Tensor src, Tensor chunk)
    {
        int t = (int)src.Shape[1];
        Tensor ctx = new Tensor(new TensorShape(1, t, 128), DType.F32);
        float* cp = (float*)ctx.DataPointer;
        float* sp = (float*)src.DataPointer;
        float* kp = (float*)chunk.DataPointer;
        for (int i = 0; i < t; i++)
        {
            for (int c = 0; c < 64; c++) cp[i * 128 + c] = sp[i * 64 + c];
            for (int c = 0; c < 64; c++) cp[i * 128 + 64 + c] = kp[i * 64 + c];
        }
        src.Dispose(); chunk.Dispose();
        return ctx;
    }

    /// <summary>Reads an int32 rank + dims header followed by little-endian f32 data (the oracle's bin format).</summary>
    private static Tensor LoadBin(string dir, string name)
    {
        byte[] raw = File.ReadAllBytes(Path.Combine(dir, name + ".bin"));
        int o = 0;
        int rank = BitConverter.ToInt32(raw, o); o += 4;
        long[] shape = new long[rank];
        for (int i = 0; i < rank; i++) { shape[i] = BitConverter.ToInt32(raw, o); o += 4; }
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) { p[i] = BitConverter.ToSingle(raw, o); o += 4; }
        return t;
    }

    private static (double MaxAbs, double Corr) Diff(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer; float* pb = (float*)b.DataPointer;
        long n = a.Shape.ElementCount;
        double mx = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (long i = 0; i < n; i++)
        {
            double x = pa[i], y = pb[i];
            mx = Math.Max(mx, Math.Abs(x - y));
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
        }
        double cov = sab / n - (sa / n) * (sb / n);
        double va = saa / n - (sa / n) * (sa / n);
        double vb = sbb / n - (sb / n) * (sb / n);
        return (mx, cov / Math.Sqrt(va * vb + 1e-30));
    }
}
